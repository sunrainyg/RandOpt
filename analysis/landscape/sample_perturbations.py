#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from tasks import grade, load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local path, e.g. Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--sigma", type=float, default=0.001)
    ap.add_argument("--seeds", default="1-1000", help="first-last")
    ap.add_argument("--base-runs", type=int, default=64, help="number of unperturbed evaluations (pooled base)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--repetition-penalty", type=float, default=1.05, help="1.05 = Qwen2.5 generation_config")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    ap.add_argument("--out-dir", default="landscape_results")
    args = ap.parse_args()

    out_dir = (Path(args.out_dir) / Path(args.model.rstrip("/")).name /
               f"sigma{args.sigma:g}_rp{args.repetition_penalty:g}_len{args.max_tokens}")
    out_dir.mkdir(parents=True, exist_ok=True)
    first, last = map(int, args.seeds.split("-"))
    jobs = [out_dir / f"base{k:03d}.json" for k in range(args.base_runs)] + \
           [out_dir / f"seed{s:05d}.json" for s in range(first, last + 1)]
    jobs = [f for f in jobs[args.shard::args.num_shards] if not f.exists()]
    print(f"{out_dir}: {len(jobs)} evaluations to do (shard {args.shard}/{args.num_shards})", flush=True)
    if not jobs:
        return

    data = load_tasks()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp, gpu_memory_utilization=0.85,
              max_model_len=4096, enable_prefix_caching=False, seed=0, worker_extension_cls="perturb_ext.PerturbExt")
    llm.collective_rpc("snapshot")
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty)
    text = data.pop("nll")
    prompts = {t: [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True) for r in rows]
               for t, rows in data.items()}
    ids = tok(text, add_special_tokens=False).input_ids
    nll_chunks = [{"prompt_token_ids": ids[i * 1024:(i + 1) * 1024]} for i in range(min(32, len(ids) // 1024))]
    sp_nll = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    def evaluate():
        res = {}
        for t, ps in prompts.items():
            texts = [o.outputs[0].text for o in llm.generate(ps, sp, use_tqdm=False)]
            correct = [int(grade(t, x, r)) for x, r in zip(texts, data[t])]
            res[t] = {"acc": sum(correct) / len(correct), "n": len(correct), "correct": correct}
        lps = [lp for o in llm.generate(nll_chunks, sp_nll, use_tqdm=False) for lp in o.prompt_logprobs[1:] if lp]
        res["nll"] = {"nll": -sum(next(iter(lp.values())).logprob for lp in lps) / len(lps)}
        return res

    for f in jobs:
        t0 = time.time()
        perturbed = f.name.startswith("seed")
        rec = {"model": args.model, "sigma": args.sigma if perturbed else 0.0,
               "repetition_penalty": args.repetition_penalty, "max_tokens": args.max_tokens}
        if perturbed:
            rec["seed"] = int(f.stem[4:])
            parts = llm.collective_rpc("perturb", args=(rec["seed"], args.sigma))  # one 2D projection per TP rank
            rec["proj"] = [sum(p[k] for p in parts) / len(parts) ** 0.5 for k in range(2)]
        rec["tasks"] = evaluate()
        if perturbed:
            llm.collective_rpc("restore")
        rec["sec"] = round(time.time() - t0, 1)
        f.with_suffix(".tmp").write_text(json.dumps(rec))
        os.replace(f.with_suffix(".tmp"), f)
        print(f"{f.name}: " + " ".join(f"{t}={v.get('acc', v.get('nll')):.3f}" for t, v in rec["tasks"].items())
              + f" ({rec['sec']}s)", flush=True)


if __name__ == "__main__":
    main()
