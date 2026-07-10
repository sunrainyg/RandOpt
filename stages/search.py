"""Weight-space search stage for iterative RandOpt
"""
import argparse
import json
import os
import time

import numpy as np
import ray
from transformers import AutoTokenizer
from vllm import SamplingParams

from ..data import get_dataset_handler, list_datasets
from ..engine import launch_engines, cleanup_engines


def parse_args():
    p = argparse.ArgumentParser(description="Weight-space search: perturb -> score -> select top-k")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=list_datasets())
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--model_name", type=str, required=True,
                   help="Current model θ to search around (base_model_path for seeds.json)")
    p.add_argument("--seeds_out", type=str, required=True,
                   help="Where to write {base_model_path, top_k_models:[{seed,sigma}]}")
    p.add_argument("--population", type=int, default=30,
                   help="Number of Gaussian perturbations to draw and score this round")
    p.add_argument("--sigma", type=float, default=1e-3,
                   help="Perturbation scale θ -> θ + sigma * noise")
    p.add_argument("--top_k", type=int, default=8,
                   help="Keep the top-k highest-scoring perturbations for distillation")
    p.add_argument("--score_samples", type=int, default=256,
                   help="Questions used to score each perturbed model (greedy)")
    p.add_argument("--score_temperature", type=float, default=0.0,
                   help="Sampling temperature for scoring (0 = greedy)")
    p.add_argument("--include_base", action="store_true",
                   help="Also score the unperturbed model (reference only; never selected)")
    p.add_argument("--precision", type=str, default="bfloat16")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--num_engines", type=int, default=8)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--cuda_devices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--global_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens

    data_path = args.data_path or handler.default_train_path
    datas = handler.load_data(data_path, split="train", max_samples=args.score_samples)
    print(f"[search] scoring on {len(datas)} questions from {data_path}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    is_instruct = any(x in args.model_name.lower() for x in ["instruct", "chat", "it"])

    def fmt_prompt(messages):
        if is_instruct and tok.chat_template:
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    prompts = [fmt_prompt(d["messages"]) for d in datas]
    gts = [d["ground_truth"] for d in datas]

    if args.score_temperature and args.score_temperature > 0:
        score_sp = SamplingParams(temperature=args.score_temperature, top_p=0.95, n=1,
                                  seed=args.global_seed, max_tokens=max_tokens)
    else:
        score_sp = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=max_tokens)

    def accuracy(outs):
        c = 0
        for j, o in enumerate(outs):
            txt = o.outputs[0].text if o.outputs else ""
            if handler.is_answer_correct(txt, gts[j]):
                c += 1
        return c / max(1, len(outs))

    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(address="local", ignore_reinit_error=True)

    engines, pgs = launch_engines(args.num_engines, args.model_name,
                                  precision=args.precision, tensor_parallel_size=args.tp)
    n_eng = len(engines)

    rng = np.random.default_rng(args.global_seed)
    seeds = [int(s) for s in rng.integers(1, 2**31 - 1, size=args.population)]

    scored = []  # (seed, accuracy)
    try:
        base_acc = None
        if args.include_base:
            t0 = time.time()
            outs = ray.get(engines[0].generate.remote(prompts, score_sp, use_tqdm=False))
            base_acc = accuracy(outs)
            print(f"[search] base (unperturbed) acc={base_acc*100:.2f}% ({time.time()-t0:.1f}s)",
                  flush=True)

        nb = (len(seeds) + n_eng - 1) // n_eng
        for b in range(nb):
            batch = seeds[b * n_eng:(b + 1) * n_eng]
            t0 = time.time()
            # perturb θ -> θ + sigma*noise(seed) on each engine (its own seed).
            ray.get([engines[i].collective_rpc.remote(
                        "perturb_self_weights", args=(int(s), args.sigma, False))
                     for i, s in enumerate(batch)])
            outs = ray.get([engines[i].generate.remote(prompts, score_sp, use_tqdm=False)
                            for i in range(len(batch))])
            # restore with the same sigma so θ is exactly recovered for the next batch.
            ray.get([engines[i].collective_rpc.remote(
                        "restore_self_weights", args=(int(s), args.sigma, False))
                     for i, s in enumerate(batch)])
            for i, s in enumerate(batch):
                acc = accuracy(outs[i])
                scored.append((int(s), acc))
            best_b = max(scored[-len(batch):], key=lambda x: x[1])
            print(f"[search] batch {b+1}/{nb} ({time.time()-t0:.1f}s) "
                  f"best-in-batch seed={best_b[0]} acc={best_b[1]*100:.2f}%", flush=True)
    finally:
        cleanup_engines(engines, pgs)

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:max(1, args.top_k)]
    top_k_models = [{"seed": s, "sigma": args.sigma} for s, _ in top]

    out = {
        "base_model_path": args.model_name,
        "sigma": args.sigma,
        "population": args.population,
        "score_samples": len(datas),
        "base_acc": base_acc,
        "top_k_models": top_k_models,
        "scored": [{"seed": s, "acc": a} for s, a in scored],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.seeds_out)), exist_ok=True)
    with open(args.seeds_out, "w") as f:
        json.dump(out, f, indent=2)

    top_accs = ", ".join(f"{a*100:.1f}%" for _, a in top)
    base_str = f"{base_acc*100:.2f}%" if base_acc is not None else "n/a"
    print(f"[search] base={base_str}  selected top-{len(top)} of {len(scored)}: "
          f"seeds={[s for s, _ in top]}  accs=[{top_accs}]", flush=True)
    print(f"[search] wrote {args.seeds_out}", flush=True)
    print("SUCCESS: weight-space search complete", flush=True)


if __name__ == "__main__":
    main()
