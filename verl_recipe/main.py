"""verl-native iterative-RandOpt recipe.

For teams already on verl: this driver reuses verl's OWN components -- the FSDP
SFT trainer (``verl.trainer.fsdp_sft_trainer``), the checkpoint->HF merger
(``python -m verl.model_merger``) -- and only adds the outer loop (on-policy
rollouts -> reward rejection -> cumulative SFT from base, iterate). Your
model/data/trainer config stays the same.

Run:
    python -m iterative_randopt.verl_recipe.main --config verl_recipe/config/iterative_randopt_gsm8k.yaml

Requires a working ``verl`` install + vllm.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import List

import pandas as pd

from ..reward import get_reward
from ..rejection import Rollout, cumulative_dedup, select_correct_traces

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _prompt_text(tok, messages):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return "\n".join(m["content"] for m in messages) + "\n"


def _answer_extractor(dataset):
    if dataset == "gsm8k":
        from ..rewards import gsm8k as g

        def _x(t):
            a = g.extract_solution(t, method="strict")
            if a is None:
                a = g.extract_solution(t, method="flexible")
            return a or ""
        return _x
    return lambda t: ""


def generate_round(cfg, policy_model, rows, reward_func, extract):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(policy_model)
    prompts_msgs = [r["prompt"] for r in rows]
    gts = [str(r["ground_truth"]) for r in rows]
    prompt_texts = [_prompt_text(tok, m) for m in prompts_msgs]
    question_texts = [m[-1]["content"] for m in prompts_msgs]

    llm = LLM(model=policy_model, dtype=cfg.get("vllm_dtype", "bfloat16"),
              tensor_parallel_size=cfg.get("vllm_tensor_parallel_size", 1),
              gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.75),
              enforce_eager=True, enable_prefix_caching=False, disable_log_stats=True)
    sp = SamplingParams(temperature=cfg.get("gen_temperature", 1.0), top_p=0.95,
                        n=cfg.get("num_generations", 16), max_tokens=cfg.get("max_tokens", 1024))
    outs = llm.generate(prompt_texts, sp, use_tqdm=True)

    sft_rows = []
    for i, out in enumerate(outs):
        texts = [c.text for c in out.outputs]
        scores = reward_func(prompts=[prompts_msgs[i]] * len(texts), completions=texts,
                             ground_truth=[gts[i]] * len(texts))
        rollouts = [Rollout(text=t, answer=extract(t), correct=(s > 0))
                    for t, s in zip(texts, scores)]
        for k in select_correct_traces(rollouts, n_canonical=cfg.get("n_canonical", 8),
                                       keep_incorrect_frac=cfg.get("keep_incorrect_frac", 0.0)):
            sft_rows.append({"prompt": question_texts[i], "response": k.text})
    del llm
    try:
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    print(f"[iter-randopt-verl] kept {len(sft_rows)} correct traces from {len(rows)} questions")
    return sft_rows


def _run(cmd):
    print("[iter-randopt-verl] $", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = _PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, env=env, check=True)


def _consolidate_lora(merged_dir: str, lora_alpha: int) -> None:
    """Fold a verl LoRA adapter back into the merged HF model, in place.

    verl's FSDP model_merger, for a LoRA checkpoint, writes only the BASE weights
    to ``merged_dir`` and dumps the adapter under ``merged_dir/lora_adapter`` with
    ``lora_alpha=0`` (a known merger limitation). We fix the alpha and merge the
    adapter into the base so downstream eval / next-round warm-start use the
    actually-trained weights. No-op for full-FT runs (no adapter dir)."""
    adapter = os.path.join(merged_dir, "lora_adapter")
    cfg_path = os.path.join(adapter, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return  # full fine-tuning: merged_dir already holds the trained weights

    with open(cfg_path) as f:
        ac = json.load(f)
    if not ac.get("lora_alpha"):
        ac["lora_alpha"] = lora_alpha
        with open(cfg_path, "w") as f:
            json.dump(ac, f, indent=2)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[iter-randopt-verl] folding LoRA adapter into {merged_dir} (alpha={ac['lora_alpha']})", flush=True)
    base = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(merged_dir).save_pretrained(merged_dir)
    del base, merged
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(cfg):
    out_root = os.path.abspath(cfg["output_dir"])
    os.makedirs(out_root, exist_ok=True)
    dataset = cfg.get("dataset", "gsm8k")
    reward_func = get_reward(dataset)
    extract = _answer_extractor(dataset)
    ngpu = cfg.get("num_gpus", 8)
    m0 = cfg["model"]

    df = pd.read_parquet(cfg["train_files"])
    rows = []
    for r in df.to_dict("records"):
        msgs = r["prompt"]
        msgs = msgs.tolist() if hasattr(msgs, "tolist") else list(msgs)
        rm = r.get("reward_model", {})
        gt = rm.get("ground_truth") if isinstance(rm, dict) else r.get("ground_truth")
        rows.append({"prompt": msgs, "ground_truth": gt})
    if cfg.get("max_train_questions"):
        rows = rows[: cfg["max_train_questions"]]

    policy = m0
    all_rows: List[dict] = []
    history = []
    for R in range(1, cfg["rounds"] + 1):
        print(f"\n{'='*60}\n[iter-randopt-verl] ROUND {R}/{cfg['rounds']} (policy={policy})\n{'='*60}")
        rdir = os.path.join(out_root, f"r{R}")
        os.makedirs(rdir, exist_ok=True)

        round_rows = generate_round(cfg, policy, rows, reward_func, extract)
        all_rows.extend(round_rows)
        train_rows = cumulative_dedup(all_rows) if cfg.get("cumulative", True) else round_rows
        train_pq = os.path.join(rdir, "sft_train.parquet")
        pd.DataFrame(train_rows).to_parquet(train_pq)

        base = m0 if cfg.get("sft_from_base", True) else policy
        ckpt_dir = os.path.join(rdir, "sft_ckpt")
        _run(["torchrun", "--standalone", f"--nproc_per_node={ngpu}",
              "-m", "verl.trainer.fsdp_sft_trainer",
              f"data.train_files={train_pq}", f"data.val_files={train_pq}",
              "data.prompt_key=prompt", "data.response_key=response",
              # SFT packs prompt+response, so max_length must exceed generation
              # max_tokens; truncation=right guards rare over-long traces.
              f"data.max_length={cfg.get('sft_max_length', max(1024, 2 * cfg.get('max_tokens', 512)))}",
              "data.truncation=right",
              f"data.train_batch_size={cfg.get('train_batch_size', 256)}",
              f"model.partial_pretrain={base}",
              f"+model.attn_implementation={cfg.get('attn_implementation', 'flash_attention_2')}",
              f"model.lora_rank={cfg.get('lora_rank', 192)}",
              f"model.lora_alpha={cfg.get('lora_alpha', 2 * cfg.get('lora_rank', 192))}",
              f"optim.lr={cfg.get('lr', 1e-4)}",
              f"trainer.total_epochs={cfg.get('epochs', 2)}",
              f"trainer.default_local_dir={ckpt_dir}",
              f"trainer.n_gpus_per_node={ngpu}", "trainer.nnodes=1",
              "trainer.logger=[console]"])

        # verl saves the FSDP shards + huggingface/ config under
        # {default_local_dir}/global_step_N/, so merge that step dir (not ckpt_dir).
        steps = glob.glob(os.path.join(ckpt_dir, "global_step_*"))
        merge_src = max(steps, key=lambda p: int(p.rsplit("_", 1)[-1])) if steps else ckpt_dir
        hf_dir = os.path.join(rdir, "merged")
        _run([sys.executable, "-m", "verl.model_merger", "merge",
              "--backend", "fsdp", "--local_dir", merge_src, "--target_dir", hf_dir])
        # For LoRA runs verl's merger writes the BASE weights to hf_dir and the
        # adapter to hf_dir/lora_adapter (with a broken lora_alpha=0). Fold the
        # adapter back in so the eval/next-round policy uses the trained weights.
        _consolidate_lora(hf_dir, int(cfg.get("lora_alpha", 2 * cfg.get("lora_rank", 192))))

        eval_json = os.path.join(rdir, f"distilled_r{R}.json")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        env["PYTHONPATH"] = _PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-m", "iterative_randopt.stages.eval",
                        "--model", hf_dir, "--dataset", dataset, "--tp", "1",
                        "--max_tokens", str(cfg.get("max_tokens", 1024)),
                        "--label", f"distilled_r{R}", "--output_json", eval_json],
                       env=env, check=True)
        acc = json.load(open(eval_json))["accuracy"]
        history.append({"round": R, "model": hf_dir, "greedy_acc": acc})
        print(f"[iter-randopt-verl] round {R}: greedy = {acc*100:.2f}%")
        policy = hf_dir

    # Continual training can drift down late; report the best round's checkpoint.
    best = max(history, key=lambda h: h["greedy_acc"]) if history else None
    if best:
        print(f"[iter-randopt-verl] best = round {best['round']} "
              f"(greedy={best['greedy_acc']*100:.2f}%) -> {best['model']}")
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump({"rounds": history,
                   "best_round": (best["round"] if best else None),
                   "best_model": (best["model"] if best else None)}, f, indent=2)
    return history


def main():
    p = argparse.ArgumentParser(description="verl-native iterative-RandOpt recipe")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    run(_load_cfg(args.config))


if __name__ == "__main__":
    main()
