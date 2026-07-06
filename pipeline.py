"""Native single-node iterative-RandOpt pipeline.

Drives the proven stages (packaged inside iterative_randopt.stages):
  gen (on-policy rollouts + rejection sampling) -> cumulative dedup ->
  train (hard-CE SFT from base) -> eval (greedy),
iterated for `rounds`. Runs entirely on one node's GPUs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import List, Optional

import pandas as pd

from .config import IterativeRandOptConfig

# repo root / install root that makes `-m iterative_randopt...` importable.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: List[str], extra_env: Optional[dict] = None) -> None:
    print("[iter-randopt] $", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = _PKG_PARENT + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, env=env, check=True)


def _write_seeds(path: str, base_model: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"base_model_path": base_model, "top_k_models": []}, f)


def _merge_cumulative(out_path: str, parquets: List[str]) -> str:
    frames = [pd.read_parquet(p) for p in parquets if p and os.path.exists(p)]
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    if "response_ids" in df.columns:
        df["_rkey"] = df["response_ids"].apply(
            lambda x: tuple(int(t) for t in (x.tolist() if hasattr(x, "tolist") else x)))
        df = df.drop_duplicates(subset=["prompt", "_rkey"]).drop(columns=["_rkey"])
    else:
        df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    print(f"[iter-randopt] cumulative: {len(parquets)} parquets {n0} -> {len(df)} rows -> {out_path}")
    return out_path


def run(cfg: IterativeRandOptConfig) -> List[dict]:
    root = os.path.abspath(cfg.output_dir)
    os.makedirs(root, exist_ok=True)
    devices = ",".join(str(i) for i in range(cfg.num_gpus))
    py = sys.executable

    # 0. data prep (once).
    data_dir = os.path.join(root, "data", cfg.dataset)
    if cfg.dataset == "gsm8k":
        _run([py, "-m", "iterative_randopt.stages.prepare_gsm8k", "--local_save_dir", data_dir])
        train_path = os.path.join(data_dir, "train.parquet")
        test_path = os.path.join(data_dir, "test.parquet")
    else:
        raise ValueError(f"Only gsm8k has an auto data-prep step; got {cfg.dataset}")

    m0 = cfg.model
    cur_model = m0
    round_parquets: List[str] = []
    results: List[dict] = []
    best: Optional[dict] = None   # {"round", "model", "score"} of the highest-scoring round

    for r in range(1, cfg.rounds + 1):
        print(f"\n{'#'*70}\n# iterative RandOpt ROUND {r}/{cfg.rounds}  (policy={cur_model})\n{'#'*70}",
              flush=True)
        rdir = os.path.join(root, f"r{r}")
        ddir = os.path.join(rdir, "distill_data")
        seeds = os.path.join(rdir, "seeds.json")
        _write_seeds(seeds, cur_model)

        # 1. on-policy rollouts + rejection sampling (correct-only).
        _run([py, "-m", "iterative_randopt.stages.gen",
              "--dataset", cfg.dataset, "--model_name", cur_model,
              "--seeds_file", seeds, "--data_path", train_path,
              "--num_engines", str(cfg.num_gpus), "--tp", "1",
              "--cuda_devices", devices, "--max_tokens", str(cfg.max_tokens),
              "--max_samples", str(cfg.max_train_questions),
              "--n_canonical", str(cfg.n_canonical),
              "--gen_temperature", str(cfg.gen_temperature),
              "--gen_n_samples", str(cfg.num_generations),
              "--output_dir", ddir, "--run_name", f"{cfg.dataset}_distill",
              "--on_policy", "--keep_incorrect_frac", str(cfg.keep_incorrect_frac),
              "--skip_teacher"])
        round_pq = os.path.join(ddir, f"{cfg.dataset}_distill", "kd", "train_kd.parquet")
        round_parquets.append(round_pq)

        # 2. cumulative dedup across rounds.
        if cfg.cumulative and len(round_parquets) > 1:
            train_pq = _merge_cumulative(
                os.path.join(ddir, f"{cfg.dataset}_distill", "kd", "train_cumulative.parquet"),
                round_parquets)
        else:
            train_pq = round_pq

        # 3. SFT (hard CE, kd_alpha=1.0) from the ORIGINAL base each round.
        sft_base = m0 if cfg.sft_from_base else cur_model
        merged = os.path.join(rdir, "model", "merged")
        _run(["torchrun", "--standalone", f"--nproc_per_node={cfg.num_gpus}",
              "-m", "iterative_randopt.stages.train",
              "--train_parquet", train_pq, "--base_model", sft_base,
              "--output_dir", merged, "--adapter_dir", os.path.join(rdir, "model", "adapter"),
              "--max_length", str(cfg.max_tokens), "--epochs", str(cfg.epochs),
              "--lr", str(cfg.lr), "--warmup_ratio", str(cfg.warmup_ratio),
              "--lora_rank", str(cfg.lora_rank), "--lora_alpha", str(cfg.resolved_lora_alpha()),
              "--lora_dropout", str(cfg.lora_dropout),
              "--per_device_batch_size", str(cfg.per_device_batch_size),
              "--grad_accum", str(cfg.grad_accum), "--precision", cfg.precision,
              "--seed", str(cfg.seed), "--kd_alpha", "1.0", "--ce_on_correct_only"])

        # 4. eval greedy (single-model, temperature=0).
        eval_json = os.path.join(rdir, f"distilled_r{r}.json")
        _run([py, "-m", "iterative_randopt.stages.eval",
              "--model", merged, "--dataset", cfg.dataset, "--tp", "1",
              "--test_data_path", test_path, "--max_tokens", str(cfg.max_tokens),
              "--label", f"distilled_r{r}", "--output_json", eval_json],
             extra_env={"CUDA_VISIBLE_DEVICES": "0"})
        greedy = json.load(open(eval_json))

        results.append({"round": r, "model": merged, "greedy_acc": greedy["accuracy"]})
        print(f"[iter-randopt] round {r}: greedy={greedy['accuracy']*100:.2f}%")

        # Track the best round by greedy (single-model) accuracy.
        score = greedy["accuracy"]
        if best is None or score > best["score"]:
            best = {"round": r, "model": merged, "score": score}

        # r-1's model is no longer needed: from-base always trains from m0, and
        # continual has already consumed it as this round's warm-start base.
        # Never delete the current best round's checkpoint when keep_best is set.
        if r > 1:
            prev_model_dir = os.path.join(root, f"r{r-1}", "model")
            if not (cfg.keep_best and best and best["round"] == r - 1):
                subprocess.run(["rm", "-rf", prev_model_dir], check=False)
        cur_model = merged

    print("\n===================== iterative RandOpt SUMMARY =====================")
    for res in results:
        star = "  <- best" if (best and res["round"] == best["round"]) else ""
        print(f"  round {res['round']}: greedy={res['greedy_acc']*100:.2f}%" + star)

    # Expose the best round's checkpoint at a stable path (<out>/best) so callers
    # don't have to reason about late-round continual drift.
    best_out = None
    if cfg.keep_best and best:
        best_out = os.path.join(root, "best")
        if os.path.islink(best_out) or os.path.exists(best_out):
            subprocess.run(["rm", "-rf", best_out], check=False)
        try:
            os.symlink(best["model"], best_out)
        except OSError:
            best_out = best["model"]  # symlink unsupported: just report the real dir
        print(f"[iter-randopt] best = round {best['round']} "
              f"(greedy={best['score']*100:.2f}%) -> {best_out}")

    summary = {
        "rounds": results,
        "best_round": (best["round"] if best else None),
        "best_model": (best["model"] if best else None),
        "best_path": best_out,
    }
    with open(os.path.join(root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return results
