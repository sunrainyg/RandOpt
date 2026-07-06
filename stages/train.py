#!/usr/bin/env python3
"""distillation trainer.

Trains a LoRA student to match the ENSEMBLE-AVERAGED top-K next-token
distributions produced by distill_kd_gen.py (forward-KL), plus a small CE term
on the gold response tokens. This transfers the ensemble's "dark knowledge"
(relative probabilities over alternative tokens), which plain hard-label SFT
on the same traces cannot.

KD dataset columns (per solved question):
  prompt_ids[int], response_ids[int],
  teacher_ids[[int]], teacher_probs[[float]]  (one entry per response token),
  ground_truth.

Alignment: HF logits[t] predicts token at sequence position t+1. The teacher
distribution for response token m (full position p+m, where p=len(prompt_ids))
is therefore placed at prediction row t = p+m-1.
"""
import argparse
import os

# Reduce CUDA fragmentation OOM during multi-node KD (large transient logits).
# Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

IGNORE_INDEX = -100


def parse_args():
    p = argparse.ArgumentParser(description="Logit-KD distillation trainer")
    p.add_argument("--train_parquet", type=str, required=True)
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=256)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--kd_alpha", type=float, default=0.3,
                   help="Weight on hard-CE; (1-kd_alpha) goes to soft-KD.")
    p.add_argument("--kd_temperature", type=float, default=1.0,
                   help="KD softmax temperature T>1 to expose dark-knowledge tail.")
    p.add_argument("--kd_loss", type=str, default="fkl", choices=["fkl", "rkl", "jsd"],
                   help="Divergence over teacher top-K support: forward-KL (mode-covering), "
                        "reverse-KL (mode-seeking, GKD-preferred for small students), or "
                        "generalized JSD.")
    p.add_argument("--kd_jsd_beta", type=float, default=0.5,
                   help="Generalized-JSD interpolation (beta->1 ~ reverse-KL).")
    p.add_argument("--ce_on_correct_only", action="store_true",
                   help="On-policy GKD: apply the hard-CE term only to CORRECT rollouts "
                        "(incorrect self-rollouts get pure soft-KD teacher feedback).")
    p.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_dataset(parquet_path, eos_id, max_length, out_topk, ce_on_correct_only=False):
    df = pd.read_parquet(parquet_path)
    need = {"prompt_ids", "response_ids", "teacher_ids", "teacher_probs"}
    if not need.issubset(df.columns):
        raise ValueError(f"KD parquet missing columns; have {list(df.columns)}")
    has_correct = "matched_majority" in df.columns
    has_harvest = "is_harvest" in df.columns

    examples = []
    skipped = 0
    n_ce_masked = 0
    for row in df.to_dict("records"):
        pids = [int(x) for x in row["prompt_ids"]]
        rids = [int(x) for x in row["response_ids"]]
        t_ids = list(row["teacher_ids"])
        t_probs = list(row["teacher_probs"])
        p, r = len(pids), len(rids)
        if p == 0 or r == 0 or len(t_ids) != r:
            skipped += 1
            continue
        if p >= max_length:
            skipped += 1
            continue

        # On-policy GKD: do NOT teach the student to reproduce its OWN wrong tokens
        # via hard-CE; incorrect rollouts get ONLY soft-KD (ensemble teacher) feedback.
        correct = bool(row["matched_majority"]) if has_correct else True
        is_harvest = bool(row["is_harvest"]) if has_harvest else False
        # Mask hard-CE for (a) incorrect on-policy rollouts (don't reinforce the
        # student's own wrong tokens) and (b) harvested failure-set traces (don't
        # force the small student to memorize hard off-policy solutions -- give it
        # only the ensemble's SOFT correct-path distribution as guidance).
        ce_off = (ce_on_correct_only and not correct) or is_harvest
        if ce_off:
            n_ce_masked += 1

        input_ids = pids + rids + [eos_id]
        if ce_off:
            labels = [IGNORE_INDEX] * (p + r + 1)
        else:
            labels = [IGNORE_INDEX] * p + rids + [eos_id]
        L = len(input_ids)

        kd_ids = np.zeros((L, out_topk), dtype=np.int64)
        kd_probs = np.zeros((L, out_topk), dtype=np.float32)
        for m in range(r):
            t = p + m - 1  # prediction row for response token m
            if t < 0 or t >= L:
                continue
            ids_m = [int(x) for x in (t_ids[m] if t_ids[m] is not None else [])]
            pr_m = [float(x) for x in (t_probs[m] if t_probs[m] is not None else [])]
            k = min(len(ids_m), out_topk)
            if k == 0:
                continue
            kd_ids[t, :k] = ids_m[:k]
            kd_probs[t, :k] = pr_m[:k]

        if L > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            kd_ids = kd_ids[:max_length]
            kd_probs = kd_probs[:max_length]

        examples.append({
            "input_ids": input_ids,
            "labels": labels,
            "kd_ids": kd_ids.tolist(),
            "kd_probs": kd_probs.tolist(),
        })

    print(f"[data] built {len(examples)} KD examples (skipped {skipped}; "
          f"ce-masked incorrect rollouts={n_ce_masked})")
    if not examples:
        raise RuntimeError("No usable KD examples.")
    return Dataset.from_list(examples)


class KDCollator:
    def __init__(self, pad_id, out_topk):
        self.pad_id = pad_id
        self.out_topk = out_topk

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        B, K = len(features), self.out_topk
        input_ids = torch.full((B, maxlen), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        labels = torch.full((B, maxlen), IGNORE_INDEX, dtype=torch.long)
        kd_ids = torch.zeros((B, maxlen, K), dtype=torch.long)
        kd_probs = torch.zeros((B, maxlen, K), dtype=torch.float32)
        for b, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[b, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            attn[b, :n] = 1
            labels[b, :n] = torch.tensor(f["labels"], dtype=torch.long)
            ki = torch.tensor(f["kd_ids"], dtype=torch.long)
            kp = torch.tensor(f["kd_probs"], dtype=torch.float32)
            kd_ids[b, :ki.shape[0]] = ki
            kd_probs[b, :kp.shape[0]] = kp
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "kd_ids": kd_ids,
            "kd_probs": kd_probs,
        }


class KDTrainer(Trainer):
    def __init__(self, *a, kd_alpha=0.3, kd_temperature=1.0,
                 kd_loss="fkl", kd_jsd_beta=0.5, **kw):
        super().__init__(*a, **kw)
        self.kd_alpha = kd_alpha
        self.kd_temperature = max(1.0, float(kd_temperature))
        self.kd_loss = kd_loss
        self.kd_jsd_beta = float(kd_jsd_beta)

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        kd_ids = inputs.pop("kd_ids")
        kd_probs = inputs.pop("kd_probs")
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        logits = outputs.logits  # (B, L, V), bf16

        # ---- hard CE (next-token) ----
        # NOTE: do NOT upcast the full (B,L,V) tensor to fp32 -- cross_entropy fuses
        # the log_softmax and accumulates in fp32 internally, so passing bf16 logits
        # avoids a multi-GB fp32 transient that intermittently OOM'd multi-node KD.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        # ---- soft KD (over teacher top-K, temperature-softened) ----
        # teacher dist at row t targets the student's prediction at logits[t].
        # Temperature T>1 softens BOTH distributions to expose the ensemble's
        # "dark knowledge" tail (Hinton 2015); scale by T^2 to preserve grad mag.
        # Memory-efficient: get log p_S(id) = logits_T(id) - logsumexp(logits_T) via
        # gather + logsumexp instead of a full (B,L,V) fp32 log_softmax tensor.
        T = self.kd_temperature
        logits_T = logits / T                                   # (B,L,V) bf16
        # bf16 logsumexp is numerically stable (subtracts the row max internally);
        # avoids the full (B,L,V) fp32 copy. logZ is (B,L,1).
        logZ = torch.logsumexp(logits_T, dim=-1, keepdim=True).float()
        gathered = torch.gather(logits_T, 2, kd_ids).float() - logZ     # (B,L,K) = log p_S^full
        if T > 1.0:
            t_probs = kd_probs.clamp(min=0).pow(1.0 / T)        # soften teacher
            t_probs = t_probs / t_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        else:
            t_probs = kd_probs
        eps = 1e-9
        if self.kd_loss == "fkl":
            # forward KL / cross-entropy: mode-covering. -sum p_T log p_S^full
            kd_per = -(t_probs * gathered).sum(dim=-1)
        else:
            # restrict student to teacher's top-K support: q = softmax(gathered)
            log_q = F.log_softmax(gathered, dim=-1)             # (B,L,K)
            q = log_q.exp()
            log_p = torch.log(t_probs.clamp(min=eps))
            if self.kd_loss == "rkl":
                # reverse KL KL(q||p): mode-seeking (GKD-preferred for small students)
                kd_per = (q * (log_q - log_p)).sum(dim=-1)
            else:  # generalized JSD
                beta = self.kd_jsd_beta
                m = (beta * t_probs + (1.0 - beta) * q).clamp(min=eps)
                log_m = torch.log(m)
                kd_per = (beta * (t_probs * (log_p - log_m)).sum(dim=-1)
                          + (1.0 - beta) * (q * (log_q - log_m)).sum(dim=-1))
        kd_per = kd_per * (T * T)                               # (B, L)
        mask = (kd_probs.sum(dim=-1) > 0).float()               # (B, L)
        denom = mask.sum().clamp(min=1.0)
        kd_loss = (kd_per * mask).sum() / denom

        loss = self.kd_alpha * ce + (1.0 - self.kd_alpha) * kd_loss
        if self.state.global_step % 25 == 0:
            print(f"[kd:{self.kd_loss}] step={self.state.global_step} ce={ce.item():.4f} "
                  f"kd={kd_loss.item():.4f} loss={loss.item():.4f}")
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # infer out_topk from data
    df0 = pd.read_parquet(args.train_parquet)
    out_topk = 1
    for tp in df0["teacher_probs"]:
        for row in tp:
            if row is not None and len(row) > out_topk:
                out_topk = len(row)
    print(f"[kd] out_topk = {out_topk}")
    del df0

    train_ds = build_dataset(args.train_parquet, eos_id, args.max_length, out_topk,
                             ce_on_correct_only=args.ce_on_correct_only)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        bf16=(args.precision == "bfloat16"),
        fp16=(args.precision == "float16"),
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        label_names=["labels"],
        # Multi-node robustness: drop_last guarantees every DP rank takes the SAME
        # number of optimizer steps (uneven on-policy dataset sizes otherwise cause
        # a rank to fall a step behind -> NCCL collective timeout / SIGABRT). A long
        # ddp_timeout also tolerates a transient slow all-reduce instead of aborting.
        dataloader_drop_last=True,
        ddp_timeout=7200,
    )

    trainer = KDTrainer(
        model=model, args=training_args, train_dataset=train_ds,
        data_collator=KDCollator(pad_id, out_topk), kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        kd_loss=args.kd_loss, kd_jsd_beta=args.kd_jsd_beta,
    )
    trainer.train()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.adapter_dir:
            os.makedirs(args.adapter_dir, exist_ok=True)
            model.save_pretrained(args.adapter_dir)
        print("[merge] merging LoRA adapter...")
        merged = model.merge_and_unload()
        merged.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[save] merged model -> {args.output_dir}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
