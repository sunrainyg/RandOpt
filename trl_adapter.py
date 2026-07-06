"""TRL-native iterative-RandOpt trainer.

For teams already on TRL: ``IterativeRandOptTrainer`` mirrors the ``GRPOTrainer``
surface (a ``SFTConfig`` subclass for ``args`` + reward functions with the SAME
``(prompts, completions, **kwargs) -> list[float]`` signature), so adopting the
method is essentially swapping the trainer class::

    from iterative_randopt.trl import IterativeRandOptTrainer, IterativeRandOptTRLConfig
    from iterative_randopt.reward import gsm8k_reward

    trainer = IterativeRandOptTrainer(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        args=IterativeRandOptTRLConfig(output_dir="out", num_rounds=4,
                                       num_generations=16, gen_temperature=1.0),
        train_dataset=ds,                 # columns: 'prompt' (messages) + 'ground_truth'
        reward_funcs=gsm8k_reward,        # identical signature to GRPOTrainer
        processing_class=tokenizer,
    )
    trainer.train()

Each round the CURRENT policy samples ``num_generations`` rollouts/question, we
keep only reward>0 traces (rejection sampling), dedup+accumulate across rounds,
and SFT the ORIGINAL base with hard CE (TRL ``SFTTrainer``, completion-only loss).
Generation uses vLLM. Because this trainer co-locates vLLM + SFTTrainer in ONE
process, run it on a SINGLE visible GPU (``CUDA_VISIBLE_DEVICES=0 python train.py``)
-- otherwise HF Trainer auto-DataParallels across every visible GPU and collides
with vLLM's CUDA context. For a fully-sharded 8-GPU reproduction use the native
pipeline (``iter_randopt``) instead.
"""
from __future__ import annotations

import dataclasses
import gc
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

from .rejection import Rollout, cumulative_dedup, select_correct_traces


def _lazy_imports():
    try:
        import torch  # noqa
        from trl import SFTConfig, SFTTrainer  # noqa
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "iterative_randopt.trl requires `trl` (and torch). pip install trl") from e
    return SFTConfig, SFTTrainer


try:
    from trl import SFTConfig as _SFTConfig
    _HAS_TRL = True
except Exception:  # pragma: no cover
    _SFTConfig = object  # type: ignore
    _HAS_TRL = False


if _HAS_TRL:
    @dataclass
    class IterativeRandOptTRLConfig(_SFTConfig):
        """SFTConfig + iterative-RandOpt loop knobs (GRPOConfig-style extras)."""
        num_rounds: int = 4
        num_generations: int = 16
        gen_temperature: float = 1.0
        gen_top_p: float = 0.95
        n_canonical: int = 8
        keep_incorrect_frac: float = 0.0
        sft_from_base: bool = True    # True = canonical ReST^EM (SFT base each round, more stable); False = continual (warm-start prev round)
        cumulative: bool = True
        max_train_questions: Optional[int] = None
        max_new_tokens: int = 1024
        lora_rank: int = 192
        lora_alpha: Optional[int] = None
        lora_dropout: float = 0.05
        vllm_tensor_parallel_size: int = 1
        vllm_gpu_memory_utilization: float = 0.75
        vllm_dtype: str = "bfloat16"
else:  # pragma: no cover
    class IterativeRandOptTRLConfig:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("iterative_randopt.trl requires `trl`. pip install trl")


class IterativeRandOptTrainer:
    """Iterative rejection-sampling self-training on top of TRL ``SFTTrainer``."""

    def __init__(
        self,
        model: Union[str, "object"],
        args: "IterativeRandOptTRLConfig",
        train_dataset,
        reward_funcs: Union[Callable, Sequence[Callable]],
        processing_class=None,
        answer_extractor: Optional[Callable[[str], str]] = None,
        eval_fn: Optional[Callable[[str], float]] = None,
    ):
        self.SFTConfig, self.SFTTrainer = _lazy_imports()
        if not isinstance(model, str):
            raise ValueError("Pass a model id/path string as `model` so each round can "
                             "reload the base for SFT-from-base / vLLM generation.")
        self.base_model = model
        self.args = args
        self.train_dataset = train_dataset
        self.reward_funcs = list(reward_funcs) if isinstance(reward_funcs, (list, tuple)) \
            else [reward_funcs]
        self.processing_class = processing_class
        self.answer_extractor = answer_extractor
        self.eval_fn = eval_fn
        self.history: List[dict] = []

    def _prompt_text(self, messages):
        tok = self.processing_class
        if tok is not None and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    def _score(self, prompts, completions, ground_truths):
        total = [0.0] * len(completions)
        for rf in self.reward_funcs:
            scores = rf(prompts=prompts, completions=completions, ground_truth=ground_truths)
            for i, s in enumerate(scores):
                total[i] += float(s)
        return total

    def _generate_round(self, policy_model: str):
        from vllm import LLM, SamplingParams

        rows = list(self.train_dataset)
        if self.args.max_train_questions:
            rows = rows[: self.args.max_train_questions]
        prompts_msgs = [r["prompt"] for r in rows]
        gts = [str(r.get("ground_truth", "")) for r in rows]
        prompt_texts = [self._prompt_text(m) for m in prompts_msgs]

        llm = LLM(model=policy_model, dtype=self.args.vllm_dtype,
                  tensor_parallel_size=self.args.vllm_tensor_parallel_size,
                  gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                  enforce_eager=True, enable_prefix_caching=False, disable_log_stats=True)
        sp = SamplingParams(temperature=self.args.gen_temperature, top_p=self.args.gen_top_p,
                            n=max(1, self.args.num_generations), max_tokens=self.args.max_new_tokens)
        outs = llm.generate(prompt_texts, sp, use_tqdm=True)

        sft_rows = []
        for i, out in enumerate(outs):
            texts = [c.text for c in out.outputs]
            scores = self._score([prompts_msgs[i]] * len(texts), texts, [gts[i]] * len(texts))
            rollouts = [Rollout(text=t, answer=(self.answer_extractor(t) if self.answer_extractor else ""),
                                correct=(s > 0)) for t, s in zip(texts, scores)]
            for k in select_correct_traces(rollouts, n_canonical=self.args.n_canonical,
                                           keep_incorrect_frac=self.args.keep_incorrect_frac):
                sft_rows.append({"prompt": prompts_msgs[i],
                                 "completion": [{"role": "assistant", "content": k.text}]})
        del llm
        _free_gpu()
        print(f"[iter-randopt-trl] kept {len(sft_rows)} correct traces from {len(rows)} questions")
        return sft_rows

    def _sft_round(self, base_model: str, sft_rows, out_dir: str) -> str:
        from datasets import Dataset
        ds = Dataset.from_list(sft_rows)
        cfg = _clone_sftconfig(self.SFTConfig, self.args, out_dir)
        trainer = self.SFTTrainer(model=base_model, args=cfg, train_dataset=ds,
                                  processing_class=self.processing_class,
                                  peft_config=_peft_config(self.args))
        trainer.train()
        merged_dir = os.path.join(out_dir, "merged")
        model = trainer.model
        try:
            model = model.merge_and_unload()
        except Exception:
            pass
        model.save_pretrained(merged_dir, safe_serialization=True)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(merged_dir)
        del trainer, model
        _free_gpu()
        return merged_dir

    def train(self):
        m0 = self.base_model
        policy = m0
        all_rows: List[dict] = []
        for r in range(1, self.args.num_rounds + 1):
            print(f"\n{'='*60}\n[iter-randopt-trl] ROUND {r}/{self.args.num_rounds} "
                  f"(policy={policy})\n{'='*60}")
            round_rows = self._generate_round(policy)
            all_rows.extend(round_rows)
            train_rows = cumulative_dedup(all_rows) if self.args.cumulative else round_rows
            if not train_rows:
                print("[iter-randopt-trl] no correct traces this round; stopping.")
                break
            base = m0 if self.args.sft_from_base else policy
            out_dir = os.path.join(self.args.output_dir, f"r{r}")
            merged = self._sft_round(base, train_rows, out_dir)
            acc = self.eval_fn(merged) if self.eval_fn else None
            self.history.append({"round": r, "model": merged, "acc": acc})
            if acc is not None:
                print(f"[iter-randopt-trl] round {r}: greedy acc = {acc*100:.2f}%")
            policy = merged
        return self.history


def _free_gpu():
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
    except Exception:
        pass


def _peft_config(args: "IterativeRandOptTRLConfig"):
    if getattr(args, "lora_rank", 0) and args.lora_rank > 0:
        from peft import LoraConfig
        alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank
        return LoraConfig(r=args.lora_rank, lora_alpha=alpha, lora_dropout=args.lora_dropout,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    return None


def _clone_sftconfig(SFTConfig, args: "IterativeRandOptTRLConfig", out_dir: str):
    restem_only = {"num_rounds", "num_generations", "gen_temperature", "gen_top_p",
                   "n_canonical", "keep_incorrect_frac", "sft_from_base", "cumulative",
                   "max_train_questions", "max_new_tokens", "lora_rank", "lora_alpha",
                   "lora_dropout", "vllm_tensor_parallel_size", "vllm_gpu_memory_utilization",
                   "vllm_dtype"}
    kw = {}
    for f in dataclasses.fields(args):
        # skip our loop-only knobs and TrainingArguments' internal / init=False
        # fields (e.g. `_n_gpu`) that are not valid SFTConfig constructor args.
        if f.name in restem_only or not f.init or f.name.startswith("_"):
            continue
        kw[f.name] = getattr(args, f.name)
    kw["output_dir"] = out_dir
    kw.setdefault("completion_only_loss", True)
    return SFTConfig(**kw)
