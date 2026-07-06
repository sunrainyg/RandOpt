"""Generate ENSEMBLE-AVERAGED soft labels for distillation.

Pipeline:
  Pass 1 (sample): generate n_samples traces per model at temperature T; collect
                   correct traces per question and tally per-question answer
                   votes across the whole ensemble.
  Select: per question, keep up to n_canonical distinct correct traces, ranked
          by (matches-majority-vote, shortest); fall back to greedy if needed.
  Pass 2 (teacher-force): for each model, prompt_logprobs over
                   prompt_ids + response_ids for every kept trace; accumulate
                   exp(logprob) per token id at each response position; average
                   over models.
Output: kd/train_kd.parquet with one row per kept (question, trace):
  { prompt: str, prompt_ids: [int], response_ids: [int],
    teacher_ids: [[int]], teacher_probs: [[float]],   # per response position
    ground_truth: str, matched_majority: bool }
"""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import ray
from transformers import AutoTokenizer
from vllm import SamplingParams
try:
    from vllm.inputs import TokensPrompt
except Exception:  # older/newer vLLM: a plain dict is also accepted
    def TokensPrompt(prompt_token_ids):
        return {"prompt_token_ids": prompt_token_ids}

from ..data import get_dataset_handler, list_datasets
from ..engine import launch_engines, cleanup_engines


def parse_args():
    p = argparse.ArgumentParser(description="Logit-KD soft-label generator (diverse, vote-aligned)")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=list_datasets())
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--seeds_file", type=str, required=True)
    p.add_argument("--max_models", type=int, default=32,
                   help="Ensemble size used for soft-label averaging")
    p.add_argument("--teacher_topk", type=int, default=20,
                   help="prompt_logprobs depth requested from each model")
    p.add_argument("--out_topk", type=int, default=16,
                   help="Top-K kept per position after ensemble averaging")
    p.add_argument("--pass2_chunk", type=int, default=2048,
                   help="Teacher-force this many sequences per ray.get to bound driver memory")
    # ---- diversity / majority-vote alignment ----
    p.add_argument("--n_canonical", type=int, default=4,
                   help="Max distinct correct traces kept per question for soft labels")
    p.add_argument("--gen_temperature", type=float, default=0.8,
                   help="Sampling temperature for the diverse generation pass (0 = greedy)")
    p.add_argument("--gen_n_samples", type=int, default=2,
                   help="Samples per model in the diverse generation pass")
    p.add_argument("--require_majority_match", action="store_true",
                   help="Only keep traces whose final answer == ensemble majority vote "
                        "(falls back to any correct trace if none match)")
    p.add_argument("--min_votes", type=int, default=1,
                   help="Skip a question unless the majority answer has >= this many votes")
    # ---- on-policy GKD (Agarwal et al. 2023): rollouts come from the UNPERTURBED
    # student itself (the base loaded in the engines = CUR_MODEL), and the ensemble
    # teacher-forces over THOSE self-generated sequences. This fixes the off-policy
    # train-inference mismatch that made fixed-trace logit-KD underperform: the
    # student now receives ensemble (teacher) token-level feedback on its OWN
    # trajectories, including its mistakes. ----
    p.add_argument("--on_policy", action="store_true",
                   help="Sample rollouts from the unperturbed student (on-policy GKD)")
    p.add_argument("--keep_incorrect_frac", type=float, default=0.5,
                   help="On-policy: fraction of per-question kept slots allowed to be "
                        "INCORRECT student rollouts (learn-from-mistakes via teacher feedback)")
    # ---- failure-set rejection sampling (coverage expansion) ----
    # Pure on-policy GKD plateaus because the ensemble teacher ~= the student at the
    # TOKEN level (it is just perturbations of the student), so the KD gradient
    # ~ (p_student - p_teacher) ~ 0 on questions the student already answers. The
    # ONLY place the ensemble is genuinely stronger is the student's FAILURE set:
    # questions the unperturbed student gets wrong but some perturbed member solves.
    # We harvest those ensemble-correct traces (STaR/ReST rejection sampling, but
    # TARGETED at failures, not blanket) and add them as hard-CE training items, so
    # the student gains coverage on exactly the questions that drive the gap.
    p.add_argument("--harvest_failures", action="store_true",
                   help="On-policy: also harvest ensemble-correct traces on the student's "
                        "failure set (questions with no correct student rollout)")
    p.add_argument("--harvest_n_canonical", type=int, default=2,
                   help="Max ensemble-correct traces kept per failure-set question")
    # ---- pure on-policy rejection-sampling self-training (ReST^EM / STaR) -------
    # When set, SKIP the ensemble teacher-forcing (Pass2) entirely and emit traces
    # with EMPTY soft targets. Train with kd_alpha=1.0 -> pure hard-CE on the
    # student's OWN correct samples. This sharpens greedy decoding toward the
    # model's pass@k (we measured greedy 73% vs SC@20 83%: the model already
    # "knows" the answers but mis-commits greedily; self-training raises pass@1).
    p.add_argument("--skip_teacher", action="store_true",
                   help="Skip Pass2 teacher-forcing; emit empty soft targets (pure-CE SFT)")
    # ---- residual self-distillation (target the pass@1 <-> pass@k gap) ----------
    # ReST on ALL correct samples plateaus (greedy ~77%, SC@20 ~84%): it spends
    # equal capacity on questions greedy already solves. The gap lives on the
    # RECOVERABLE residual: questions greedy gets WRONG but temperature sampling
    # (and thus self-consistency) gets RIGHT. We probe the greedy answer per
    # question and OVERSAMPLE the correct (SC-majority-aligned) sampled traces on
    # exactly those questions, focusing CE on converting pass@k into pass@1.
    p.add_argument("--residual_focus", action="store_true",
                   help="On-policy: probe greedy answer; oversample correct sampled traces "
                        "on questions where greedy is WRONG but sampling recovers the answer")
    p.add_argument("--residual_weight", type=int, default=3,
                   help="Replication factor for correct traces on greedy-failure questions")
    # ----
    p.add_argument("--precision", type=str, default="bfloat16")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--num_engines", type=int, default=8)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--cuda_devices", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="distill_data")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def load_data_from_parquet(data_path, max_samples=None):
    df = pd.read_parquet(data_path)
    datas = []
    for row in df.to_dict("records"):
        messages = row["prompt"]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        rm = row.get("reward_model", {})
        gt = str(rm.get("ground_truth", "")) if isinstance(rm, dict) else ""
        datas.append({"messages": list(messages), "ground_truth": gt})
        if max_samples and len(datas) >= max_samples:
            break
    print(f"  Loaded {len(datas)} samples from {data_path}")
    return datas


def _gt_for_parquet(gt):
    """Store ground_truth in a parquet-safe form. gsm8k uses a plain string;
    countdown uses a dict {numbers, target}. Serialize non-strings to JSON so
    pyarrow never has to infer a (version-sensitive) struct column. Nothing
    downstream reads ground_truth back for training, so the exact form is
    immaterial -- this just keeps to_parquet robust across datasets."""
    return gt if isinstance(gt, str) else json.dumps(gt)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens
    # gsm8k uses a "#### <ans>" final-answer marker that lets us cheaply skip
    # traces without a final answer before running the (costlier) reward. Tag-based
    # datasets (e.g. countdown's <answer>..</answer>) have no such marker, so we
    # disable the pre-filter and rely on is_answer_correct / extract_answer instead.
    ans_marker = "####" if handler.name == "gsm8k" else None

    with open(args.seeds_file) as f:
        seeds_info = json.load(f)
    base_model_path = seeds_info.get("base_model_path", args.model_name)
    perturbs = [(m["seed"], m["sigma"]) for m in seeds_info["top_k_models"]]
    if args.max_models and args.max_models > 0:
        perturbs = perturbs[:args.max_models]
    print(f"[KD] ensemble size = {len(perturbs)}  base = {base_model_path}")

    run_name = args.run_name or f"{args.dataset}_kd_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = f"{args.output_dir}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    data_path = args.data_path or handler.default_train_path
    # Use the dataset handler's loader so this works for both parquet datasets
    # (gsm8k: string ground_truth) and JSON datasets (countdown: dict ground_truth
    # with {numbers, target}). The handler preserves the native ground_truth type.
    datas = handler.load_data(data_path, split="train", max_samples=args.max_samples)
    print(f"  Loaded {len(datas)} samples from {data_path}")

    tok = AutoTokenizer.from_pretrained(base_model_path)
    is_instruct = any(x in args.model_name.lower() for x in ["instruct", "chat", "it"])

    def fmt_prompt(messages):
        if is_instruct and tok.chat_template:
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    prompts = [fmt_prompt(d["messages"]) for d in datas]
    prompt_ids_list = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]

    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(address="local", ignore_reinit_error=True)

    engines, pgs = launch_engines(args.num_engines, base_model_path,
                                  precision=args.precision, tensor_parallel_size=args.tp)
    n_eng = len(engines)

    try:
        # ---------------- Pass 1: diverse sampling, gather traces + votes ----
        if args.gen_temperature and args.gen_temperature > 0:
            gen_sp = SamplingParams(temperature=args.gen_temperature,
                                    top_p=0.95,
                                    n=max(1, args.gen_n_samples),
                                    seed=args.global_seed,
                                    max_tokens=max_tokens)
        else:
            gen_sp = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=max_tokens)

        votes = defaultdict(Counter)         # sidx -> Counter(answer -> count)
        items = []                           # (sidx, response_ids, answer, correct)
        n_q_kept = 0
        n_majority_skipped = 0

        if args.on_policy:
            # ---- On-policy GKD: rollouts from the UNPERTURBED student (base) ----
            # all_traces[sidx] = list of (rids, answer, correct, text), deduped later
            all_traces = defaultdict(list)
            shards = [[] for _ in range(n_eng)]
            shard_idx = [[] for _ in range(n_eng)]
            for i, p in enumerate(prompts):
                e = i % n_eng
                shards[e].append(p)
                shard_idx[e].append(i)
            print(f"[Pass1/on-policy] sampling student rollouts T={args.gen_temperature} "
                  f"n={args.gen_n_samples} across {n_eng} engines")
            t0 = time.time()
            active = [e for e in range(n_eng) if shards[e]]
            outs = ray.get([engines[e].generate.remote(shards[e], gen_sp, use_tqdm=False)
                            for e in active])
            for oi, e in enumerate(active):
                for local_j, o in enumerate(outs[oi]):
                    sidx = shard_idx[e][local_j]
                    gt = datas[sidx]["ground_truth"]
                    for comp in o.outputs:
                        txt = comp.text
                        if ans_marker is not None and ans_marker not in txt:
                            continue
                        ans = handler.extract_answer(txt)
                        if ans:
                            votes[sidx][str(ans).strip()] += 1
                        correct = handler.is_answer_correct(txt, gt)
                        all_traces[sidx].append(
                            (list(comp.token_ids), str(ans).strip() if ans else "", correct, txt))
            print(f"  [Pass1/on-policy] rollouts for {len(all_traces)} questions "
                  f"({time.time()-t0:.1f}s)")

            # ---- optional greedy probe: label each question greedy-correct/wrong ----
            greedy_wrong = set()
            if args.residual_focus:
                gsp = SamplingParams(temperature=0.0, n=1, seed=args.global_seed,
                                     max_tokens=max_tokens)
                t0 = time.time()
                gouts = ray.get([engines[e].generate.remote(shards[e], gsp, use_tqdm=False)
                                 for e in active])
                for oi, e in enumerate(active):
                    for local_j, o in enumerate(gouts[oi]):
                        sidx = shard_idx[e][local_j]
                        gt = datas[sidx]["ground_truth"]
                        txt = o.outputs[0].text if o.outputs else ""
                        if not handler.is_answer_correct(txt, gt):
                            greedy_wrong.add(sidx)
                print(f"  [residual] greedy probe: {len(greedy_wrong)}/{len(prompts)} "
                      f"questions greedy-WRONG ({time.time()-t0:.1f}s)")

            n_keep = max(1, args.n_canonical)
            n_wrong_max = int(round(max(0.0, args.keep_incorrect_frac) * n_keep))
            n_residual_os = 0
            for sidx in sorted(all_traces.keys()):
                seen, uniq = set(), []
                for rids, ans, correct, txt in all_traces[sidx]:
                    if txt in seen:
                        continue
                    seen.add(txt); uniq.append((rids, ans, correct))
                if not uniq:
                    continue
                # Rank correct traces by SC-consensus alignment, then brevity, so the
                # student is sharpened toward the self-consistency MAJORITY path (the
                # high-mass mode greedy keeps missing) rather than an arbitrary trace.
                maj_ans = votes[sidx].most_common(1)[0][0] if votes[sidx] else None
                correct_list = [u for u in uniq if u[2]]
                correct_list.sort(key=lambda u: (maj_ans is None or u[1] != maj_ans, len(u[0])))
                wrong_list = [u for u in uniq if not u[2]]
                kept = correct_list[:n_keep]
                room = n_keep - len(kept)
                if room > 0 and n_wrong_max > 0:
                    kept += wrong_list[:min(room, n_wrong_max)]
                if not kept:  # student got it all wrong -> still learn from mistakes
                    kept = wrong_list[:max(1, n_wrong_max or 1)]
                # Residual oversampling: greedy missed it but sampling recovered the
                # (consensus) answer -> replicate the correct traces so CE capacity is
                # spent converting this pass@k success into a pass@1 (greedy) success.
                rep = 1
                if args.residual_focus and sidx in greedy_wrong and correct_list:
                    rep = max(1, args.residual_weight)
                    n_residual_os += 1
                for rids, ans, correct in kept:
                    for _ in range(rep if correct else 1):
                        items.append((sidx, rids, ans, correct, False))
                n_q_kept += 1
            n_corr = sum(1 for it in items if it[3])
            print(f"[Select/on-policy] kept {len(items)} rollouts across {n_q_kept} questions "
                  f"({n_corr} correct, {len(items)-n_corr} incorrect); "
                  f"residual-oversampled {n_residual_os} greedy-fail questions "
                  f"x{max(1, args.residual_weight) if args.residual_focus else 1}; "
                  f"avg {len(items)/max(1,n_q_kept):.2f}/question")

            # ---- Failure-set rejection sampling (coverage expansion) ----------
            if args.harvest_failures:
                # failure set = questions with NO correct student rollout (the gap).
                fail_sidx = [s for s in range(len(prompts))
                             if not any(c for (_, _, c, _) in all_traces.get(s, []))]
                print(f"[Harvest] student failure set = {len(fail_sidx)} questions; "
                      f"sampling {len(perturbs)} ensemble members for correct traces")
                # ensemble-correct traces per failure question: {sidx: {text:(rids,ans)}}
                harvest = defaultdict(dict)
                hvotes = defaultdict(Counter)
                fail_prompts = [prompts[s] for s in fail_sidx]
                nbh = (len(perturbs) + n_eng - 1) // n_eng
                for b in range(nbh):
                    batch = perturbs[b * n_eng:(b + 1) * n_eng]
                    if not batch or not fail_prompts:
                        continue
                    t0 = time.time()
                    ray.get([engines[i].collective_rpc.remote("perturb_self_weights", args=(int(s), sg, False))
                             for i, (s, sg) in enumerate(batch)])
                    outs = ray.get([engines[i].generate.remote(fail_prompts, gen_sp, use_tqdm=False)
                                    for i in range(len(batch))])
                    ray.get([engines[i].collective_rpc.remote("restore_self_weights", args=(int(s), sg, False))
                             for i, (s, sg) in enumerate(batch)])
                    for li in range(len(batch)):
                        for fi, o in enumerate(outs[li]):
                            sidx = fail_sidx[fi]
                            gt = datas[sidx]["ground_truth"]
                            for comp in o.outputs:
                                txt = comp.text
                                if ans_marker is not None and ans_marker not in txt:
                                    continue
                                ans = handler.extract_answer(txt)
                                if ans:
                                    hvotes[sidx][str(ans).strip()] += 1
                                if not handler.is_answer_correct(txt, gt):
                                    continue
                                if txt not in harvest[sidx]:
                                    harvest[sidx][txt] = (list(comp.token_ids), str(ans).strip())
                    print(f"  [Harvest] batch {b+1}/{nbh} ({time.time()-t0:.1f}s) "
                          f"recovered={len(harvest)}")
                # select up to harvest_n_canonical per recovered question, prefer
                # majority-vote-aligned + shortest (rejection sampling on the gap).
                n_hv_q, n_hv_tr = 0, 0
                for sidx in sorted(harvest.keys()):
                    traces = harvest[sidx]
                    if not traces:
                        continue
                    maj_ans = hvotes[sidx].most_common(1)[0][0] if hvotes[sidx] else None
                    cand = []
                    for txt, (rids, ans) in traces.items():
                        matched = (maj_ans is not None and ans == maj_ans)
                        cand.append((not matched, len(rids), rids, ans, matched))
                    cand.sort(key=lambda x: (x[0], x[1]))
                    for _, _, rids, ans, _m in cand[:max(1, args.harvest_n_canonical)]:
                        # is_harvest=True -> trained with SOFT-KD only (no hard-CE), so
                        # the small student is NOT forced to memorize hard off-policy
                        # solutions (which degraded accuracy); it only receives the
                        # ensemble's soft correct-path distribution as guidance.
                        items.append((sidx, rids, ans, True, True))
                        n_hv_tr += 1
                    n_hv_q += 1
                print(f"[Harvest] added {n_hv_tr} ensemble-correct traces over {n_hv_q} "
                      f"recovered failure questions (coverage expansion)")
        else:
            # ---- Off-policy: ensemble models generate the traces (original path) --
            correct_traces = defaultdict(dict)   # sidx -> {text: (rids, answer)}
            nb = (len(perturbs) + n_eng - 1) // n_eng
            print(f"[Pass1] diverse generation T={args.gen_temperature} n={args.gen_n_samples}, "
                  f"{len(perturbs)} models, {nb} batches")
            for b in range(nb):
                start = b * n_eng
                batch = perturbs[start:start + n_eng]
                t0 = time.time()
                ray.get([engines[i].collective_rpc.remote("perturb_self_weights", args=(int(s), sg, False))
                         for i, (s, sg) in enumerate(batch)])
                outs = ray.get([engines[i].generate.remote(prompts, gen_sp, use_tqdm=False)
                                for i in range(len(batch))])
                ray.get([engines[i].collective_rpc.remote("restore_self_weights", args=(int(s), sg, False))
                         for i, (s, sg) in enumerate(batch)])
                for li in range(len(batch)):
                    for sidx, o in enumerate(outs[li]):
                        gt = datas[sidx]["ground_truth"]
                        for comp in o.outputs:
                            txt = comp.text
                            if ans_marker is not None and ans_marker not in txt:
                                continue
                            ans = handler.extract_answer(txt)
                            if ans:
                                votes[sidx][str(ans).strip()] += 1
                            if not handler.is_answer_correct(txt, gt):
                                continue
                            if txt not in correct_traces[sidx]:
                                correct_traces[sidx][txt] = (list(comp.token_ids), str(ans).strip())
                print(f"  [Pass1] batch {b+1}/{nb} ({time.time()-t0:.1f}s) "
                      f"questions w/correct={len(correct_traces)}")

            for sidx in sorted(correct_traces.keys()):
                traces = correct_traces[sidx]
                if not traces:
                    continue
                maj_ans, maj_cnt = (votes[sidx].most_common(1)[0]
                                    if votes[sidx] else (None, 0))
                if maj_ans is not None and maj_cnt < args.min_votes:
                    n_majority_skipped += 1
                    continue
                cand = []
                for txt, (rids, ans) in traces.items():
                    matched = (maj_ans is not None and ans == maj_ans)
                    cand.append((not matched, len(rids), txt, rids, ans, matched))
                cand.sort(key=lambda x: (x[0], x[1]))
                if args.require_majority_match:
                    matched_cand = [c for c in cand if c[5]]
                    if matched_cand:
                        cand = matched_cand
                kept = cand[:max(1, args.n_canonical)]
                for _, _, _txt, rids, ans, matched in kept:
                    items.append((sidx, rids, ans, matched, False))
                n_q_kept += 1

            print(f"[Select] kept {len(items)} traces across {n_q_kept} questions "
                  f"(skipped {n_majority_skipped} low-vote); "
                  f"avg {len(items)/max(1,n_q_kept):.2f} traces/question")
        if not items:
            raise RuntimeError("No traces found to build soft labels")

        # ---- pure-CE fast path: skip Pass2, emit empty soft targets -----------
        if args.skip_teacher:
            rows = []
            for (sidx, rids, ans, matched, is_harvest) in items:
                pids = prompt_ids_list[sidx]
                rows.append({
                    "prompt": prompts[sidx],
                    "prompt_ids": [int(x) for x in pids],
                    "response_ids": [int(x) for x in rids],
                    "teacher_ids": [[] for _ in rids],
                    "teacher_probs": [[] for _ in rids],
                    "ground_truth": _gt_for_parquet(datas[sidx]["ground_truth"]),
                    "matched_majority": bool(matched),
                    "is_harvest": bool(is_harvest),
                })
            kd_dir = f"{out_dir}/kd"
            os.makedirs(kd_dir, exist_ok=True)
            pq = f"{kd_dir}/train_kd.parquet"
            pd.DataFrame(rows).to_parquet(pq)
            meta = {
                "dataset": args.dataset, "base_model": base_model_path,
                "ensemble_size": len(perturbs), "questions": n_q_kept,
                "traces": len(rows), "total": len(datas),
                "on_policy": bool(args.on_policy), "skip_teacher": True,
                "harvest_failures": bool(args.harvest_failures),
            }
            with open(f"{out_dir}/meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[KD/skip_teacher] saved {len(rows)} CE-only rows -> {pq}")
            print("SUCCESS: KD soft-label generation complete")
            return

        # Build full token sequences and response slices per item.
        full_ids = []
        resp_slice = []
        for (sidx, rids, _ans, _m, _h) in items:
            pids = prompt_ids_list[sidx]
            full_ids.append(list(pids) + list(rids))
            resp_slice.append((len(pids), len(pids) + len(rids)))

        # accumulators: item -> list(over response positions) of dict{token_id: prob_sum}
        accum = [[defaultdict(float) for _ in range(s1 - s0)] for (s0, s1) in resp_slice]

        # ---------------- Pass 2: teacher-force, average distributions -------
        tf_sp = SamplingParams(temperature=0.0, max_tokens=1,
                               prompt_logprobs=args.teacher_topk, detokenize=False)
        token_prompts = [TokensPrompt(prompt_token_ids=fid) for fid in full_ids]
        n_items = len(items)
        # Chunk the sequences so a single ray.get never materializes the full
        # (n_items x positions x topk) prompt_logprobs object on the driver --
        # at scale (~20k traces) that is ~10^8 python objects per engine, which
        # blows up driver RAM and stalls. Chunking bounds it and gives progress.
        chunk = max(256, int(args.pass2_chunk))
        n_chunks = (n_items + chunk - 1) // chunk
        # number of perturb batches over the ensemble (independent of Pass1 path)
        nb = (len(perturbs) + n_eng - 1) // n_eng
        print(f"[Pass2] teacher-forcing {n_items} sequences over {len(perturbs)} models "
              f"in {n_chunks} chunk(s) of {chunk}")

        def accumulate(out_li, offset):
            # Fold one model's prompt_logprobs (for a chunk) into per-position sums.
            for j in range(len(out_li)):
                it = offset + j
                plogprobs = out_li[j].prompt_logprobs  # len == len(full_ids[it])
                s0, s1 = resp_slice[it]
                acc = accum[it]
                for pos in range(s0, s1):
                    d = plogprobs[pos]
                    if not d:
                        continue
                    bucket = acc[pos - s0]
                    for tid, lp in d.items():
                        logp = lp.logprob if hasattr(lp, "logprob") else float(lp)
                        bucket[int(tid)] += math.exp(logp)

        import gc
        for b in range(nb):
            start = b * n_eng
            batch = perturbs[start:start + n_eng]
            t0 = time.time()
            ray.get([engines[i].collective_rpc.remote("perturb_self_weights", args=(int(s), sg, False))
                     for i, (s, sg) in enumerate(batch)])
            for c in range(n_chunks):
                c0 = c * chunk
                c1 = min(c0 + chunk, n_items)
                tp_chunk = token_prompts[c0:c1]
                # Kick off all engines in parallel on this chunk, but pull + fold +
                # free results ONE engine at a time so the driver never holds all
                # engines' prompt_logprobs simultaneously.
                refs = [engines[i].generate.remote(tp_chunk, tf_sp, use_tqdm=False)
                        for i in range(len(batch))]
                for li, ref in enumerate(refs):
                    out_li = ray.get(ref)
                    accumulate(out_li, c0)
                    del out_li
                    refs[li] = None
                    gc.collect()
                print(f"  [Pass2] batch {b+1}/{nb} chunk {c+1}/{n_chunks} "
                      f"({time.time()-t0:.1f}s elapsed)")
            ray.get([engines[i].collective_rpc.remote("restore_self_weights", args=(int(s), sg, False))
                     for i, (s, sg) in enumerate(batch)])
            print(f"  [Pass2] batch {b+1}/{nb} done ({time.time()-t0:.1f}s)")

        # ---------------- finalize: normalize + top-K per position -----------
        rows = []
        for it, (sidx, rids, ans, matched, is_harvest) in enumerate(items):
            pids = prompt_ids_list[sidx]
            t_ids, t_probs = [], []
            for pos_bucket in accum[it]:
                if not pos_bucket:
                    t_ids.append([])
                    t_probs.append([])
                    continue
                ranked = sorted(pos_bucket.items(), key=lambda kv: kv[1], reverse=True)[:args.out_topk]
                ids = [int(i) for i, _ in ranked]
                probs = np.array([p for _, p in ranked], dtype=np.float64)
                probs = probs / max(probs.sum(), 1e-12)  # renormalize over kept top-K
                t_ids.append(ids)
                t_probs.append([float(x) for x in probs])
            rows.append({
                "prompt": prompts[sidx],
                "prompt_ids": [int(x) for x in pids],
                "response_ids": [int(x) for x in rids],
                "teacher_ids": t_ids,
                "teacher_probs": t_probs,
                "ground_truth": _gt_for_parquet(datas[sidx]["ground_truth"]),
                "matched_majority": bool(matched),
                "is_harvest": bool(is_harvest),
            })

        kd_dir = f"{out_dir}/kd"
        os.makedirs(kd_dir, exist_ok=True)
        df = pd.DataFrame(rows)
        pq = f"{kd_dir}/train_kd.parquet"
        df.to_parquet(pq)
        n_matched = int(sum(1 for r in rows if r["matched_majority"]))
        meta = {
            "dataset": args.dataset, "base_model": base_model_path,
            "ensemble_size": len(perturbs), "questions": n_q_kept,
            "traces": len(rows), "matched_majority": n_matched,
            "total": len(datas), "teacher_topk": args.teacher_topk,
            "out_topk": args.out_topk, "max_tokens": max_tokens,
            "n_canonical": args.n_canonical, "gen_temperature": args.gen_temperature,
            "gen_n_samples": args.gen_n_samples,
            "require_majority_match": bool(args.require_majority_match),
            "on_policy": bool(args.on_policy),
            "keep_incorrect_frac": args.keep_incorrect_frac,
            "harvest_failures": bool(args.harvest_failures),
            "harvest_n_canonical": args.harvest_n_canonical,
        }
        with open(f"{out_dir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[KD] saved {len(rows)} rows ({n_matched} match majority) -> {pq}")
        print("SUCCESS: KD soft-label generation complete")
    finally:
        cleanup_engines(engines, pgs)


if __name__ == "__main__":
    main()
