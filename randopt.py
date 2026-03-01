#!/usr/bin/env python3
"""
RandOpt simplified. Fully parallelized.
supports multiple datasets and models.
supports resume from previous run.
"""

import argparse
from collections import Counter
from datetime import datetime
import gc
import json
import os
import random
from typing import List, Dict, Tuple

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from data_handlers import get_dataset_handler, list_datasets
from core import launch_engines, cleanup_engines


def parse_args():
    parser = argparse.ArgumentParser(
        description="RandOpt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=list_datasets(),
                        help="Dataset to use")
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="Override default train data path")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Override default test data path")
    parser.add_argument("--train_samples", type=int, default=200,
                        help="Number of train samples for perturbation selection")
    parser.add_argument("--test_samples", type=int, default=None,
                        help="Max test samples (None = all)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--precision", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Override default max_tokens for dataset")
    parser.add_argument("--sigma_values", type=str, default="0.0001,0.0005,0.001,0.002,0.005,0.01",
                        help="Comma-separated sigma values")
    parser.add_argument("--population_size", type=int, default=30,
                        help="Total number of perturbations to evaluate")
    parser.add_argument("--top_k_ratios", type=str, default="0.01,0.05,0.1",
                        help="Comma-separated ratios of population_size (e.g., '0.01,0.05,0.1' for 1%,5%,10%)")
    parser.add_argument("--num_engines", type=int, default=4,
                        help="Number of vLLM engines (typically = num GPUs / tp)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size per engine (use 2+ for 7B+ models)")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--experiment_dir", type=str, default="es-experiment")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Resume from a previous run directory (skips sampling, goes directly to ensemble eval)")
    
    args = parser.parse_args()
    
    args.sigma_list = [float(s.strip()) for s in args.sigma_values.split(",")]
    ratios = [float(r.strip()) for r in args.top_k_ratios.split(",")]
    args.top_k_list = sorted(set(max(1, int(r * args.population_size)) for r in ratios), reverse=True)
    args.max_top_k = args.top_k_list[0]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    torch.cuda.manual_seed_all(args.global_seed)
    
    return args


def load_data(handler, args):
    """Load train and test data."""
    train_path = args.train_data_path or handler.default_train_path
    test_path = args.test_data_path or handler.default_test_path
    
    print(f"Loading {handler.name} data...")
    if train_path == test_path:
        # Same file - split by index (e.g., MATH500)
        all_data = handler.load_data(train_path, split="train", max_samples=None)
        train_datas = all_data[:args.train_samples]
        test_datas = all_data[args.train_samples:] if args.test_samples is None else all_data[args.train_samples:args.train_samples + args.test_samples]
        if len(test_datas) < 50:
            print(f"  Warning: Only {len(test_datas)} test samples. Using all for both.")
            test_datas = all_data
    else:
        train_datas = handler.load_data(train_path, split="train", max_samples=args.train_samples)
        test_datas = handler.load_data(test_path, split="test", max_samples=args.test_samples)
    
    print(f"  Train: {len(train_datas)} | Test: {len(test_datas)}")
    return train_datas, test_datas


def evaluate_base_model(engines, handler, train_prompts, test_prompts, train_datas, test_datas, sampling_params):
    """Evaluate base model on train and test sets."""
    print(f"\n{'='*60}\nBASE MODEL EVALUATION\n{'='*60}")
    
    train_outputs = ray.get(engines[0].generate.remote(train_prompts, sampling_params, use_tqdm=False))
    base_train = handler.postprocess_outputs(train_outputs, train_datas)
    print(f"Train: {base_train*100:.2f}%")
    
    test_outputs = ray.get(engines[0].generate.remote(test_prompts, sampling_params, use_tqdm=False))
    base_test = handler.postprocess_outputs(test_outputs, test_datas)
    print(f"Test:  {base_test*100:.2f}%")
    
    return base_train, base_test


def run_sampling(args, engines, handler, train_prompts, train_datas, sampling_params):
    """Run perturbation sampling."""
    print(f"\n{'='*60}\nPERTURBATION SAMPLING\n{'='*60}")
    print(f"Budget: {args.population_size} | Sigmas: {args.sigma_list}")
    
    rng = np.random.default_rng(seed=args.global_seed)
    perf: Dict[Tuple[int, float], float] = {}
    
    # Pre-generate unique seeds and sigmas
    all_seeds = rng.choice(2**31, size=args.population_size, replace=False).tolist()
    all_sigmas = rng.choice(args.sigma_list, size=args.population_size).tolist()
    seed_idx = 0
    
    samples_evaluated, batch_idx = 0, 0
    
    while samples_evaluated < args.population_size:
        batch_size = min(args.num_engines, args.population_size - samples_evaluated)
        batch = [(all_seeds[seed_idx + i], all_sigmas[seed_idx + i]) for i in range(batch_size)]
        seed_idx += batch_size
        
        # Evaluate batch
        ray.get([engines[i].collective_rpc.remote("perturb_self_weights", args=(int(s), sig, False)) 
                 for i, (s, sig) in enumerate(batch)])
        
        outputs = ray.get([engines[i].generate.remote(train_prompts, sampling_params, use_tqdm=False) 
                          for i in range(len(batch))])
        
        ray.get([engines[i].collective_rpc.remote("restore_self_weights", args=(int(s), sig, False)) 
                 for i, (s, sig) in enumerate(batch)])
        
        # Process results
        rewards = []
        for i, (seed, sigma) in enumerate(batch):
            r = handler.postprocess_outputs(outputs[i], train_datas)
            perf[(seed, sigma)] = r
            rewards.append(r)
        
        samples_evaluated += len(batch)
        batch_idx += 1
        print(f"  Batch {batch_idx} | {samples_evaluated}/{args.population_size} | {['%.3f' % r for r in rewards]}")
    
    print(f"\nSampling done.")
    
    # Summary
    print(f"\n{'='*60}\nSAMPLING COMPLETE\n{'='*60}")
    
    # Calculate sigma stats
    sigma_rewards: Dict[float, List[float]] = {s: [] for s in args.sigma_list}
    for (seed, sigma), reward in perf.items():
        sigma_rewards[sigma].append(reward)
    
    for sigma in args.sigma_list:
        rewards_list = sigma_rewards[sigma]
        if rewards_list:
            print(f"  σ={sigma}: mean={np.mean(rewards_list):.4f}, n={len(rewards_list)}")
    
    # Find best sigma
    best_sigma = max(args.sigma_list, key=lambda s: np.mean(sigma_rewards[s]) if sigma_rewards[s] else 0)
    print(f"\n★ Best sigma: {best_sigma}")
    
    return perf, best_sigma


def run_ensemble_evaluation(args, engines, handler, test_prompts, test_datas, top_k_perturbs, sampling_params, base_test):
    """Run ensemble evaluation with majority voting. Memory-efficient version."""
    max_k = min(args.max_top_k, len(top_k_perturbs))
    num_samples = len(test_datas)
    print(f"\n{'='*60}\nENSEMBLE EVALUATION\n{'='*60}")
    eval_k_values = [k for k in args.top_k_list if k <= max_k]
    print(f"K values: {eval_k_values} | Test samples: {num_samples}")
    
    # Memory-efficient: only store extracted answers (strings), not full text
    # all_answers[model_idx][sample_idx] = answer_string (or "" if invalid)
    all_answers = [None] * max_k
    
    total_batches = (max_k + args.num_engines - 1) // args.num_engines
    
    for batch_idx in range(total_batches):
        start, end = batch_idx * args.num_engines, min((batch_idx + 1) * args.num_engines, max_k)
        batch_perturbs = top_k_perturbs[start:end]
        
        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch_perturbs)} models)...", flush=True)
        
        ray.get([engines[i].collective_rpc.remote("perturb_self_weights", args=(int(s), sig, False)) 
                 for i, (s, sig) in enumerate(batch_perturbs)])
        
        batch_outputs = ray.get([engines[i].generate.remote(test_prompts, sampling_params, use_tqdm=False) 
                                 for i in range(len(batch_perturbs))])
        
        ray.get([engines[i].collective_rpc.remote("restore_self_weights", args=(int(s), sig, False)) 
                 for i, (s, sig) in enumerate(batch_perturbs)])
        
        # Extract answers immediately and discard outputs to save memory
        for local_idx, global_idx in enumerate(range(start, end)):
            outputs = batch_outputs[local_idx]
            answers_for_model = []
            for i in range(num_samples):
                response_text = outputs[i].outputs[0].text
                # For countdown: extract_answer_for_voting returns (answer, is_valid, reject_info)
                if handler.name == "countdown":
                    numbers = test_datas[i].get("numbers")
                    answer, is_valid, _ = handler.extract_answer_for_voting(response_text, numbers=numbers)
                    answers_for_model.append(answer if is_valid else "")
                elif hasattr(handler, 'extract_answer_for_voting'):
                    answer = handler.extract_answer_for_voting(response_text)
                    answers_for_model.append(answer or "")
                else:
                    answer = handler.extract_answer(response_text)
                    answers_for_model.append(answer or "")
            all_answers[global_idx] = answers_for_model
        
        # Immediately free batch outputs
        del batch_outputs
        gc.collect()
    
    print(f"\nGeneration completed.")
    
    # Evaluate for each K value (majority voting)
    print(f"\nMajority voting...")
    ensemble_results = {}
    
    for k_value in eval_k_values:
        correct = 0
        
        for idx, data in enumerate(test_datas):
            # Collect answers from top-k models (only non-empty)
            answers = [all_answers[m][idx] for m in range(k_value) if all_answers[m][idx]]
            
            if answers:
                counter = Counter(answers)
                final = counter.most_common(1)[0][0]
                if hasattr(handler, 'is_voted_answer_correct'):
                    is_correct = handler.is_voted_answer_correct(final, data["ground_truth"])
                else:
                    formatted = handler.format_answer_for_check(final)
                    is_correct = handler.is_answer_correct(formatted, data["ground_truth"])
                if is_correct:
                    correct += 1
        
        acc = correct / num_samples * 100
        ensemble_results[k_value] = {"accuracy": acc, "correct": correct}
        print(f"  K={k_value}: {acc:.2f}% ({correct}/{num_samples}) [+{acc - base_test*100:.2f}%]")
    
    # Clean up all_answers after evaluation
    del all_answers
    gc.collect()
    
    return ensemble_results


def save_results(args, logging_dir, model_saves_dir, base_model_path, handler, 
                 base_train, base_test, top_k_perturbs, top_k_rewards, 
                 ensemble_results, perf, best_sigma):
    print(f"\n=== Saving Results ===")
    
    seeds_info = {
        "base_model_path": base_model_path,
        "best_sigma": best_sigma,
        "top_k_models": [
            {"rank": i+1, "seed": int(seed), "sigma": float(sigma), "train_reward": float(reward)}
            for i, ((seed, sigma), reward) in enumerate(zip(top_k_perturbs, top_k_rewards))
        ],
    }
    with open(f"{model_saves_dir}/top_k_seeds.json", "w") as f:
        json.dump(seeds_info, f, indent=4)
    
    # Calculate sigma stats from perf
    sigma_rewards: Dict[float, List[float]] = {s: [] for s in args.sigma_list}
    for (seed, sigma), reward in perf.items():
        sigma_rewards[sigma].append(reward)
    sigma_stats = {
        str(s): {"mean": float(np.mean(sigma_rewards[s])) if sigma_rewards[s] else 0.0, 
                 "count": len(sigma_rewards[s])} 
        for s in args.sigma_list
    }
    
    # Save full results
    results = {
        "dataset": args.dataset,
        "model": args.model_name,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "base_train_accuracy": base_train,
        "base_test_accuracy": base_test,
        "sigma_stats": sigma_stats,
        "best_sigma": best_sigma,
        "ensemble_results": {str(k): v for k, v in ensemble_results.items()},
        "top_k_perturbs": [(int(s), float(sig)) for s, sig in top_k_perturbs],
        "top_k_train_rewards": [float(r) for r in top_k_rewards],
    }
    
    with open(f"{logging_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {logging_dir}/")


def main(args):
    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens
    
    is_resume = args.resume_dir is not None
    
    print(f"{'='*60}")
    print(f"ES Ensemble - {handler.name.upper()} {'[RESUME]' if is_resume else ''}")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Population: {args.population_size} | Top-K: {args.top_k_list} | Engines: {args.num_engines} | TP: {args.tp}")
    
    # Ray setup
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(address="local", ignore_reinit_error=True)
    
    if is_resume:
        # Resume mode: load saved seeds
        with open(f"{args.resume_dir}/model_saves/top_k_seeds.json", "r") as f:
            saved = json.load(f)
        base_model_path = saved["base_model_path"]
        best_sigma = saved["best_sigma"]
        top_k_perturbs = [(m["seed"], m["sigma"]) for m in saved["top_k_models"]]
        top_k_rewards = [m["train_reward"] for m in saved["top_k_models"]]
        
        # Load previous results for base accuracies
        with open(f"{args.resume_dir}/results.json", "r") as f:
            prev_results = json.load(f)
        base_train, base_test = prev_results["base_train_accuracy"], prev_results["base_test_accuracy"]
        perf = {(s, sig): r for (s, sig), r in zip(top_k_perturbs, top_k_rewards)}
        
        logging_dir = f"{args.experiment_dir}/{args.dataset}_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_saves_dir = f"{logging_dir}/model_saves"
        os.makedirs(model_saves_dir, exist_ok=True)
        
        print(f"Resumed from: {args.resume_dir}")
        print(f"Base model: {base_model_path}")
        print(f"Loaded {len(top_k_perturbs)} perturbations")
    else:
        # Training mode: setup directories and save model
        logging_dir = f"{args.experiment_dir}/{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_saves_dir = f"{logging_dir}/model_saves"
        os.makedirs(model_saves_dir, exist_ok=True)
    
    with open(f"{logging_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Load data
    train_datas, test_datas = load_data(handler, args)
    
    if not is_resume:
        base_model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    is_instruct_model = any(x in args.model_name.lower() for x in ['instruct', 'chat', 'it'])
    
    def format_prompt(messages):
        if is_instruct_model and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"
    
    train_prompts = [format_prompt(d["messages"]) for d in train_datas]
    test_prompts = [format_prompt(d["messages"]) for d in test_datas]
    sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=max_tokens)
    
    # Launch engines
    engines, pgs = launch_engines(args.num_engines, base_model_path, precision=args.precision, tensor_parallel_size=args.tp)
    
    try:
        if not is_resume:
            base_train, base_test = evaluate_base_model(
                engines, handler, train_prompts, test_prompts, train_datas, test_datas, sampling_params)
            
            # Perturbation sampling
            perf, best_sigma = run_sampling(
                args, engines, handler, train_prompts, train_datas, sampling_params)
            
            # Selection: Get top-k by sorting all results by score
            print(f"\n{'='*60}\nSELECTION\n{'='*60}")
            sorted_perturbs = sorted(perf.items(), key=lambda x: x[1], reverse=True)
            top_k_perturbs = [(seed, sigma) for (seed, sigma), _ in sorted_perturbs[:args.max_top_k]]
            top_k_rewards = [reward for _, reward in sorted_perturbs[:args.max_top_k]]
            
            print(f"Selected top-{args.max_top_k} from {args.population_size} perturbations")
            print(f"\n=== Top-{args.max_top_k} Perturbations ===")
            for i, ((seed, sigma), reward) in enumerate(sorted_perturbs[:10]):
                print(f"  {i+1}. seed={seed}, σ={sigma}: {reward:.4f}")
        else:
            print(f"\n=== Loaded Top-{len(top_k_perturbs)} Perturbations ===")
            for i, ((seed, sigma), reward) in enumerate(zip(top_k_perturbs[:10], top_k_rewards[:10])):
                print(f"  {i+1}. seed={seed}, σ={sigma}: {reward:.4f}")
        
        # Ensemble evaluation
        ensemble_results = run_ensemble_evaluation(
            args, engines, handler, test_prompts, test_datas, top_k_perturbs, sampling_params, base_test)
        
        save_results(args, logging_dir, model_saves_dir, base_model_path, handler,
                    base_train, base_test, top_k_perturbs, top_k_rewards,
                    ensemble_results, perf, best_sigma)
    
    finally:
        cleanup_engines(engines, pgs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
