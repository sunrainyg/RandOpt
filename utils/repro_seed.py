"""
Reproduce a perturbed model from saved seed information.

Usage:
    python reproduce_model_from_seed.py --seeds_info_path <path> --rank 1 --output_path <path>
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def perturb_model_weights(model, seed, sigma, is_negative=False):
    """
    Apply perturbation to model weights using a deterministic seed.
    
    Args:
        model: The model to perturb
        seed: Random seed for reproducibility
        sigma: Perturbation magnitude
        is_negative: If True, apply -noise instead of +noise (for antithetic sampling)
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    sign = -1.0 if is_negative else 1.0
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.randn(param.shape, generator=rng, dtype=param.dtype) * sigma * sign
                param.add_(noise.to(param.device))


def reproduce_model(seeds_info_path, rank, output_path=None, precision="bfloat16"):
    """
    Reproduce a perturbed model from saved seed information.
    
    Supports both old and new formats:
    - Old format: sigma is global (from es_ensemble_fix_ensemble.py)
    - New format: sigma is per-model with is_negative flag (from es_ensemble_ensemble_parallel.py)
    
    Args:
        seeds_info_path: Path to the top_k_seeds.json file
        rank: Which top-k model to reproduce (1 for best, 2 for second best, etc.)
        output_path: Optional path to save the reproduced model
        precision: Model precision (bfloat16 or float16)
    
    Returns:
        perturbed_model: The reproduced perturbed model
        tokenizer: The tokenizer
    """
    # Load seeds info
    with open(seeds_info_path, "r") as f:
        seeds_info = json.load(f)
    
    # Get the seed for the requested rank
    top_k_models = seeds_info["top_k_models"]
    if rank < 1 or rank > len(top_k_models):
        raise ValueError(f"Rank must be between 1 and {len(top_k_models)}")
    
    model_info = top_k_models[rank - 1]
    seed = model_info["seed"]
    train_reward = model_info["train_reward"]
    base_model_path = seeds_info["base_model_path"]
    
    # Support both old and new formats
    # Old format: sigma is global
    # New format: sigma is per-model
    if "sigma" in model_info:
        sigma = model_info["sigma"]
        is_negative = model_info.get("is_negative", False)
    else:
        sigma = seeds_info["sigma"]
        is_negative = False
    
    print(f"\n=== Reproducing Model ===")
    print(f"Rank: {rank}/{len(top_k_models)}")
    print(f"Seed: {seed}")
    print(f"Sigma: {sigma}")
    if is_negative:
        print(f"Direction: -noise (antithetic)")
    else:
        print(f"Direction: +noise")
    print(f"Train Reward: {train_reward:.4f}")
    print(f"Base Model: {base_model_path}")
    
    # Load base model
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    print(f"\nLoading base model with {precision} precision...")
    perturbed_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Apply perturbation
    sign_str = "-noise" if is_negative else "+noise"
    print(f"Applying perturbation with seed {seed} ({sign_str})...")
    perturb_model_weights(perturbed_model, seed, sigma, is_negative)
    
    print(f"✓ Model successfully reproduced!")
    
    # Optionally save the model
    if output_path:
        print(f"\nSaving reproduced model to {output_path}...")
        perturbed_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"✓ Model saved!")
    
    return perturbed_model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Reproduce a perturbed model from seed")
    parser.add_argument("--seeds_info_path", type=str, required=True,
                        help="Path to top_k_seeds.json file")
    parser.add_argument("--rank", type=int, required=True,
                        help="Which top-k model to reproduce (1=best, 2=second, ...)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional: Path to save the reproduced model")
    parser.add_argument("--precision", type=str, choices=["float16", "bfloat16"],
                        default="bfloat16", help="Model precision")
    args = parser.parse_args()
    
    model, tokenizer = reproduce_model(
        args.seeds_info_path,
        args.rank,
        args.output_path,
        args.precision
    )
    
    print("\n" + "="*60)
    print("Model is ready to use!")
    print("You can now use it for inference or further fine-tuning.")
    print("="*60)
    
    # Display model info
    if hasattr(model, 'num_parameters'):
        print(f"\nModel has {model.num_parameters():,} parameters")


if __name__ == "__main__":
    main()

