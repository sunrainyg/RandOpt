#!/usr/bin/env python3
"""
Smoke test for the randopt_server perturbation endpoints.

Checks both ways of undoing a perturbation:

  1. baseline inference
  2. /store_base
  3. /perturb (seed, sigma)      -> perturbed inference
  4. /restore (same seed, sigma) -> inference should match baseline
  5. /perturb (seed, sigma)      -> perturbed inference
  6. /reset                      -> inference should match baseline

Greedy decoding on a short prompt is a weak check: a small sigma may not change
the argmax token, and /restore can differ from baseline by a rounding ulp in
bf16 or fp8. Treat a /reset mismatch as a bug; treat a /restore mismatch as a
hint to prefer /store_base + /reset for low-precision weights.

Run:
  python3 scripts/test_perturb_restore.py --model Qwen/Qwen2.5-32B-Instruct \\
      [--base-url http://localhost:8000] [--seed 12345] [--sigma 0.001]
"""

import argparse
import sys

import requests

PROMPT = "What is 2 + 2? Answer with just the number."
MAX_TOKENS = 16


def _chat(base_url, model, prompt):
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _post(base_url, route, body=None):
    resp = requests.post(f"{base_url}{route}", json=body or {}, timeout=600)
    resp.raise_for_status()
    print(f"  {route} -> {resp.json()}")


def _perturb_then_undo(base_url, model, seed, sigma, undo_route, undo_body):
    print(f"  /perturb seed={seed} sigma={sigma}")
    _post(base_url, "/perturb", {"seed": seed, "sigma": sigma})
    perturbed = _chat(base_url, model, PROMPT)
    print(f"  perturbed: {perturbed!r}")
    _post(base_url, undo_route, undo_body)
    undone = _chat(base_url, model, PROMPT)
    print(f"  after {undo_route}: {undone!r}")
    return perturbed, undone


def _report(label, baseline, perturbed, undone):
    changed = "different" if perturbed != baseline else "same (sigma too small to flip argmax?)"
    matched = "matches baseline" if undone == baseline else "MISMATCH"
    print(f"{label:9}: perturbed {changed}; undone {matched}")
    return undone == baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="served model name for /v1/chat/completions")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sigma", type=float, default=0.001)
    args = parser.parse_args()
    url = args.base_url.rstrip("/")

    print(f"Target: {url}  model: {args.model}\nPrompt: {PROMPT!r}\n")

    print("Step 1: baseline")
    baseline = _chat(url, args.model, PROMPT)
    print(f"  baseline: {baseline!r}\n")

    print("Step 2: /store_base")
    _post(url, "/store_base")

    print("\nSteps 3-4: /perturb then /restore")
    restore_body = {"seed": args.seed, "sigma": args.sigma}
    perturbed_a, restored = _perturb_then_undo(url, args.model, args.seed, args.sigma, "/restore", restore_body)

    print("\nSteps 5-6: /perturb then /reset")
    perturbed_b, reset = _perturb_then_undo(url, args.model, args.seed, args.sigma, "/reset", None)

    print("\n" + "=" * 60)
    print(f"baseline : {baseline!r}")
    restore_ok = _report("/restore", baseline, perturbed_a, restored)
    reset_ok = _report("/reset", baseline, perturbed_b, reset)

    if not reset_ok:
        print("\nFAIL: /reset did not reproduce the baseline output.")
        sys.exit(1)
    if not restore_ok:
        print("\nWARNING: /restore differs from baseline (expected to within rounding on bf16/fp8).")


if __name__ == "__main__":
    main()
