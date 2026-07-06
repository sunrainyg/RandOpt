"""Console entry point: `iter_randopt` / `python -m iterative_randopt`.

    iter_randopt --config configs/iterative_randopt_gsm8k.yaml
    iter_randopt --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k --rounds 4
"""
from __future__ import annotations

import argparse
import dataclasses

from .config import IterativeRandOptConfig
from .pipeline import run


def main():
    p = argparse.ArgumentParser(description="Iterative RandOpt (single-node)")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(IterativeRandOptConfig):
        if f.type in ("int", int):
            p.add_argument(f"--{f.name}", type=int, default=None)
        elif f.type in ("float", float):
            p.add_argument(f"--{f.name}", type=float, default=None)
        elif f.type in ("bool", bool):
            p.add_argument(f"--{f.name}", type=lambda x: str(x).lower() in ("1", "true", "yes"),
                           default=None)
        else:
            p.add_argument(f"--{f.name}", type=str, default=None)
    args = p.parse_args()

    cfg = IterativeRandOptConfig.from_yaml(args.config) if args.config else IterativeRandOptConfig()
    for f in dataclasses.fields(IterativeRandOptConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)

    print("[iter-randopt] config:")
    for k, v in cfg.to_dict().items():
        print(f"    {k} = {v}")
    run(cfg)


if __name__ == "__main__":
    main()
