# Seed LoRA adapters vs full-matrix perturbation

Results for the seed LoRA adapters produced by `scripts/make_seed_lora.py`
(see [Serve many seeds at once as LoRA adapters](../../README.md#serve-many-seeds-at-once-as-lora-adapters)).

Qwen2.5-32B-Instruct, greedy decoding, rank 16 / alpha 16 adapters on all
attention and MLP projections. pass@k is the unbiased estimator of
[Chen et al. 2021](https://arxiv.org/abs/2107.03374) over the n generations per
example (one per seed, or one per sample for the base-model rows); pass@n is
the coverage.

**Facts-search** (multi-hop retrieval agent, 200 test examples, seeds ranked on
a disjoint 200-example train split):

| Setting | n | pass@1 | pass@2 | pass@5 | pass@10 | pass@n |
|---|---|---|---|---|---|---|
| Base model, T=1.0 samples | 20 | 0.436 | 0.457 | 0.478 | 0.490 | 0.500 |
| Full-matrix perturbation, sigma 0.001 | 18 | 0.429 | 0.500 | 0.572 | 0.609 | 0.630 |
| Seed LoRAs, `noise_scale=0.001` | 10 | 0.450 | 0.530 | 0.612 | 0.660 | 0.660 |

Seed LoRAs kept pass@1 at the base model's level and gave the best pass@k at
every k, with fewer seeds than the full-matrix run.

**Countdown** (arithmetic puzzle, 20 test examples, seeds ranked on 80 train
examples):

| Setting | n | pass@1 | pass@2 | pass@5 | pass@10 | Majority vote |
|---|---|---|---|---|---|---|
| Base model, T=0.6 samples | 10 | 0.490 | 0.619 | 0.776 | 0.900 | 0.70 |
| Full-matrix perturbation, sigma 0.001, top-10 of 100 | 10 | 0.525 | 0.647 | 0.769 | 0.800 | 0.75 |
| Seed LoRAs, `noise_scale=0.01` | 6 | degenerate | - | - | - | - |
| Seed LoRAs, `noise_scale=0.001` | 6 | coherent, low diversity | - | - | - | - |

At `noise_scale=0.01` five or six of the six seed LoRAs produced repetitive,
non-terminating output on Countdown prompts, so no accuracy is reported. At
`noise_scale=0.001` every seed was coherent, but seeds differed only modestly
from each other and we did not run the full evaluation. On a reasoning task
like Countdown the low-rank concentration bites harder than on the agentic
facts-search task, so treat the LoRA path as a serving convenience whose sigma
must be re-tuned per task, not as a drop-in replacement for full-matrix seeds.
