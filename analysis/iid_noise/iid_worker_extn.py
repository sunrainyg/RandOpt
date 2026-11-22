import gc
import os
import zlib

import torch

from core.engine import RandOptNcclLLM
from utils.worker_extn import WorkerExtension


class IIDWorkerExtension(WorkerExtension):

    def _tp_sharded_names(self) -> set:
        if not hasattr(self, "_sharded_names"):
            from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
            from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
            names = set()
            for mod_name, mod in self.model_runner.model.named_modules():
                prefix = f"{mod_name}." if mod_name else ""
                for p_name, _ in mod.named_parameters(recurse=False):
                    if isinstance(mod, (ColumnParallelLinear, VocabParallelEmbedding)) or (
                            isinstance(mod, RowParallelLinear) and p_name != "bias"):
                        names.add(prefix + p_name)
            self._sharded_names = names
        return self._sharded_names

    def _noise(self, name, p, seed):
        """Standard-normal noise for parameter `name` under perturbation `seed` (same dtype/device as p)."""
        rank = 0
        if name in self._tp_sharded_names():
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        tensor_seed = ((int(seed) * 1_000_003 + zlib.crc32(name.encode())) * 64 + rank) % (2 ** 63)
        gen = torch.Generator(device=p.device)
        gen.manual_seed(tensor_seed)
        return torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        coef = (-1.0 if negate else 1.0) * float(noise_scale)
        for name, p in self.model_runner.model.named_parameters():
            if self._should_perturb(name):
                p.data.add_(coef * self._noise(name, p, seed))
        self._sync()
        return True

    def restore_self_weights(self, seed, SIGMA, negate=False):
        """Undo perturbation. Must use same negate value as perturb_self_weights."""
        self._set_seed(seed)
        coef = (-1.0 if negate else 1.0) * float(SIGMA)
        for name, p in self.model_runner.model.named_parameters():
            if self._should_perturb(name):
                p.data.add_(-coef * self._noise(name, p, seed))
        self._sync()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        for name, p in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                continue
            acc = torch.zeros_like(p.data, dtype=torch.float32)
            for seed, c in zip(seeds, coeffs):
                self._set_seed(seed)
                acc.add_(self._noise(name, p, seed).to(torch.float32).mul_(c))
            p.data.add_(acc.div_(population_size).mul_(alpha).to(p.dtype))
            del acc
        self._sync()
        gc.collect()
        return True

    def dump_noise_for_seed(self, seed: int, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        noise_state = {name: self._noise(name, p, seed).cpu() for name, p in self.model_runner.model.named_parameters()}
        torch.save(noise_state, os.path.join(out_dir, f"noise_seed_{int(seed)}.pt"))
        self._sync()
        return True

    def apply_perturbation(self, seed, sigma):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        self._set_seed(seed)
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name])
            if self._should_perturb(name):
                p.data.add_(float(sigma) * self._noise(name, p, seed))
        self._sync()
        return True

    def apply_averaged_perturbations(self, seeds_sigmas, weights=None):
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        K = len(seeds_sigmas)
        weights = [1.0 / K] * K if weights is None else [w / sum(weights) for w in weights]
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name])
            if self._should_perturb(name):
                acc = torch.zeros_like(p.data, dtype=torch.float32)
                for (seed, sigma), w in zip(seeds_sigmas, weights):
                    acc.add_(self._noise(name, p, seed).to(torch.float32).mul_(w * float(sigma)))
                p.data.add_(acc.to(p.dtype))
                del acc
        self._sync()
        gc.collect()
        return True


class IIDNcclLLM(RandOptNcclLLM):
    """RandOptNcclLLM whose vLLM workers use IIDWorkerExtension."""

    def __init__(self, *args, **kwargs):
        kwargs["worker_extension_cls"] = "analysis.iid_noise.iid_worker_extn.IIDWorkerExtension"
        super().__init__(*args, **kwargs)
