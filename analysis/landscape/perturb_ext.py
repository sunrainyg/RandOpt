from __future__ import annotations

import math

import torch

CHUNK = 64 * 1024 * 1024


class PerturbExt:
    def _named(self):
        if not hasattr(self, "_pe_named"):
            self._pe_named = sorted(self.model_runner.model.named_parameters(), key=lambda kv: kv[0])
        return self._pe_named

    def snapshot(self, proj_coords: int = 20_000_000, proj_seed: int = 20260305) -> int:
        params = [p for _, p in self._named()]
        self._pe_base = [p.detach().to("cpu", copy=True) for p in params]
        self._pe_pc = int(min(proj_coords, sum(p.numel() for p in params)))
        g = torch.Generator(device="cpu").manual_seed(proj_seed)
        self._pe_proj = torch.randn(2, self._pe_pc, generator=g, dtype=torch.float32).to(params[0].device)
        return self._pe_pc

    @torch.no_grad()
    def perturb(self, seed: int, sigma: float) -> list[float]:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        proj, pc, consumed = self._pe_proj, self._pe_pc, 0
        coords = torch.zeros(2, dtype=torch.float64, device=proj.device)
        for i, (_, p) in enumerate(self._named()):
            sharded = hasattr(p, "output_dim") or hasattr(p, "input_dim")
            g = torch.Generator(device=p.device).manual_seed((seed * 1_000_003 + i) * 64 + (rank if sharded else 0))
            flat = p.data.view(-1)
            for a in range(0, flat.numel(), CHUNK):
                b = min(a + CHUNK, flat.numel())
                noise = torch.randn(b - a, generator=g, device=p.device, dtype=torch.float32)
                if consumed < pc:
                    take = min(b - a, pc - consumed)
                    coords += (proj[:, consumed:consumed + take] @ noise[:take]).double()
                    consumed += take
                flat[a:b] = flat[a:b].float().add_(noise, alpha=sigma).to(flat.dtype)
        torch.cuda.synchronize()
        return (coords / math.sqrt(pc)).tolist()

    @torch.no_grad()
    def restore(self) -> bool:
        for (_, p), b in zip(self._named(), self._pe_base):
            p.data.copy_(b, non_blocking=True)
        torch.cuda.synchronize()
        return True
