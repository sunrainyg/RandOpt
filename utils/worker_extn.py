import gc
import time
import random
import numpy as np
import torch
import os
import inspect

try:
    from utils.perturbation_norms import (
        SUPPORTED_PERTURBATION_METHODS,
        build_modular_scales,
        load_mass_config,
        natural_norm,
        normalization_denominator,
    )
except ImportError:
    # Fallback when worker_extn.py is loaded with utils/ directly on sys.path.
    from perturbation_norms import (
        SUPPORTED_PERTURBATION_METHODS,
        build_modular_scales,
        load_mass_config,
        natural_norm,
        normalization_denominator,
    )

try:
    from vllm.forward_context import set_forward_context
except ImportError:
    set_forward_context = None

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """
    Methods used by the ES trainer:
    - perturb_self_weights(seed, sigma_or_scale, coeff=1.0, negate=False)
    - restore_self_weights(seed, SIGMA)
    - update_weights_from_seeds(seeds, coeffs)  <-- NEW METHOD
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)
    
    Ensemble methods:
    - store_base_weights()
    - apply_perturbation(seed, sigma)
    - reset_to_base_weights()
    - get_next_token_logits(input_ids)
    """
    
    def cleanup_gpu_memory(self):
        """Explicitly clean up GPU memory. Call this between iterations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
    
    # Prefixes of visual encoder parameters to skip during perturbation (for VL models)
    _VISUAL_PREFIXES = ("visual.", "model.visual.")

    def _should_perturb(self, name: str) -> bool:
        """Check if a parameter should be perturbed.
        
        By default, skips visual encoder params for VL models.
        Set env PERTURB_VISUAL=1 to also perturb visual encoder.
        """
        if os.environ.get("PERTURB_VISUAL", "0") == "1":
            return True  # Perturb ALL parameters including visual encoder
        return not name.startswith(self._VISUAL_PREFIXES)

    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _named_perturbable_parameters(self):
        return [
            (name, param)
            for name, param in self.model_runner.model.named_parameters()
            if self._should_perturb(name)
        ]

    @staticmethod
    def _sample_parameter_noise(param, seed):
        """Match the original RandOpt noise stream exactly."""
        generator = torch.Generator(device=param.device)
        generator.manual_seed(int(seed))
        return torch.randn(
            param.shape,
            dtype=param.dtype,
            device=param.device,
            generator=generator,
        )

    def _get_modular_scales(self, named_parameters, mass_config):
        config = load_mass_config(mass_config)
        cache_key = tuple(sorted(config.items()))
        cache = getattr(self, "_modular_scale_cache", {})
        if cache_key not in cache:
            cache[cache_key] = build_modular_scales(
                named_parameters,
                mass_config=config,
            )
            self._modular_scale_cache = cache
        return cache[cache_key]

    def _compute_modular_radial_denominator(
        self,
        named_parameters,
        seed,
        modular_scales,
        power_iterations,
    ):
        global_denominator = 0.0
        for name, param in named_parameters:
            modular_scale = modular_scales[name]
            if modular_scale == float("inf"):
                continue
            noise = self._sample_parameter_noise(param, seed)
            local_denominator = modular_scale * natural_norm(
                name,
                noise,
                power_iterations=power_iterations,
            )
            global_denominator = max(global_denominator, local_denominator)
            del noise

        if global_denominator <= 0.0:
            raise RuntimeError(
                "Could not compute a positive modular-radial denominator. "
                "Check mass_config and perturbable parameters."
            )
        return global_denominator

    def _apply_weight_perturbation(
        self,
        seed,
        radius,
        negate=False,
        method="isotropic",
        mass_config=None,
        power_iterations=8,
        restore=False,
    ):
        self._set_seed(seed)

        method = str(method)
        if method not in SUPPORTED_PERTURBATION_METHODS:
            raise ValueError(
                f"Unknown perturbation method '{method}'. "
                f"Choose from {SUPPORTED_PERTURBATION_METHODS}."
            )

        radius = float(radius)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius must be finite and non-negative")
        if int(power_iterations) < 1:
            raise ValueError("power_iterations must be at least 1")

        named_parameters = self._named_perturbable_parameters()
        modular_scales = None
        global_radial_denominator = None

        if method in {"modular_shell", "modular_radial"}:
            modular_scales = self._get_modular_scales(
                named_parameters,
                mass_config,
            )

        if method == "modular_radial":
            global_radial_denominator = self._compute_modular_radial_denominator(
                named_parameters=named_parameters,
                seed=seed,
                modular_scales=modular_scales,
                power_iterations=int(power_iterations),
            )

        perturb_sign = -1.0 if negate else 1.0
        if restore:
            perturb_sign *= -1.0

        with torch.no_grad():
            for name, param in named_parameters:
                noise = self._sample_parameter_noise(param, seed)
                modular_scale = (
                    modular_scales[name] if modular_scales is not None else 1.0
                )
                denominator = normalization_denominator(
                    method=method,
                    name=name,
                    noise=noise,
                    modular_scale=modular_scale,
                    global_radial_denominator=global_radial_denominator,
                    power_iterations=int(power_iterations),
                )

                # Zero-mass modular groups have denominator = infinity and are
                # intentionally frozen for modular perturbations.
                if denominator != float("inf"):
                    alpha = perturb_sign * radius / denominator
                    param.data.add_(noise, alpha=float(alpha))
                del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def perturb_self_weights(
        self,
        seed,
        radius,
        negate=False,
        method="isotropic",
        mass_config=None,
        power_iterations=8,
    ):
        """Apply one of the supported random perturbation methods."""
        return self._apply_weight_perturbation(
            seed=seed,
            radius=radius,
            negate=negate,
            method=method,
            mass_config=mass_config,
            power_iterations=power_iterations,
            restore=False,
        )

    def restore_self_weights(
        self,
        seed,
        radius,
        negate=False,
        method="isotropic",
        mass_config=None,
        power_iterations=8,
    ):
        """Undo a perturbation using the identical seed and normalization."""
        return self._apply_weight_perturbation(
            seed=seed,
            radius=radius,
            negate=negate,
            method=method,
            mass_config=mass_config,
            power_iterations=power_iterations,
            restore=True,
        )

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """
        Mimics the Original implementation's update loop structure:
        Iterate Param -> Iterate Seeds -> Accumulate -> Single Update.
        """
        # seeds and coeffs should be lists of equal length
        # coeffs[i] should be: (alpha / population_size) * normalized_reward
        
        num_seeds = len(seeds)
        param_count = 0
        
        for name, p in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                param_count += 1
                continue
            # float32
            update_accumulator = torch.zeros_like(p.data, dtype=torch.float32)
            
            for i, seed in enumerate(seeds):
                self._set_seed(seed)
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                
                # Generate noise (in native precision, usually float16/bfloat16)
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                
                # FIXED: Convert noise to float32 BEFORE multiplication.
                # Use in-place operation to avoid extra memory allocation
                noise_fp32 = noise.to(torch.float32)
                del noise  # Free original noise immediately
                
                # Scale in-place and accumulate
                noise_fp32.mul_(coeffs[i])
                update_accumulator.add_(noise_fp32)
                
                # Clean up immediately to avoid memory accumulation
                del noise_fp32
            
            # div by population_size multiply by alpha (scalar)
            update_accumulator.div_(population_size)
            update_accumulator.mul_(alpha)
            # Apply final update to weight (cast back to model dtype at the very end)
            p.data.add_(update_accumulator.to(p.dtype))
            
            del update_accumulator
            param_count += 1
            
            # Periodic cache clearing for large models (every 50 parameters)
            if param_count % 50 == 0:
                torch.cuda.empty_cache()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        return True

    def get_worker_ip(self):
        """Return the IP address of this worker's node."""
        from vllm.utils import get_ip
        return get_ip()

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
    
    def dump_noise_for_seed(self, seed: int, out_dir: str):
        """
        Generate per-parameter noise using the same method as perturb/restore
        and save them to disk for determinism comparison.
        """
        os.makedirs(out_dir, exist_ok=True)
        noise_state = {}
        for name, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            noise_state[name] = noise.detach().cpu()
            del noise
        torch.save(noise_state, os.path.join(out_dir, f"noise_seed_{int(seed)}.pt"))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
    
    # debug
    def print_model_weights_stats(self):
        for name, p in self.model_runner.model.named_parameters():
            print(f"Param: {name}, Shape: {p.shape}")
        return True
    
    # ==================== Ensemble Methods ====================
    
    def store_base_weights(self):
        """Store a copy of current weights as base weights for ensemble."""
        self._base_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self._base_weights[name] = p.data.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    
    def apply_perturbation(
        self,
        seed,
        radius,
        method="isotropic",
        mass_config=None,
        power_iterations=8,
    ):
        """Apply a perturbation from stored base weights."""
        if not hasattr(self, "_base_weights"):
            raise RuntimeError("Must call store_base_weights first")
        for name, param in self.model_runner.model.named_parameters():
            param.data.copy_(self._base_weights[name])
        return self.perturb_self_weights(
            seed=seed,
            radius=radius,
            negate=False,
            method=method,
            mass_config=mass_config,
            power_iterations=power_iterations,
        )

    def reset_to_base_weights(self):
        """Reset model weights to stored base weights."""
        if not hasattr(self, '_base_weights'):
            raise RuntimeError("Must call store_base_weights first")
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    
    def clear_base_weights(self):
        """Free memory used by stored base weights."""
        if hasattr(self, '_base_weights'):
            del self._base_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    def apply_averaged_perturbations(self, seeds_sigmas, weights=None):
        """
        Apply the weighted average of multiple perturbations from base weights.
        This creates a single weight-averaged model from K perturbed models.
        
        Args:
            seeds_sigmas: List of (seed, sigma) tuples
            weights: Optional list of weights for each perturbation (default: equal weights)
        
        The averaged model is: W_base + sum(w_i * sigma_i * noise_i) / sum(w_i)
        """
        if not hasattr(self, '_base_weights'):
            raise RuntimeError("Must call store_base_weights first")
        
        K = len(seeds_sigmas)
        if weights is None:
            weights = [1.0 / K] * K  # Equal weights, normalized
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        param_count = 0
        for name, p in self.model_runner.model.named_parameters():
            # Start with base weights
            p.data.copy_(self._base_weights[name])
            
            if self._should_perturb(name):
                # Accumulate weighted perturbations in float32 for precision
                perturbation = torch.zeros_like(p.data, dtype=torch.float32)
                
                for (seed, sigma), weight in zip(seeds_sigmas, weights):
                    gen = torch.Generator(device=p.device)
                    gen.manual_seed(int(seed))
                    noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                    
                    # Convert to float32 and scale in-place
                    noise_fp32 = noise.to(torch.float32)
                    del noise  # Free original noise immediately
                    
                    noise_fp32.mul_(weight * float(sigma))
                    perturbation.add_(noise_fp32)
                    del noise_fp32  # Clean up immediately
                
                # Apply averaged perturbation
                p.data.add_(perturbation.to(p.dtype))
                del perturbation
            
            param_count += 1
            
            # Periodic cache clearing for large models
            if param_count % 50 == 0:
                torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        return True
    
    def get_logits_for_prompt(self, input_ids_list):
        """
        Get logits for the last token position for a batch of prompts.
        Returns logits as CPU tensors for ensemble averaging.
        
        Args:
            input_ids_list: List of input_ids (each is a list of token ids)
        
        Returns:
            List of logits tensors (vocab_size,) for each prompt
        """
        model = self.model_runner.model
        model.eval()
        
        results = []
        with torch.no_grad():
            for input_ids in input_ids_list:
                seq_len = len(input_ids)
                # vLLM V1 expects flattened tensors (not batched)
                # input_ids: (seq_len,), positions: (seq_len,)
                ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                positions = torch.arange(seq_len, dtype=torch.long, device=self.device)
                
                # Forward pass - get logits
                # vLLM v0.11+ requires forward context
                if set_forward_context is not None and hasattr(self.model_runner, "vllm_config"):
                    with set_forward_context(attn_metadata=None, 
                                           vllm_config=self.model_runner.vllm_config):
                        outputs = model(input_ids=ids_tensor, positions=positions)
                else:
                    # Fallback for older vLLM versions
                    if 'positions' in inspect.signature(model.forward).parameters:
                        outputs = model(input_ids=ids_tensor, positions=positions)
                    else:
                        outputs = model(input_ids=ids_tensor.unsqueeze(0))
                
                # Get logits for the last position
                # outputs may have .logits attribute or be the logits tensor directly
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                # vLLM V1: logits shape is (seq_len, vocab_size) for flattened input
                # or (batch, seq_len, vocab_size) for batched input
                if logits.ndim == 2:
                    # Flattened: (seq_len, vocab_size)
                    last_logits = logits[-1, :].cpu()
                else:
                    # Batched: (batch, seq_len, vocab_size)
                    last_logits = logits[0, -1, :].cpu()
                results.append(last_logits)
                
                del ids_tensor, positions, outputs, logits
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        return results
    
    def generate_with_logits_callback(self, input_ids, max_new_tokens, temperature=1.0):
        """
        Generate tokens step by step and return the logits at each step.
        This is for debugging/analysis - actual ensemble should use get_logits_for_prompt.
        
        Returns: (generated_ids, list_of_logits_at_each_step)
        """
        model = self.model_runner.model
        model.eval()
        
        current_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        all_logits = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids=current_ids)
                last_logits = outputs.logits[0, -1, :]
                all_logits.append(last_logits.cpu())
                
                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = last_logits.argmax(dim=-1, keepdim=True)
                
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
                
                del outputs
        
        generated = current_ids[0].cpu().tolist()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        return generated, all_logits
