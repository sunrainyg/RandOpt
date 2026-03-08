import gc
import time
import random
import numpy as np
import torch
import os
import inspect
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
    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, SIGMA, negate=False):
        """Undo perturbation. Must use same negate value as perturb_self_weights."""
        self._set_seed(seed)
        sign = -1.0 if negate else 1.0  # Same sign as perturb
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            # Undo: subtract what we added (sign * sigma * noise)
            p.data.add_(-sign * float(SIGMA) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """
        Mimics the Original implementation's update loop structure:
        Iterate Param -> Iterate Seeds -> Accumulate -> Single Update.
        """
        # seeds and coeffs should be lists of equal length
        # coeffs[i] should be: (alpha / population_size) * normalized_reward
        
        for _, p in self.model_runner.model.named_parameters():
            # Use model's native dtype for accumulator to save memory
            # Scale coefficients to avoid precision issues
            update_accumulator = torch.zeros_like(p.data)
            
            for i, seed in enumerate(seeds):
                self._set_seed(seed)
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                
                # Generate noise in native precision
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                
                # Scale and accumulate in-place (memory efficient)
                # Use float32 coefficient to maintain precision
                update_accumulator.add_(noise, alpha=float(coeffs[i]))
                
                # Immediately free noise tensor
                del noise
            
            # div by population_size multiply by alpha (scalar)
            # Apply update in-place
            update_accumulator.mul_(alpha / population_size)
            p.data.add_(update_accumulator)
            
            del update_accumulator
            
            # Periodically clear cache to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
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
    
    def apply_perturbation(self, seed, sigma):
        """Apply perturbation from base weights (not current weights)."""
        if not hasattr(self, '_base_weights'):
            raise RuntimeError("Must call store_base_weights first")
        
        self._set_seed(seed)
        for name, p in self.model_runner.model.named_parameters():
            # Restore base weights first
            p.data.copy_(self._base_weights[name])
            # Then apply perturbation
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True
    
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
        
        for name, p in self.model_runner.model.named_parameters():
            # Start with base weights
            p.data.copy_(self._base_weights[name])
            
            # Accumulate weighted perturbations in float32 for precision
            perturbation = torch.zeros_like(p.data, dtype=torch.float32)
            
            for (seed, sigma), weight in zip(seeds_sigmas, weights):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                perturbation.add_(weight * float(sigma) * noise.to(torch.float32))
                del noise
            
            # Apply averaged perturbation
            p.data.add_(perturbation.to(p.dtype))
            del perturbation
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
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
