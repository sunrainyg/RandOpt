# Text-to-Image experiment script

import os, re, random
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from diffusers import StableDiffusionXLPipeline
from transformers import CLIPProcessor, CLIPModel


# ----------------------------
# Helpers
# ----------------------------
def sanitize_filename(s: str, max_len: int = 80) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "text"

def make_prompt_plus_style_texts(prompts: List[str], style_text: str) -> List[str]:
    style_text = (style_text or "").strip()
    if not style_text:
        return [p for p in prompts]
    return [f"{style_text}" for p in prompts]


def collect_named_params(module: torch.nn.Module, prefix: str) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(f"{prefix}.{n}", p) for n, p in module.named_parameters()]


def build_sigma_map(named_params: List[Tuple[str, torch.nn.Parameter]], sigma: float) -> Dict[str, float]:
    return {name: float(sigma) for name, _ in named_params}


@torch.no_grad()
def add_gaussian_noise_inplace(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    sigmas: Dict[str, float],
    g: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    applied: Dict[str, torch.Tensor] = {}
    for name, p in named_params:
        sigma = sigmas.get(name, 0.0)
        if sigma == 0.0:
            continue
        noise = torch.randn(p.shape, generator=g, device=device, dtype=torch.float32).to(dtype)
        p.add_(noise, alpha=sigma)
        applied[name] = noise
    return applied


@torch.no_grad()
def remove_noise_inplace(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    applied: Dict[str, torch.Tensor],
    sigmas: Dict[str, float],
):
    for name, p in named_params:
        sigma = sigmas.get(name, 0.0)
        if sigma == 0.0:
            continue
        n = applied.get(name, None)
        if n is None:
            continue
        p.add_(n, alpha=-sigma)


def _chunks(n: int, bs: int):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)


# ----------------------------
# Grid making + headings
# ----------------------------
def make_table_grid(images_2d: List[List[Image.Image]], pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    rows = len(images_2d)
    cols = len(images_2d[0]) if rows else 0
    assert rows > 0 and cols > 0
    w, h = images_2d[0][0].size
    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for r in range(rows):
        for c in range(cols):
            im = images_2d[r][c]
            assert im.size == (w, h)
            x = pad + c * (w + pad)
            y = pad + r * (h + pad)
            canvas.paste(im, (x, y))
    return canvas


def _try_load_font(size: int = 16) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_column_headings(
    grid: Image.Image,
    headings: List[str],
    cell_w: int,
    pad: int = 8,
    header_h: int = 32,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
) -> Image.Image:
    W, H = grid.size
    out = Image.new("RGB", (W, H + header_h + pad), bg)
    out.paste(grid, (0, header_h + pad))

    draw = ImageDraw.Draw(out)
    font = _try_load_font(size=32)

    for c, title in enumerate(headings):
        x0 = pad + c * (cell_w + pad)
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + (cell_w - tw) // 2
        ty = max(0, (header_h - th) // 2)
        draw.text((tx, ty), title, fill=fg, font=font)

    draw.line([(0, header_h + pad // 2), (W, header_h + pad // 2)], fill=(200, 200, 200), width=1)
    return out


# ----------------------------
# Mean ensemble on denoising trajectory (SDXL)
# ----------------------------
@torch.no_grad()
def sample_noise_tensors(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    sigmas: Dict[str, float],
    g: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, p in named_params:
        sigma = sigmas.get(name, 0.0)
        if sigma == 0.0:
            continue
        n = torch.randn(p.shape, generator=g, device=device, dtype=torch.float32).to(dtype)
        out[name] = n
    return out


@torch.no_grad()
def apply_noise_inplace(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    noise: Dict[str, torch.Tensor],
    sigmas: Dict[str, float],
    sign: float,
):
    for name, p in named_params:
        sigma = sigmas.get(name, 0.0)
        if sigma == 0.0:
            continue
        n = noise.get(name, None)
        if n is None:
            continue
        p.add_(n, alpha=sign * sigma)

def add_gaussian_noise_inplace_onepass(named_params, sigmas, g, device, dtype):
    applied = {}
    for name, p in named_params:
        sigma = sigmas.get(name, 0.0)
        if sigma == 0.0:
            continue
        # generate directly in dtype (no fp32->fp16 transient)
        n = torch.randn(p.shape, generator=g, device=device, dtype=dtype)
        p.add_(n, alpha=sigma)
        applied[name] = n
    return applied

@torch.no_grad()
def sdxl_denoise_mean_topk(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    topk_model_seeds: List[int],
    sigma: float,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    height: int = 1024,
    width: int = 1024,
    diffusion_seed: int = 0,
    perturb_what: str = "unet",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    SDXL denoising with mean ensembling over K parameter perturbations.

    FIX: Avoid OOM by NOT precomputing/storing K full UNet-sized noise dicts (noises_k).
         Instead, for each k we:
           (seed -> generate noise -> apply -> forward -> remove)
         Also ensure noise is generated directly in parameter dtype (no fp32->fp16 transient).
    """
    device = pipe._execution_device if hasattr(pipe, "_execution_device") else next(pipe.unet.parameters()).device
    pipe.unet.eval()

    # ----------------------------
    # Collect parameters to perturb
    # ----------------------------
    named_params: List[Tuple[str, torch.nn.Parameter]] = []
    named_params += collect_named_params(pipe.unet, "unet")
    if perturb_what == "all":
        if getattr(pipe, "text_encoder", None) is not None:
            named_params += collect_named_params(pipe.text_encoder, "text_encoder")
        if getattr(pipe, "text_encoder_2", None) is not None:
            named_params += collect_named_params(pipe.text_encoder_2, "text_encoder_2")

    sigmas = build_sigma_map(named_params, sigma)

    # Use param dtype for noise (avoid fp32 peak)
    p_dtype = next(pipe.unet.parameters()).dtype

    # Model-seed generator
    g_model = torch.Generator(device=device)

    # ----------------------------
    # Prompt encodings (SDXL)
    # ----------------------------
    do_cfg = guidance_scale is not None and guidance_scale > 1.0
    neg = negative_prompt if negative_prompt is not None else ""

    prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        negative_prompt=neg,
        negative_prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
    )

    if do_cfg:
        if neg_prompt_embeds is None:
            neg_prompt_embeds = torch.zeros_like(prompt_embeds)
        prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

    text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])

    def _get_time_ids_pair():
        try:
            res = pipe._get_add_time_ids(
                original_size=(height, width),
                crops_coords_top_left=(0, 0),
                target_size=(height, width),
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        except TypeError:
            res = pipe._get_add_time_ids(
                original_size=(height, width),
                crops_coords_top_left=(0, 0),
                target_size=(height, width),
                dtype=prompt_embeds.dtype,
            )

        if isinstance(res, (tuple, list)):
            if len(res) == 2:
                add_time_ids, add_neg_time_ids = res
            else:
                add_time_ids = res[0] if len(res) >= 1 else res
                add_neg_time_ids = add_time_ids
        else:
            add_time_ids = res
            add_neg_time_ids = add_time_ids
        return add_time_ids, add_neg_time_ids

    add_time_ids, add_neg_time_ids = _get_time_ids_pair()
    add_time_ids = add_time_ids.to(device)
    add_neg_time_ids = add_neg_time_ids.to(device)

    if do_cfg:
        add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

    if do_cfg:
        if neg_pooled_prompt_embeds is None:
            neg_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        text_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        text_embeds = pooled_prompt_embeds

    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": add_time_ids}

    # ----------------------------
    # Timesteps + initial latents
    # ----------------------------
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    g_latent = torch.Generator(device=device)
    g_latent.manual_seed(int(diffusion_seed))
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=g_latent,
    )

    # ----------------------------
    # Denoising loop with mean ensemble across K perturbations
    # ----------------------------
    K = len(topk_model_seeds)
    assert K >= 1

    for t in timesteps:
        latent_model_input = latents
        if do_cfg:
            latent_model_input = torch.cat([latents, latents], dim=0)

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        acc = None

        for k_seed in topk_model_seeds:
            # Deterministic perturbation per k
            g_model.manual_seed(int(k_seed))

            # Apply parameter noise (generated directly in param dtype to reduce peak mem)
            applied = add_gaussian_noise_inplace(
                named_params=named_params,
                sigmas=sigmas,
                g=g_model,
                device=device,
                dtype=p_dtype,   # IMPORTANT: fp16/bf16 noise directly, no fp32->fp16 transient
            )

            # UNet forward
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # Remove parameter noise
            remove_noise_inplace(named_params, applied, sigmas)

            # Classifier-free guidance after UNet
            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            acc = noise_pred if acc is None else (acc + noise_pred)

        pred_mean = acc / float(K)
        latents = pipe.scheduler.step(pred_mean, t, latents, return_dict=False)[0]

    return latents


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    orig_dtype = pipe.vae.dtype
    pipe.vae.to(dtype=torch.float32)
    image = pipe.vae.decode(latents.to(dtype=torch.float32), return_dict=False)[0]
    pipe.vae.to(dtype=orig_dtype)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return pipe.numpy_to_pil(image)


# ----------------------------
# CLIP scoring
# ----------------------------
class CLIPScorer:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def score_images_to_text_per_image(self, images: List[Image.Image], texts: List[str], batch_size: int = 8) -> torch.Tensor:
        assert len(images) == len(texts)
        scores = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            batch_txts = texts[i:i + batch_size]
            inputs = self.processor(text=batch_txts, images=batch_imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            out = self.model(**inputs)
            img_emb = F.normalize(out.image_embeds, dim=-1)
            txt_emb = F.normalize(out.text_embeds, dim=-1)
            sim = (img_emb * txt_emb).sum(dim=-1)
            scores.append(sim.detach().float().cpu())
        return torch.cat(scores, dim=0)


# ----------------------------
# Batched SDXL generation with fixed per-prompt diffusion seeds
# ----------------------------
@torch.no_grad()
def generate_prompts_batched_fixed_noise(
    pipe: StableDiffusionXLPipeline,
    prompts: List[str],
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    diffusion_seeds: List[int],
    device: torch.device,
    batch_size_prompts: int = 4,
) -> List[Image.Image]:
    assert len(prompts) == len(diffusion_seeds)
    out_images: List[Optional[Image.Image]] = [None] * len(prompts)

    for s, e in _chunks(len(prompts), batch_size_prompts):
        p_chunk = prompts[s:e]
        seeds_chunk = diffusion_seeds[s:e]

        gens = []
        for sd in seeds_chunk:
            g = torch.Generator(device=device)
            g.manual_seed(int(sd))
            gens.append(g)

        out = pipe(
            prompt=p_chunk,
            negative_prompt=[negative_prompt] * len(p_chunk) if negative_prompt else None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=gens,
        )
        imgs = out.images
        for j, im in enumerate(imgs):
            out_images[s + j] = im

    return [im for im in out_images]


# ----------------------------
# Generate & cache images for a set of perturb indices (plus optional base)
# ----------------------------
@torch.no_grad()
def generate_images_for_indices_cached(
    *,
    pipe: StableDiffusionXLPipeline,
    named_params: List[Tuple[str, torch.nn.Parameter]],
    sigmas: Dict[str, float],
    g_model: torch.Generator,
    prompts: List[str],
    diffusion_seeds: List[int],
    indices: List[Optional[int]],  # None means BASE (no perturb)
    seed: int,
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    device: torch.device,
    prompt_batch_size: int,
) -> Dict[Optional[int], List[Image.Image]]:
    cache: Dict[Optional[int], List[Image.Image]] = {}
    for idx in indices:
        if idx is None:
            imgs = generate_prompts_batched_fixed_noise(
                pipe=pipe,
                prompts=prompts,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                diffusion_seeds=diffusion_seeds,
                device=device,
                batch_size_prompts=prompt_batch_size,
            )
            cache[idx] = imgs
            continue

        model_seed = 10_000_000 + seed + int(idx)
        g_model.manual_seed(int(model_seed))
        applied = add_gaussian_noise_inplace(
            named_params=named_params,
            sigmas=sigmas,
            g=g_model,
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )

        imgs = generate_prompts_batched_fixed_noise(
            pipe=pipe,
            prompts=prompts,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            diffusion_seeds=diffusion_seeds,
            device=device,
            batch_size_prompts=prompt_batch_size,
        )
        remove_noise_inplace(named_params, applied, sigmas)
        cache[idx] = imgs

    return cache


# ----------------------------
# Compute TOP-K ensemble images ONCE per prompt set
# ----------------------------
@torch.no_grad()
def compute_topk_ensemble_images_once(
    *,
    pipe: StableDiffusionXLPipeline,
    prompts: List[str],
    diffusion_seeds: List[int],
    negative_prompt: str,
    topk_idx: List[int],
    seed: int,
    sigma: float,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    perturb: str,
) -> List[Image.Image]:
    ens_images: List[Image.Image] = []
    topk_model_seeds = [10_000_000 + seed + int(j) for j in topk_idx]
    for p, dseed in zip(prompts, diffusion_seeds):
        lat = sdxl_denoise_mean_topk(
            pipe=pipe,
            prompt=p,
            negative_prompt=negative_prompt,
            topk_model_seeds=topk_model_seeds,
            sigma=sigma,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            diffusion_seed=int(dseed),
            perturb_what=perturb,
            dtype=next(pipe.unet.parameters()).dtype,
        )
        ens_images.append(decode_latents_to_pil(pipe, lat)[0])
    return ens_images


# ----------------------------
# Visualization using cached images (BASE + TOP1 + TOPK-ENS + RANDOM)
# ----------------------------
@torch.no_grad()
def visualize_prompt_set_from_cache(
    *,
    prompts: List[str],
    top1_idx: int,
    topk_idx: List[int],
    rand_idx: List[int],
    base_images: List[Image.Image],
    single_cache: Dict[int, List[Image.Image]],  # must contain top1_idx and all rand_idx
    ensemble_images: List[Image.Image],          # aligned with prompts
    title_prefix: str = "SET",
) -> Dict[str, Any]:
    assert len(base_images) == len(prompts)
    assert len(ensemble_images) == len(prompts)
    for j in [top1_idx] + rand_idx:
        assert j in single_cache, f"cache missing perturb idx {j}"
        assert len(single_cache[j]) == len(prompts)

    headings: List[str] = []
    headings.append("BASE")
    headings.append(f"TOP-1 (idx {top1_idx})")
    headings.append(f"TOP-K mean (K={len(topk_idx)})")
    for k, j in enumerate(rand_idx):
        headings.append(f"Random #{k+1} (idx {j})")

    images_2d: List[List[Image.Image]] = []
    for pi in range(len(prompts)):
        row: List[Image.Image] = []
        row.append(base_images[pi])
        row.append(single_cache[top1_idx][pi])
        row.append(ensemble_images[pi])
        for j in rand_idx:
            row.append(single_cache[j][pi])
        images_2d.append(row)

    grid = make_table_grid(images_2d, pad=8)
    cell_w, _ = images_2d[0][0].size
    grid = add_column_headings(grid, headings=headings, cell_w=cell_w, pad=8, header_h=34)

    return {"grid": grid, "headings": headings, "rand_idx": rand_idx}


# ----------------------------
# TEST EVAL: CLIP (base vs TOP1 vs TOPK-ENS) using cached images
# ----------------------------
@torch.no_grad()
def eval_test_clip_base_top1_topk_cached(
    *,
    clip: CLIPScorer,
    test_prompts: List[str],
    target_text: str,
    base_images: List[Image.Image],
    top1_images: List[Image.Image],
    topk_ens_images: List[Image.Image],
    clip_batch: int,
):
    if len(test_prompts) == 0:
        print("[test-eval] no test prompts; skipping")
        return None
    assert len(base_images) == len(test_prompts)
    assert len(top1_images) == len(test_prompts)
    assert len(topk_ens_images) == len(test_prompts)

    texts = make_prompt_plus_style_texts(test_prompts, target_text)

    base_scores = clip.score_images_to_text_per_image(base_images, texts, batch_size=clip_batch)
    top1_scores = clip.score_images_to_text_per_image(top1_images, texts, batch_size=clip_batch)
    topk_scores = clip.score_images_to_text_per_image(topk_ens_images, texts, batch_size=clip_batch)

    base_avg = float(base_scores.mean())
    top1_avg = float(top1_scores.mean())
    topk_avg = float(topk_scores.mean())

    def _rel(a, b):
        # (b-a)/|a|
        return ((b - a) / (abs(a) + 1e-12)) * 100.0

    print("\n[TEST CLIP eval] (cached)")
    print(f"  base_avg_clip  : {base_avg:.6f}")
    print(f"  top1_avg_clip  : {top1_avg:.6f}   (Δ={top1_avg-base_avg:+.6f}, { _rel(base_avg, top1_avg):+.2f}%)")
    print(f"  topk_avg_clip  : {topk_avg:.6f}   (Δ={topk_avg-base_avg:+.6f}, { _rel(base_avg, topk_avg):+.2f}%)")
    print(f"  topk - top1    : {topk_avg-top1_avg:+.6f}   ({ _rel(top1_avg, topk_avg):+.2f}% vs top1)")

    print("\n  Per-prompt CLIP:")
    for p, b, t1, tk in zip(test_prompts, base_scores.tolist(), top1_scores.tolist(), topk_scores.tolist()):
        print(f"   - {p[:70]:70s}  base {b:.4f} | top1 {t1:.4f} ({t1-b:+.4f}) | topk {tk:.4f} ({tk-b:+.4f})")

    return {
        "base_avg": base_avg,
        "top1_avg": top1_avg,
        "topk_avg": topk_avg,
        "base_scores": base_scores,
        "top1_scores": top1_scores,
        "topk_scores": topk_scores,
        "abs_delta_top1_minus_base": top1_avg - base_avg,
        "abs_delta_topk_minus_base": topk_avg - base_avg,
        "abs_delta_topk_minus_top1": topk_avg - top1_avg,
        "rel_pct_top1_minus_base": _rel(base_avg, top1_avg),
        "rel_pct_topk_minus_base": _rel(base_avg, topk_avg),
        "rel_pct_topk_minus_top1": _rel(top1_avg, topk_avg),
    }


# ----------------------------
# Main: GLOBAL selection + train/test viz (cached)
# ----------------------------
@torch.no_grad()
def sdxl_thickets_global_select_train_and_test(
    train_prompts: List[str],
    target_text: str,
    test_prompts: Optional[List[str]] = None,
    N: int = 64,
    K: int = 8,
    sigma: float = 1e-3,
    steps: int = 30,
    guidance: float = 5.0,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    negative_prompt: str = "",
    perturb: str = "unet",
    outdir: str = "out_thickets_global_select",
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    dtype: torch.dtype = torch.float16,
    clip_model_id: str = "openai/clip-vit-large-patch14",
    clip_batch: int = 8,
    viz_random: int = 8,
    viz_seed: int = 123,
    prompt_batch_size: int = 4,
    show: bool = True,
    save: bool = True,
):
    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Please use a CUDA runtime for SDXL (Runtime -> Change runtime type -> GPU).")

    # SDXL
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(device)

    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Parameters to perturb
    named_params: List[Tuple[str, torch.nn.Parameter]] = []
    named_params += collect_named_params(pipe.unet, "unet")
    if perturb == "all":
        if getattr(pipe, "text_encoder", None) is not None:
            named_params += collect_named_params(pipe.text_encoder, "text_encoder")
        if getattr(pipe, "text_encoder_2", None) is not None:
            named_params += collect_named_params(pipe.text_encoder_2, "text_encoder_2")

    sigmas = build_sigma_map(named_params, sigma)
    g_model = torch.Generator(device=device)

    # Fixed diffusion seeds per prompt
    train_diff_seeds = [seed + pi * 1000 for pi in range(len(train_prompts))]
    if test_prompts is None:
        test_prompts = []
    test_diff_seeds = [seed + 50_000 + pi * 1000 for pi in range(len(test_prompts))]

    # CLIP
    clip = CLIPScorer(model_id=clip_model_id, device="cuda")

    # ----------------------------
    # TRAIN: Generate BASE once, then generate N perturbations once (cache) and score from cache
    # ----------------------------
    print("\n[TRAIN] Generating BASE images once...")
    train_base_images = generate_prompts_batched_fixed_noise(
        pipe=pipe,
        prompts=train_prompts,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        diffusion_seeds=train_diff_seeds,
        device=device,
        batch_size_prompts=prompt_batch_size,
    )

    print("\n[TRAIN] Generating images once for all N perturbations (and scoring via cache)...")
    train_cache: Dict[int, List[Image.Image]] = {}
    avg_scores = torch.empty((N,), dtype=torch.float32)

    target_texts_train = make_prompt_plus_style_texts(train_prompts, target_text)

    for i in range(N):
        model_seed = 10_000_000 + seed + i
        g_model.manual_seed(int(model_seed))

        applied = add_gaussian_noise_inplace(
            named_params=named_params,
            sigmas=sigmas,
            g=g_model,
            device=device,
            dtype=next(pipe.unet.parameters()).dtype,
        )

        images = generate_prompts_batched_fixed_noise(
            pipe=pipe,
            prompts=train_prompts,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            diffusion_seeds=train_diff_seeds,
            device=device,
            batch_size_prompts=prompt_batch_size,
        )
        remove_noise_inplace(named_params, applied, sigmas)

        train_cache[i] = images

        scores = clip.score_images_to_text_per_image(images, target_texts_train, batch_size=clip_batch)
        avg_scores[i] = scores.mean()

        if (i + 1) % max(1, N // 8) == 0:
            print(f"[train cached+scored] {i+1:>4d}/{N}  avg_clip={float(avg_scores[i]):.4f}")

    K_eff = min(K, N)
    topk = torch.topk(avg_scores, k=K_eff, largest=True)
    topk_idx = topk.indices.tolist()
    topk_scores = topk.values
    top1_idx = topk_idx[0]

    print("\n[GLOBAL TOP-K perturbations by avg CLIP over TRAIN prompts]")
    for rank, (idx, sc) in enumerate(zip(topk_idx, topk_scores.tolist()), start=1):
        print(f"  #{rank:02d}: perturb_idx={idx:>4d}  avg_clip={sc:.4f}")

    # ----------------------------
    # Random viz indices from same universe [0..N-1]
    # ----------------------------
    rng = random.Random(viz_seed)
    exclude = set(topk_idx) | {top1_idx}
    pool = [j for j in range(N) if j not in exclude]
    if len(pool) < viz_random:
        pool = [j for j in range(N) if j != top1_idx]  # allow topk if needed
    rand_idx = rng.sample(pool, k=min(viz_random, len(pool)))

    # ----------------------------
    # TRAIN: compute TOP-K ensemble once
    # ----------------------------
    print("\n[TRAIN] Computing TOP-K ensemble images once (for viz)...")
    train_ens_images = compute_topk_ensemble_images_once(
        pipe=pipe,
        prompts=train_prompts,
        diffusion_seeds=train_diff_seeds,
        negative_prompt=negative_prompt,
        topk_idx=topk_idx,
        seed=seed,
        sigma=sigma,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        perturb=perturb,
    )

    # ----------------------------
    # TRAIN visualization from cache (BASE + TOP1 + TOPK + random)
    # ----------------------------
    print("\n[TRAIN VIZ] Using cached images for BASE/TOP-1/random; cached ensemble for TOP-K.")
    train_single_needed = {top1_idx, *rand_idx}
    train_single_cache = {j: train_cache[j] for j in train_single_needed}

    train_viz = visualize_prompt_set_from_cache(
        prompts=train_prompts,
        top1_idx=top1_idx,
        topk_idx=topk_idx,
        rand_idx=rand_idx,
        base_images=train_base_images,
        single_cache=train_single_cache,
        ensemble_images=train_ens_images,
        title_prefix="TRAIN",
    )
    train_viz_grid = train_viz["grid"]

    # ----------------------------
    # TEST: generate base/top1/random once + ensemble once; eval includes topk
    # ----------------------------
    test_viz_grid = None
    test_eval = None
    test_rand_idx = None

    if len(test_prompts) > 0:
        test_rand_idx = rand_idx[:]  # keep consistent across train/test (feel free to resample)

        print("\n[TEST] Generating images once for BASE, TOP-1, and random single-perturb columns...")
        test_single_indices: List[Optional[int]] = [None, top1_idx] + list(test_rand_idx)
        test_cache_opt = generate_images_for_indices_cached(
            pipe=pipe,
            named_params=named_params,
            sigmas=sigmas,
            g_model=g_model,
            prompts=test_prompts,
            diffusion_seeds=test_diff_seeds,
            indices=test_single_indices,
            seed=seed,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            device=device,
            prompt_batch_size=prompt_batch_size,
        )

        test_base_images = test_cache_opt[None]
        test_top1_images = test_cache_opt[top1_idx]
        test_single_cache: Dict[int, List[Image.Image]] = {top1_idx: test_top1_images}
        for j in test_rand_idx:
            test_single_cache[j] = test_cache_opt[j]

        print("\n[TEST] Computing TOP-K ensemble images once (for viz + eval)...")
        test_ens_images = compute_topk_ensemble_images_once(
            pipe=pipe,
            prompts=test_prompts,
            diffusion_seeds=test_diff_seeds,
            negative_prompt=negative_prompt,
            topk_idx=topk_idx,
            seed=seed,
            sigma=sigma,
            steps=steps,
            guidance=guidance,
            width=width,
            height=height,
            perturb=perturb,
        )

        print("\n[TEST VIZ] Using cached images for BASE/TOP-1/random; cached ensemble for TOP-K.")
        test_viz = visualize_prompt_set_from_cache(
            prompts=test_prompts,
            top1_idx=top1_idx,
            topk_idx=topk_idx,
            rand_idx=test_rand_idx,
            base_images=test_base_images,
            single_cache=test_single_cache,
            ensemble_images=test_ens_images,
            title_prefix="TEST",
        )
        test_viz_grid = test_viz["grid"]

        # TEST eval: base vs top1 vs topk ensemble (cached)
        test_eval = eval_test_clip_base_top1_topk_cached(
            clip=clip,
            test_prompts=test_prompts,
            target_text=target_text,
            base_images=test_base_images,
            top1_images=test_top1_images,
            topk_ens_images=test_ens_images,
            clip_batch=clip_batch,
        )

    # ----------------------------
    # Save + show
    # ----------------------------
    if save:
        tag = sanitize_filename(target_text)
        train_path = os.path.join(outdir, f"TRAIN_VIZ_base_top1_topk_plus{viz_random}_target_{tag}.png")
        train_viz_grid.save(train_path)
        print(f"\n[saved] {train_path}")

        if test_viz_grid is not None:
            test_path = os.path.join(outdir, f"TEST_VIZ_base_top1_topk_plus{viz_random}_target_{tag}.png")
            test_viz_grid.save(test_path)
            print(f"[saved] {test_path}")

    if show:
        try:
            from IPython.display import display
            print("\n=== TRAIN GRID (rows=train prompts, cols=BASE | TOP1 | TOPK-ENS | random) ===")
            print("top1_idx:", top1_idx)
            print("topk_idx:", topk_idx)
            print("train rand_idx:", rand_idx)
            display(train_viz_grid)

            if test_viz_grid is not None:
                print("\n=== TEST GRID (rows=test prompts, cols=BASE | TOP1 | TOPK-ENS | random) ===")
                print("test rand_idx:", test_rand_idx)
                display(test_viz_grid)
        except Exception:
            pass

    return {
        "avg_scores": avg_scores,
        "topk_idx": topk_idx,
        "topk_scores": topk_scores,
        "top1_idx": top1_idx,
        "train_viz_grid": train_viz_grid,
        "test_viz_grid": test_viz_grid,
        "train_diffusion_seeds": train_diff_seeds,
        "test_diffusion_seeds": test_diff_seeds,
        "train_rand_idx": rand_idx,
        "test_rand_idx": test_rand_idx,
        "test_eval": test_eval,
        # caches (optional)
        "train_base_images": train_base_images,
        "train_cache": train_cache,  # i -> list[Image] over train prompts
    }


# ----------------------------
# Example usage
# ----------------------------
train_prompts = [
    "A corgi astronaut, full body, centered, clean background",
    "Kyoto street in the rain, pedestrians with umbrellas, reflections on wet pavement",
    "Boston skyline at sunset, Charles River in foreground, wide-angle view",
    "A close-up portrait of a tabby cat, bright yellow eyes",
    "A glass of iced coffee on a wooden table by a window, soft morning light",
    "A mountain biker on a forest trail, motion blur in the background, dynamic action shot",
    "A bowl of ramen with steam rising, chopsticks lifting noodles, cozy restaurant lighting, close-up",
    "A modern living room interior, mid-century chair, houseplants, natural light, wide-angle",
]

test_prompts = [
    "A dog on Mars, red rocky landscape, astronaut helmet, cinematic framing",
    "NYC at night, Times Square, neon signs, crowd, rain reflections",
    "A giant octopus emerging from the ocean near a lighthouse, stormy waves, dramatic lighting, wide-angle view",
    "A futuristic medical robot performing surgery in a sterile operating room, bright overhead lights",
    "A snowy owl in flight over a winter field, crisp feathers, zoomed in",
    "A busy Tokyo subway platform during rush hour, commuters, fluorescent lighting",
    "A vintage motorcycle parked beside a brick wall, golden hour sunlight",
    "A sailboat cutting through choppy ocean waves under a dramatic cloudy sky, spray and motion, wide shot",
]

#target_text = "cartoon, flat, bold lines"
#target_text = "photorealistic, DSLR, HD"
target_text = "blue"

results = sdxl_thickets_global_select_train_and_test(
    train_prompts=train_prompts,
    test_prompts=test_prompts,
    target_text=target_text,
    N=128,
    K=8,
    sigma=3e-3,
    steps=30,
    guidance=5.0,
    width=512,
    height=512,
    seed=0,
    negative_prompt="low quality, blurry, distorted",
    perturb="unet",
    outdir="out_thickets_global_select",
    dtype=torch.float16,
    clip_model_id="openai/clip-vit-large-patch14",
    clip_batch=8,
    viz_random=6,
    viz_seed=123,
    prompt_batch_size=4,
    show=True,
    save=True,
)
