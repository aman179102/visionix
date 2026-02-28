from __future__ import annotations

import gc
import io
import os
import time
from dataclasses import dataclass
from typing import Optional


import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


@dataclass(frozen=True)
class GenerationSettings:
    width: int
    height: int
    steps: int


MODES: dict[str, GenerationSettings] = {
    "Fast": GenerationSettings(width=384, height=384, steps=10),
    "Balanced": GenerationSettings(width=512, height=512, steps=12),
    "Quality": GenerationSettings(width=512, height=512, steps=18),
}


_PIPE: Optional[StableDiffusionPipeline] = None


def configure_torch_threads(max_threads: int = 4) -> int:
    cpu_count = os.cpu_count() or 4
    threads = max(1, min(max_threads, max(1, cpu_count // 2)))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, threads // 2))
    except Exception:
        pass
    return threads


def init_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5") -> StableDiffusionPipeline:
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    kwargs = {
        "torch_dtype": torch.float32,
        "safety_checker": None,
        "feature_extractor": None,
    }
    try:
        kwargs["low_cpu_mem_usage"] = True
    except Exception:
        pass

    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("low_cpu_mem_usage", None)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    _PIPE = pipe
    return _PIPE


def get_pipeline() -> StableDiffusionPipeline:
    if _PIPE is None:
        raise RuntimeError("Stable Diffusion pipeline not initialized")
    return _PIPE


class StableDiffusionCPUGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5") -> None:
        self.model_id = model_id

    def init(self) -> None:
        init_pipeline(self.model_id)

    def generate_png_bytes(
        self,
        prompt: str,
        *,
        mode: str = "Balanced",
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.0,
        seed: Optional[int] = None,
    ) -> bytes:
        pipe = get_pipeline()

        settings = MODES.get(mode) or MODES["Balanced"]
        steps = int(num_inference_steps) if num_inference_steps is not None else settings.steps
        steps = max(1, min(20, steps))
        guidance_scale = float(max(1.0, min(7.5, guidance_scale)))

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        t0 = time.perf_counter()
        result = None
        image = None
        buf = None
        try:
            with torch.inference_mode():
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    width=settings.width,
                    height=settings.height,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            image = result.images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()
        finally:
            del result
            del image
            del buf
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()
            _ = time.perf_counter() - t0
