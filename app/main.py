from __future__ import annotations

import asyncio
import base64
import logging
import re
import resource
import time
from dataclasses import dataclass
from typing import Any, Deque, Optional
from collections import deque



from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.gemini_service import GeminiPromptImprover
from app.sd_generator import MODES, StableDiffusionCPUGenerator, configure_torch_threads


load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


STYLES = ["Realistic", "Anime", "Cinematic", "3D Render", "Fantasy Art"]
DEFAULT_MODE = "Balanced"
GEN_TIMEOUT_SECONDS = 180

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)


@dataclass
class GenerationRecord:
    ts: float
    prompt: str
    style: str
    steps: int
    image_png_b64: str


recent_generations: Deque[GenerationRecord] = deque(maxlen=5)

_sd = StableDiffusionCPUGenerator()
_gemini: Optional[GeminiPromptImprover] = None
_generation_lock = asyncio.Lock()
_torch_threads: Optional[int] = None


@app.on_event("startup")
async def _startup() -> None:
    global _torch_threads
    _torch_threads = configure_torch_threads(max_threads=4)
    await run_in_threadpool(_sd.init)
    logger.info(
        "startup: sd pipeline initialized (cpu-only), torch_threads=%s, modes=%s",
        torch.get_num_threads(),
        ",".join(MODES.keys()),
    )


def _get_gemini() -> GeminiPromptImprover:
    global _gemini
    if _gemini is None:
        _gemini = GeminiPromptImprover()
    return _gemini


class ImproveRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    style: str = Field(min_length=1, max_length=50)


class ImproveResponse(BaseModel):
    improved_prompt: str


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)
    style: str = Field(min_length=1, max_length=50)
    mode: str = Field(default=DEFAULT_MODE, min_length=1, max_length=20)
    steps: int = Field(default=12, ge=1, le=20)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=7.5)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "styles": STYLES,
            "modes": list(MODES.keys()),
            "default_mode": DEFAULT_MODE,
        },
    )


_ws_re = re.compile(r"\s+")


def _sanitize_prompt(value: str) -> str:
    value = (value or "").replace("\x00", " ")
    value = value.strip()
    value = _ws_re.sub(" ", value)
    return value


@app.post("/improve", response_model=ImproveResponse)
async def improve(req: ImproveRequest) -> ImproveResponse:
    if req.style not in STYLES:
        raise HTTPException(status_code=400, detail="Invalid style")

    prompt = _sanitize_prompt(req.prompt)
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    gemini = _get_gemini()

    try:
        improved = await run_in_threadpool(gemini.improve, prompt, req.style)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    improved = (improved or "").strip()
    if not improved:
        raise HTTPException(status_code=500, detail="Gemini returned empty response")

    return ImproveResponse(improved_prompt=improved)


def _style_prefix(style: str) -> str:
    mapping = {
        "Realistic": "photorealistic,",
        "Anime": "anime style,",
        "Cinematic": "cinematic,",
        "3D Render": "3d render,",
        "Fantasy Art": "fantasy art,",
    }
    return mapping.get(style, "")


@app.post("/generate")
async def generate(req: GenerateRequest) -> Response:
    if req.style not in STYLES:
        raise HTTPException(status_code=400, detail="Invalid style")

    if req.mode not in MODES:
        raise HTTPException(status_code=400, detail="Invalid mode")

    prompt = _sanitize_prompt(req.prompt)
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    final_prompt = f"{_style_prefix(req.style)} {prompt}".strip()

    try:
        await asyncio.wait_for(_generation_lock.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="Generation already in progress")
    t0 = time.perf_counter()
    try:
        try:
            png_bytes = await asyncio.wait_for(
                run_in_threadpool(
                    _sd.generate_png_bytes,
                    final_prompt,
                    mode=req.mode,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance_scale,
                ),
                timeout=GEN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Generation timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        _generation_lock.release()

    gen_s = time.perf_counter() - t0
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info(
        "generate: mode=%s steps=%s guidance=%s time_s=%.2f rss_kb=%s torch_threads=%s",
        req.mode,
        req.steps,
        req.guidance_scale,
        gen_s,
        rss_kb,
        torch.get_num_threads(),
    )

    recent_generations.appendleft(
        GenerationRecord(
            ts=time.time(),
            prompt=final_prompt,
            style=req.style,
            steps=req.steps,
            image_png_b64=base64.b64encode(png_bytes).decode("utf-8"),
        )
    )

    return Response(content=png_bytes, media_type="image/png")


@app.get("/recent")
async def recent() -> JSONResponse:
    payload = [
        {
            "ts": r.ts,
            "prompt": r.prompt,
            "style": r.style,
            "steps": r.steps,
            "data_url": f"data:image/png;base64,{r.image_png_b64}",
        }
        for r in list(recent_generations)
    ]
    return JSONResponse(payload)
