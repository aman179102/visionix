# Visionix
## Local-first AI Text-to-Image (CPU) + Improve Prompt

A lightweight local-first web app that generates images with **Stable Diffusion v1.5 (CPU-only)** and improves prompts via the **Gemini API (server-side)**.

## Features

- **Text-to-Image (CPU-only)** using `runwayml/stable-diffusion-v1-5` via Hugging Face `diffusers`
- **Improve Prompt** button
  - Sends your prompt + style to `/improve`
  - Gemini rewrites it into an SD-optimized prompt (lighting, camera/lens, environment, quality keywords)
  - Returns only the final prompt text
- **Performance modes**
  - **Fast**: 384x384, 10 steps
  - **Balanced**: 512x512, 12 steps (default)
  - **Quality**: 512x512, 18 steps
- **CPU safety / low system load**
  - Stable Diffusion pipeline loads **once at startup** (singleton)
  - Attention slicing + VAE slicing enabled
  - Torch threads capped (half your cores, max 4)
  - Only **one generation at a time** (concurrent requests get HTTP `429`)
  - Request timeout protection
  - Cleanup after generation (`gc.collect()`)
- **UX**
  - Loading spinner during improve/generation
  - Image preview + download button
  - Keeps last **5 generations** in memory

## Tech Stack

- **Backend**: FastAPI (Python)
- **Image generation**: `diffusers` + `torch` (CPU)
- **Prompt improvement**: Gemini (`google-generativeai`) server-side
- **Frontend**: HTML + CSS + minimal JS

## Requirements

- Python 3.10+ (works best on Linux)
- 16GB RAM recommended (CPU-only generation is slow but stable)
- A Gemini API key

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

Notes:
- `requirements.txt` is configured to install **CPU-only PyTorch wheels**.

### 3) Configure environment variables

Create/edit `.env`:

```env
GEMINI_API_KEY=YOUR_KEY_HERE
```

Important:
- `.env` is included in `.gitignore` and should not be committed.

## Run

```bash
uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000

## API Endpoints

- `GET /` - Web UI
- `POST /improve` - Improve prompt via Gemini
  - Body: `{ "prompt": "...", "style": "Realistic" }`
- `POST /generate` - Generate image (PNG)
  - Body: `{ "prompt": "...", "style": "Realistic", "mode": "Balanced", "steps": 12, "guidance_scale": 7.0 }`
- `GET /recent` - Last 5 generations (as data URLs)

## Notes on CPU-only generation

- First run will download the Stable Diffusion model and can take time.
- CPU inference is intentionally constrained to reduce spikes:
  - single-flight generation
  - thread cap
  - capped steps/guidance

## Project Structure

```text
app/
  main.py
  sd_generator.py
  gemini_service.py
templates/
  index.html
static/
  style.css
.env.example
requirements.txt

```
