from __future__ import annotations

import os

import google.generativeai as genai


SYSTEM_INSTRUCTION = (
    "You are an expert Stable Diffusion prompt engineer. "
    "Your job is to rewrite user ideas into Stable Diffusion v1.5 optimized prompts. "
    "Return ONLY the final prompt text, no quotes, no markdown, no extra commentary. "
    "Keep the final prompt UNDER 60 words. "
    "Always add: lighting details, camera/lens details, environment details, "
    "and quality keywords (ultra detailed, high resolution, 4k)."
)


class GeminiPromptImprover:
    def __init__(self, model_name: str = "gemini-1.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def improve(self, user_prompt: str, style: str) -> str:
        prompt = (
            "Rewrite the following into a Stable Diffusion v1.5 prompt.\n"
            f"Style: {style}\n"
            f"User idea: {user_prompt}\n"
            "Constraints: under 60 words, add lighting, camera/lens, environment, "
            "and quality keywords. Return only the final prompt text."
        )

        resp = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.6,
                "max_output_tokens": 128,
            },
        )

        text = (resp.text or "").strip()
        return text
