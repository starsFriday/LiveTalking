"""Single-call Gemini audio understanding with Google Search grounding."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import wave
from typing import Optional

import aiohttp
import numpy as np


def _proxy_url() -> Optional[str]:
    return (
        os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
        or None
    )


class GeminiAudioSearchTool:
    """Listen to one utterance and search only when fresh facts are required."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.1-flash-lite",
        max_chars: int = 320,
        timeout_seconds: float = 12.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.max_chars = max(160, int(max_chars))
        self.timeout_seconds = max(3.0, float(timeout_seconds))

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.model)

    async def search_audio(
        self,
        audio_16k: np.ndarray,
    ) -> Optional[tuple[str, str]]:
        """Return grounded facts only for real-time intents; otherwise return None."""
        audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if audio.size < 4000:
            return None
        payload = {
            "model": self.model,
            "input": [
                {
                    "type": "text",
                    "text": (
                        "听写这段中文语音；如果内容询问天气、新闻、价格、股价、汇率、赛程、交通、政策、"
                        "当前人物职位、产品最新信息，或用户明确要求查询、搜索，必须调用 Google 搜索核验，"
                        "然后输出‘转写：用户原问题；资料：简洁中文实时事实’。资料不含Markdown和网址，"
                        "最多180个汉字。如果只是寒暄、闲聊、创作、常识、情感交流或观察摄像头，"
                        "不要调用搜索，只输出 NO_SEARCH。"
                    ),
                },
                {
                    "type": "audio",
                    "data": base64.b64encode(self._wav_bytes(audio)).decode("ascii"),
                    "mime_type": "audio/wav",
                },
            ],
            "tools": [{"type": "google_search"}],
        }
        timeout = aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            connect=min(8.0, self.timeout_seconds),
        )
        async with aiohttp.ClientSession(timeout=timeout) as http:
            async with http.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                proxy=_proxy_url(),
            ) as response:
                raw = await response.text()
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    body = {}
                if response.status >= 400:
                    error = body.get("error") if isinstance(body, dict) else None
                    message = error.get("message") if isinstance(error, dict) else raw[:240]
                    raise RuntimeError(message or f"Gemini Interactions HTTP {response.status}")

        steps = body.get("steps", []) if isinstance(body, dict) else []
        used_search = any(
            isinstance(step, dict) and step.get("type") == "google_search_call"
            for step in steps
        )
        if not used_search:
            return None

        output_text = str(body.get("output_text") or "").strip()
        if not output_text:
            parts: list[str] = []
            for step in steps:
                if not isinstance(step, dict) or step.get("type") != "model_output":
                    continue
                for part in step.get("content", []):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            output_text = "".join(parts).strip()
        if not output_text or output_text.upper() == "NO_SEARCH":
            return None

        decoded = self._parse_json(output_text)
        question = self._clean(decoded.get("question", "")) if decoded else ""
        facts = self._clean(decoded.get("facts", "")) if decoded else ""
        if not decoded:
            match = re.search(r"转写[：:]\s*(.*?)\s*[；;]\s*资料[：:]\s*(.*)", output_text, re.S)
            if match:
                question = self._clean(match.group(1))
                facts = self._clean(match.group(2))
            else:
                facts = self._clean(output_text)
        if not facts:
            return None
        return question or "用户刚才询问的实时信息", facts[: self.max_chars]

    @staticmethod
    def _wav_bytes(audio: np.ndarray) -> bytes:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm.tobytes())
        return output.getvalue()

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        candidate = text.strip()
        candidate = re.sub(r"^\`\`\`(?:json)?\s*|\s*\`\`\`$", "", candidate, flags=re.I)
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", candidate)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _clean(text: str) -> str:
        value = str(text or "")
        value = re.sub(r"https?://\S+", "", value)
        value = re.sub(r"\[\[?\d+\]?\]\([^)]*\)", "", value)
        value = re.sub(r"[\`*_#]+", "", value)
        return re.sub(r"\s+", " ", value).strip()

