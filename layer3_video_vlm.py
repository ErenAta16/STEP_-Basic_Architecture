"""
Layer 3v — video understanding with Gemini 2.5 Pro.

Produces three fields per video: ``title``, ``summary`` and up to five
``keywords`` copied verbatim from a closed keyword pool. Both the YouTube
URL path and the uploaded-file path go through the same prompt so outputs
stay comparable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_VIDEO_MODEL
from keyword_eval import DEFAULT_KEYWORD_POOL
from layer0_video import canonical_youtube_url, upload_local_video

_log = logging.getLogger(__name__)


_SYSTEM = (
    "You watch a mathematics video and classify it with keywords. "
    "Return EXACTLY three lines, no other output:\n"
    "1) TITLE: <short phrase describing the video subject>\n"
    "2) SUMMARY: <two short sentences in English>\n"
    "3) KEYWORDS: <exactly five comma-separated keywords copied verbatim from "
    "the closed list, ordered by relevance (most relevant first). If fewer "
    "than five items truly fit, pick the closest remaining ones so the count "
    "stays at five>"
)


def _user_prompt(pool: list[str]) -> str:
    return (
        "Closed keyword pool (use these exact phrasings only for KEYWORDS):\n"
        + ", ".join(pool)
        + "\n\nAnalyse the video and return the three required lines."
    )


def _parse_reply(text: str, pool: list[str]) -> dict:
    pool_lookup = {p.lower(): p for p in pool}
    title = ""
    summary = ""
    keywords: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif upper.startswith("KEYWORDS:"):
            parts = [p.strip() for p in line.split(":", 1)[1].split(",") if p.strip()]
            seen: set[str] = set()
            for part in parts:
                canon = pool_lookup.get(part.lower())
                if canon and canon not in seen:
                    keywords.append(canon)
                    seen.add(canon)
    return {"title": title, "summary": summary, "keywords": keywords[:5]}


def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=GEMINI_API_KEY)


def _build_result(raw_text: str, pool: list[str], model: str, elapsed_s: float) -> dict:
    parsed = _parse_reply(raw_text, pool)
    return {
        **parsed,
        "raw": raw_text,
        "pool": pool,
        "model_used": model,
        "elapsed_s": elapsed_s,
    }


def analyze_youtube(url: str, *, pool: list[str] | None = None,
                     model: str | None = None, verbose: bool = True) -> dict:
    """Analyse a public YouTube video via its canonical URL."""
    canon = canonical_youtube_url(url)
    if not canon:
        raise ValueError(f"Not a valid YouTube URL: {url!r}")

    pool = list(pool) if pool else list(DEFAULT_KEYWORD_POOL)
    model = (model or GEMINI_VIDEO_MODEL).strip()

    if verbose:
        _log.info(f"  [L3v] YouTube analysis: {canon}")
    client = _client()
    part_video = types.Part(file_data=types.FileData(file_uri=canon, mime_type="video/*"))
    part_prompt = types.Part(text=_user_prompt(pool))
    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=[part_video, part_prompt]),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            max_output_tokens=4096,
            temperature=0.0,
        ),
    )
    elapsed = round(time.time() - t0, 1)
    text = (response.text or "").strip()
    if verbose:
        _log.info(f"  [L3v] [OK] {elapsed}s")
    return _build_result(text, pool, model, elapsed)


def analyze_local_file(path: str | Path, *, pool: list[str] | None = None,
                        model: str | None = None, verbose: bool = True) -> dict:
    """Analyse an uploaded video file via the Gemini Files API."""
    path = Path(path)
    pool = list(pool) if pool else list(DEFAULT_KEYWORD_POOL)
    model = (model or GEMINI_VIDEO_MODEL).strip()

    client = _client()
    uploaded = upload_local_video(client, path)
    if verbose:
        _log.info(f"  [L3v] Local analysis: {path.name}")
    part_prompt = types.Part(text=_user_prompt(pool))
    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=[uploaded, part_prompt],
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            max_output_tokens=4096,
            temperature=0.0,
        ),
    )
    elapsed = round(time.time() - t0, 1)
    text = (response.text or "").strip()
    if verbose:
        _log.info(f"  [L3v] [OK] {elapsed}s")
    return _build_result(text, pool, model, elapsed)
