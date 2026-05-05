"""
VideoAnalyzer — high-level entry point for single-video analysis.

Wraps ingestion + VLM extraction + taxonomy classification so the web layer
only has to call one method. Results are cached on disk under
``step_pipeline/video_cache/`` keyed by YouTube video id or file SHA-256, so
repeat analysis does not re-spend the LLM budget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from config import VIDEO_CACHE_DIR, ensure_dirs
from layer0_video import (
    canonical_youtube_url,
    download_youtube_video,
    extract_frames,
    youtube_video_id,
)
from layer3_video_vlm import analyze_local_file, analyze_youtube
from layer3v_frames import analyze_frames_deep
from taxonomy import classify_taxonomy, topic_from_keywords

_log = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyse a single video (YouTube URL or local file) into structured output."""

    def __init__(self) -> None:
        ensure_dirs()

    # ------------------------------------------------------------------
    # YouTube URL path
    # ------------------------------------------------------------------
    def analyze_youtube(self, url: str, *, mode: str = "quick",
                         use_cache: bool = True) -> dict:
        canon = canonical_youtube_url(url)
        if not canon:
            return {"media": "video", "error": "Invalid YouTube URL"}
        vid = youtube_video_id(canon) or "unknown"
        mode = (mode or "quick").lower()
        cache_key = f"yt_{vid}_{mode}.json"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                cached["cached"] = True
                return cached

        t0 = time.time()
        try:
            vlm = analyze_youtube(canon)
        except Exception as e:
            return {
                "media": "video",
                "source": "youtube",
                "url": canon,
                "video_id": vid,
                "mode": mode,
                "error": str(e)[:200],
            }

        result = self._finalize(
            vlm,
            base={
                "media": "video",
                "source": "youtube",
                "url": canon,
                "video_id": vid,
                "mode": mode,
            },
            elapsed_s=round(time.time() - t0, 1),
        )

        if mode == "deep":
            self._attach_deep(result, source_hint=canon, vid=vid, is_youtube=True)

        # Only cache successful runs so transient failures can be retried.
        if not (mode == "deep" and result.get("deep_error")):
            self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Uploaded file path
    # ------------------------------------------------------------------
    def analyze_file(self, path: str | Path, *, mode: str = "quick",
                      use_cache: bool = True) -> dict:
        path = Path(path)
        if not path.exists():
            return {"media": "video", "error": f"File not found: {path.name}"}
        sha = self._sha256(path)
        mode = (mode or "quick").lower()
        cache_key = f"file_{sha}_{mode}.json"

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                cached["cached"] = True
                return cached

        t0 = time.time()
        try:
            vlm = analyze_local_file(path)
        except Exception as e:
            return {
                "media": "video",
                "source": "upload",
                "file": path.name,
                "file_sha": sha,
                "mode": mode,
                "error": str(e)[:200],
            }

        result = self._finalize(
            vlm,
            base={
                "media": "video",
                "source": "upload",
                "file": path.name,
                "file_sha": sha,
                "mode": mode,
            },
            elapsed_s=round(time.time() - t0, 1),
        )

        if mode == "deep":
            self._attach_deep(result, source_hint=path, vid=None, is_youtube=False)

        if not (mode == "deep" and result.get("deep_error")):
            self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Deep mode: frame sampling + per-scene keyword extraction
    # ------------------------------------------------------------------
    def _attach_deep(self, result: dict, *, source_hint, vid: str | None,
                      is_youtube: bool) -> None:
        """Run the Deep pipeline and mutate ``result`` in place with its output."""
        try:
            if is_youtube:
                downloads_dir = VIDEO_CACHE_DIR / "downloads"
                video_path = download_youtube_video(str(source_hint), downloads_dir)
            else:
                video_path = Path(source_hint)
            frames = extract_frames(video_path)
            _log.info(f"  [L0v] Extracted {len(frames)} frame(s) from {video_path.name}")
            if not frames:
                result["deep_error"] = "No frames could be extracted"
                result["problems"] = []
                return
            deep = analyze_frames_deep(frames)
            result["problems"] = deep.get("problems", [])
            result["deep_elapsed_s"] = deep.get("elapsed_s")
            result["deep_frame_count"] = len(frames)
        except Exception as e:
            _log.info(f"  [L3vd] [FAIL] {str(e)[:120]}")
            result["deep_error"] = str(e)[:200]
            result["problems"] = []

    # ------------------------------------------------------------------
    # Shared finalization / cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finalize(vlm: dict, *, base: dict, elapsed_s: float) -> dict:
        summary_text = vlm.get("summary") or ""
        keywords = vlm.get("keywords", []) or []
        tax = classify_taxonomy(summary_text)

        # Regex taxonomy targets LaTeX / problem prose and often misses English
        # video summaries. Fall back to the already-chosen keywords (all from a
        # known subtopic pool) when the regex path finds nothing.
        if not tax.get("topic") and keywords:
            derived = topic_from_keywords(keywords)
            if derived:
                tax = {
                    "topic": derived[0],
                    "subtopic": derived[1],
                    "keywords": list(keywords[:5]),
                }

        return {
            **base,
            "title": vlm.get("title", ""),
            "summary": summary_text,
            "keywords": keywords,
            "pool": vlm.get("pool", []),
            "model_used": vlm.get("model_used", ""),
            "vlm_elapsed_s": vlm.get("elapsed_s"),
            "elapsed_s": elapsed_s,
            "taxonomy": tax,
            "cached": False,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _load_cache(name: str) -> dict | None:
        fp = VIDEO_CACHE_DIR / name
        if not fp.exists():
            return None
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _save_cache(name: str, data: dict) -> None:
        fp = VIDEO_CACHE_DIR / name
        try:
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as e:
            _log.info(f"  [L3v] [WARN] Video cache write failed: {e}")
