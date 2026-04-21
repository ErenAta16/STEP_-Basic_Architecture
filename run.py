"""
CLI entry for the STEP PDF -> LLM pipeline (``STEPSolver``).

Uses ``logging`` (via ``step_logging.configure_logging``) for stdout-friendly
lines and ``config.ensure_dirs`` before touching pipeline paths.

Usage:
    python run.py problem.pdf
    python run.py problem.pdf --no-nougat
    python run.py Surface_Integration/
    python run.py Surface_Integration/ -n 5
    python run.py --check
"""

import argparse
import hashlib
import logging
import re
import sys
import threading
import time
import platform
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import torch
import sympy as sp

from config import (PDF_DIR, IMG_DIR, NOUGAT_OUT, VLM_OUT, RESULTS_DIR,
                    NOUGAT_DPI, VLM_ONLY_DPI,
                    get_system_prompt, ensure_dirs)
from parallel_ocr import run_parallel_nougat_vlm
from step_logging import configure_logging
from layer0_ingestion import Layer0_PDFIngestion
from layer1_profiler import Layer1_Profiler
from layer2_nougat import Layer2_Nougat
from layer3_vlm import Layer3_VLM
from layer4_synthesis import Layer4_Synthesis
from layer5_llm_solver import Layer5_LLMSolver
from layer6_verifier import Layer6_SymPyVerifier
from pipeline_logger import PipelineLogger
from taxonomy import classify_taxonomy, keywords_for_subtopic, merge_keywords

_log = logging.getLogger(__name__)


class STEPSolver:
    """Orchestrates layers 0–6: ingest, profile, optional Nougat/VLM (via ``parallel_ocr``), synthesis, LLM, extract."""

    def __init__(self, use_nougat: bool = False, use_vlm: bool = True):
        self.l0 = Layer0_PDFIngestion(IMG_DIR)
        self.l1 = Layer1_Profiler()
        self.l2 = Layer2_Nougat(IMG_DIR, NOUGAT_OUT) if use_nougat else None
        self.l3 = Layer3_VLM() if use_vlm else None
        self.l4 = Layer4_Synthesis()
        self.l5_primary = Layer5_LLMSolver(force_provider="gemini")
        self.l5_fallback = Layer5_LLMSolver(force_provider="together")
        self.l5 = self.l5_primary if self.l5_primary.is_available else self.l5_fallback
        self.l6 = Layer6_SymPyVerifier()
        self.use_nougat = use_nougat
        self.use_vlm = use_vlm
        self._result_cache: OrderedDict[str, dict] = OrderedDict()
        self._result_cache_max = 32
        # Protects ``_result_cache`` so the solver can be shared across
        # concurrent solves (e.g. Flask workers) without races on the OrderedDict.
        self._cache_lock = threading.Lock()

    @staticmethod
    def _pdf_hash(pdf_path: Path) -> str:
        import hashlib
        return hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    def solve(self, pdf_path: str | Path, verbose: bool = True,
              user_query: str | None = None) -> dict:
        """Run all enabled stages on one PDF.

        On a cache miss, ensures directories exist and attaches logging to stdout.
        On success the dict includes timings, solution text, and ``final_answer``
        (display string). On failure look for an ``error`` key; partial timings
        may still be present.

        ``user_query`` is an optional free-form instruction appended to the
        Layer 4 prompt (e.g. "answer in decimal form", "explain in Portuguese").
        It is part of the cache key so distinct notes yield distinct runs.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}

        note = (user_query or "").strip()
        note_key = hashlib.sha1(note.encode("utf-8")).hexdigest()[:12] if note else "-"

        h = self._pdf_hash(pdf_path)
        path_key = str(pdf_path.resolve())
        # Per resolved path + content hash + OCR flags + user note digest
        # (same bytes at two paths -> separate entries; differing notes too).
        cache_key = (
            f"{path_key}|{h}|nougat={int(self.use_nougat)}|vlm={int(self.use_vlm)}"
            f"|note={note_key}"
        )
        with self._cache_lock:
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                self._result_cache.move_to_end(cache_key)
        if cached is not None:
            if verbose:
                _log.info(f"  [CACHE] Returning previous result for {pdf_path.name}")
            return cached

        ensure_dirs()
        configure_logging()

        fname = pdf_path.stem
        result = {"file": fname, "pdf_path": str(pdf_path)}
        t_start = time.time()
        timings = {}

        # --- L0: PDF Ingestion ---
        t0 = time.time()
        # Align DPI with the downstream consumer: Nougat needs high-DPI pages to
        # read equations reliably; the VLM path downscales to a fixed long edge,
        # so rasterizing at 400 DPI there is pure overhead. Saves wall time and
        # disk on scanned/handwritten inputs (the common VLM case).
        render_dpi = NOUGAT_DPI if self.use_nougat else VLM_ONLY_DPI
        metadata, raw_pages, _img_meta = self.l0.extract_metadata_text_and_images(
            pdf_path, dpi=render_dpi
        )
        raw_text = "\n".join(p["text"] for p in raw_pages).strip()
        # PyMuPDF4LLM markdown is only useful when raw text is thin. Otherwise
        # we pay an extra ``fitz.open`` pass for content L4 will prefer to
        # ignore (it already takes ``base_text = max(len(md), len(raw))``).
        if len(raw_text) < 800:
            md_text = self.l0.extract_markdown(pdf_path, text_pages=raw_pages)
        else:
            md_text = raw_text
        text_quality = self.l0.analyze_text_quality(raw_pages)
        result["pages"] = metadata.get("pages", 0)
        result["chars"] = sum(len(p["text"]) for p in raw_pages)
        result["text_quality_score"] = text_quality["score"]
        timings["l0_ingest"] = round(time.time() - t0, 3)
        if verbose:
            _log.info(
                f"  [L0] {metadata.get('pages')} pages, {result['chars']} chars, "
                f"quality {text_quality['score']}/{text_quality['max_score']} ({timings['l0_ingest']:.2f}s)"
            )

        # --- L1: quick tags from raw text (cheap classifier) ---
        t1 = time.time()
        profile = self.l1.profile(fname, metadata, raw_text)
        timings["l1_profile"] = round(time.time() - t1, 4)
        if verbose:
            # Raw text can be empty on scanned/image PDFs; in that case L1b (after OCR/VLM)
            # is the meaningful profile update.
            if result["chars"] == 0 and self.use_vlm:
                _log.info(f"  [L1] deferred (raw text empty) ({timings['l1_profile']:.3f}s)")
            else:
                sec = profile.get("secondary_categories") or []
                sec_s = f" | also: {', '.join(sec)}" if sec else ""
                _log.info(
                    f"  [L1] {profile['category']} / {profile['surface_type']}{sec_s} "
                    f"({timings['l1_profile']:.3f}s)"
                )

        nougat_latex = ""
        nougat_score = 0
        vlm_latex = ""
        vlm_score = 0
        timings["l2_nougat"] = 0.0
        timings["l3_vlm"] = 0.0

        # Shared with ``pipeline.STEPPipeline`` (skip rules + ThreadPoolExecutor).
        ocr = run_parallel_nougat_vlm(
            pdf_path=pdf_path,
            fname=fname,
            img_dir=IMG_DIR,
            nougat_layer=self.l2,
            vlm_layer=self.l3,
            use_nougat=self.use_nougat,
            use_vlm=self.use_vlm,
            vlm_available=self.l3 is not None and self.l3.is_available,
            text_quality=text_quality,
            total_chars=result["chars"],
            verbose=verbose,
            nougat_verbose=verbose,
        )
        if ocr["nougat_pkg"]:
            res_n, q_n, _ = ocr["nougat_pkg"]
            nougat_latex = res_n.get("latex", "")
            nougat_score = q_n["score"]
        timings["l2_nougat"] = ocr["t2_elapsed"]
        if ocr["vlm_pkg"]:
            res_v, q_v, _ = ocr["vlm_pkg"]
            vlm_latex = res_v.get("vlm_latex", "")
            vlm_score = q_v["score"]
        timings["l3_vlm"] = ocr["t3_elapsed"]

        # --- L1b: Re-profile with OCR/VLM text if raw was empty ---
        if result["chars"] == 0 and (vlm_latex or nougat_latex):
            ocr_text = vlm_latex or nougat_latex
            profile = self.l1.profile(fname, metadata, raw_text, latex_text=ocr_text)
            if verbose:
                _log.info(f"  [L1b] Re-profiled: {profile['category']} / {profile['surface_type']}")
                sec = profile.get("secondary_categories") or []
                sec_s = f", signals={sec}" if sec else ""
                _log.info(f"  [L1] final={profile['category']} / {profile['surface_type']}{sec_s}")

        result["category"] = profile["category"]
        result["surface_type"] = profile["surface_type"]
        result["keywords"] = profile["keywords"]
        result["summary"] = profile["summary"]
        result["secondary_categories"] = profile.get("secondary_categories", [])

        # MathE-style taxonomy (Topic / Subtopic / Keywords). Classification uses
        # every piece of text we have: native PyMuPDF, OCR/VLM, and the filename.
        classify_text = " \n".join(t for t in (raw_text, vlm_latex, nougat_latex, fname) if t)
        taxonomy = classify_taxonomy(classify_text)
        result["taxonomy"] = taxonomy
        if verbose and (taxonomy.get("topic") or taxonomy.get("keywords")):
            topic = taxonomy.get("topic") or "?"
            sub = taxonomy.get("subtopic") or "?"
            kws = ", ".join(taxonomy.get("keywords", [])) or "-"
            _log.info(f"  [L1b] Taxonomy: {topic} / {sub} | kw: {kws}")

        # --- L4: Synthesis ---
        t4 = time.time()
        synthesis = self.l4.synthesize(raw_text, nougat_latex, nougat_score,
                                        vlm_latex, vlm_score, profile,
                                        md_text=md_text)
        prompt = synthesis["prompt"]
        domain = synthesis.get("domain", "general_math")

        # Optional free-form instruction supplied by the user. Injected at the
        # tail of the prompt so the solver still sees the extracted problem in
        # full, but can honour language / formatting / focus requests.
        if note:
            prompt = (
                prompt
                + "\n\n--- USER INSTRUCTION (respect if compatible with the math) ---\n"
                + note
            )
            result["user_query"] = note
        result["source"] = synthesis["source"]
        result["domain"] = domain
        result["nougat_score"] = nougat_score
        result["vlm_score"] = vlm_score
        result["prompt_chars"] = synthesis.get("prompt_chars", len(prompt))
        timings["l4_synthesis"] = round(time.time() - t4, 4)
        if verbose:
            domain_label = "surface_integral" if domain == "surface_integral" else "general_math"
            _log.info(
                f"  [L4] source={synthesis['source']}, domain={domain_label}, "
                f"prompt_chars={synthesis['prompt_chars']}"
            )

        # Fast path: for parseable one-variable definite integrals, run exactly one
        # control LLM pass, then return deterministic SymPy final answer.
        if result.get("category") == "definite_integral":
            fast_sympy = self._sympy_definite_integral_from_prompt(prompt)
            if fast_sympy:
                system_prompt = get_system_prompt(
                    domain,
                    secondary_categories=profile.get("secondary_categories"),
                    primary_category=profile.get("category"),
                )
                result["l5_system"] = {
                    "domain": domain,
                    "secondary_categories": list(profile.get("secondary_categories") or []),
                }
                t5 = time.time()
                checks = self._solve_with_consensus(
                    prompt,
                    system_prompt,
                    max_attempts=1,
                    verbose=verbose,
                    allow_extra_provider_attempt=False,
                )
                timings["l5_llm"] = round(time.time() - t5, 2)
                best_check = next((a for a in checks if a.get("solution")), checks[0] if checks else {})
                result["attempts"] = 1
                result["solution"] = best_check.get("solution", "")
                result["solution_chars"] = len(best_check.get("solution", ""))
                result["consensus"] = False
                result["model_used"] = best_check.get("model", "sympy/definite-integral-fast-path")
                result["sympy_definite_verified"] = True
                result["final_answer"] = fast_sympy
                result["llm_summary"] = self._extract_llm_summary(result["solution"]) if result["solution"] else {}
                # Follow-up context (server-side only; stripped by the web layer).
                result["_prompt"] = prompt
                result["_system_prompt"] = system_prompt
                timings["l6_extract"] = 0.0
                result["timings"] = timings
                result["elapsed_s"] = round(time.time() - t_start, 1)
                if verbose:
                    _log.info("  [L5] [CHECK] One control pass completed (definite integral)")
                    _log.info("  [L6] [SYMPY] deterministic definite integral override")
                    _log.info("  [L6] [OK] extracted final line for display")
                with self._cache_lock:
                    self._result_cache[cache_key] = result
                    self._result_cache.move_to_end(cache_key)
                    while len(self._result_cache) > self._result_cache_max:
                        self._result_cache.popitem(last=False)
                return result

        # No readable input: avoid wasting LLM calls and returning misleading
        # "no problem provided" style answers from the model.
        no_raw = result["chars"] == 0
        no_useful_ocr = nougat_score == 0 and vlm_score == 0
        if no_raw and no_useful_ocr:
            result["error"] = "No readable mathematical content detected in the PDF"
            result["final_answer"] = "(input not readable)"
            result["timings"] = timings
            result["elapsed_s"] = round(time.time() - t_start, 1)
            if verbose:
                _log.info("  [L5] [SKIP] No readable content after L0/L2/L3 quality checks; skipping LLM")
            return result

        # --- L5: LLM Solve with Consensus (2-3 attempts) ---
        system_prompt = get_system_prompt(
            domain,
            secondary_categories=profile.get("secondary_categories"),
            primary_category=profile.get("category"),
        )
        result["l5_system"] = {
            "domain": domain,
            "secondary_categories": list(profile.get("secondary_categories") or []),
        }
        if verbose:
            dom_tag = "surface" if domain == "surface_integral" else "general"
            sec = profile.get("secondary_categories") or []
            sec_part = f", signals={sec}" if sec else ""
            _log.info(f"  [L5] system={dom_tag}{sec_part}")
        # Speed profile for Gemini Pro: keep strong first pass, but avoid stacking
        # multiple long Pro calls in a row.
        max_attempts = 2
        primary_model = (self.l5_primary.model_name or "").lower()
        if self.l5_primary.is_available and "gemini-2.5-pro" in primary_model:
            max_attempts = 1
        t5 = time.time()
        answers = self._solve_with_consensus(
            prompt,
            system_prompt,
            max_attempts=max_attempts,
            verbose=verbose,
        )
        timings["l5_llm"] = round(time.time() - t5, 2)
        result["attempts"] = len(answers)

        if not answers or not any(a.get("solution") for a in answers):
            result["error"] = "All LLM attempts failed"
            result["timings"] = timings
            result["elapsed_s"] = round(time.time() - t_start, 1)
            return result

        best = next((a for a in answers if a.get("solution")), answers[0])
        result["attempts"] = best.get("attempt_count", result["attempts"])
        result["solution"] = best.get("solution", "")
        result["solution_chars"] = len(best.get("solution", ""))
        result["consensus"] = best.get("consensus", False)
        result["model_used"] = best.get("model", "")

        # --- L6: extract FINAL_ANSWER / boxed for display ---
        t6 = time.time()
        fa = self.l6._extract_final_answer(best["solution"])
        if self._looks_intermediate_answer(fa) or self._needs_category_refine(result.get("category", ""), fa):
            refined_fa = self._refine_final_answer(
                prompt=prompt,
                candidate_solution=best.get("solution", ""),
                verbose=verbose,
            )
            if refined_fa:
                fa = refined_fa
        result["sympy_definite_verified"] = False
        if result.get("category") == "definite_integral":
            sympy_fa = self._sympy_definite_integral_from_prompt(prompt)
            if sympy_fa:
                fa = sympy_fa
                result["sympy_definite_verified"] = True
                if verbose:
                    _log.info("  [L6] [SYMPY] deterministic definite integral override")
        result["final_answer"] = fa if fa else "(could not extract)"
        if verbose:
            tag = "[OK]" if fa else "[?]"
            _log.info(f"  [L6] {tag} extracted final line for display")

        # --- optional SUMMARY: block for the UI ---
        result["llm_summary"] = self._extract_llm_summary(best["solution"])

        # Enrich taxonomy keywords with what the LLM itself used/named.
        # Problem text from handwritten PDFs is often too sparse for regex
        # heuristics (e.g. "Find ∫ 3x² - 4x + 2/x dx" has no keyword), but the
        # solution and the model summary almost always spell out the technique.
        tax = result.get("taxonomy") or {}
        if tax.get("topic") and tax.get("subtopic"):
            summary = result.get("llm_summary") or {}
            summary_text = " \n".join(
                str(v) for v in summary.values() if isinstance(v, str) and v
            )
            enrich_text = "\n".join(
                t for t in (summary_text, best.get("solution", "")) if t
            )
            extra = keywords_for_subtopic(tax["topic"], tax["subtopic"], enrich_text)
            if extra:
                merged = merge_keywords(tax.get("keywords") or [], extra)
                tax["keywords"] = merged[:5]
                result["taxonomy"] = tax
                if verbose:
                    _log.info(f"  [L6] Taxonomy keywords: {', '.join(tax['keywords'])}")

        timings["l6_extract"] = round(time.time() - t6, 4)
        result["timings"] = timings
        result["elapsed_s"] = round(time.time() - t_start, 1)
        # Follow-up context (server-side only; stripped by the web layer).
        result["_prompt"] = prompt
        result["_system_prompt"] = system_prompt

        with self._cache_lock:
            self._result_cache[cache_key] = result
            self._result_cache.move_to_end(cache_key)
            while len(self._result_cache) > self._result_cache_max:
                self._result_cache.popitem(last=False)
        return result

    def ask_followup(self, *, prompt: str, prior_solution: str,
                      system_prompt: str | None, user_query: str) -> dict:
        """Answer a follow-up question against an already-solved problem.

        Uses the primary LLM with a single pass (no consensus) for responsiveness
        and feeds the original statement + prior solution as conversation context.
        """
        note = (user_query or "").strip()
        if not note:
            return {"error": "Empty follow-up query"}

        if self.l5 is None or not getattr(self.l5, "is_available", False):
            return {"error": "LLM solver is not available"}

        follow_prompt = (
            "A mathematics problem was previously solved in this conversation. "
            "Use the original problem statement and the prior solution as context, "
            "then answer the user's follow-up question concisely and precisely. "
            "Do not restate the full solution unless the follow-up asks for it.\n\n"
            "--- ORIGINAL PROBLEM ---\n"
            + (prompt or "(unavailable)")
            + "\n\n--- PRIOR SOLUTION ---\n"
            + (prior_solution or "(unavailable)")
            + "\n\n--- USER FOLLOW-UP ---\n"
            + note
        )
        sys_prompt = system_prompt or None

        t0 = time.time()
        try:
            text = self.l5.solve(follow_prompt, system_prompt=sys_prompt)
        except Exception as e:
            return {"error": str(e)[:200]}

        final_answer = self.l6._extract_final_answer(text) or ""
        elapsed_s = round(time.time() - t0, 1)
        model_used = ""
        if self.l5 is not None:
            model_used = f"{self.l5.provider}/{self.l5.model_name}"
        return {
            "user_query": note,
            "solution": text or "",
            "final_answer": final_answer,
            "elapsed_s": elapsed_s,
            "model_used": model_used,
        }

    def _solve_with_consensus(self, prompt: str, system_prompt: str,
                               max_attempts: int = 2, verbose: bool = True,
                               allow_extra_provider_attempt: bool = True) -> list[dict]:
        """Up to ``max_attempts`` solves, rotating Gemini (primary) and Together (fallback).

        Parses each reply to a numeric value when possible and stops early if two
        attempts agree. Handles 429/503-style errors by backing off and switching provider.
        """
        from latex_parser import parse_latex_to_value

        INTER_ATTEMPT_DELAY = 1
        RETRY_503_DELAY = 5

        solvers = []
        if self.l5_primary.is_available:
            solvers.append(("primary", self.l5_primary))
        if self.l5_fallback.is_available:
            solvers.append(("fallback", self.l5_fallback))

        attempts = []
        answer_counts = {}
        rate_limited = set()
        consecutive_503 = 0

        # Reserve one extra slot per additional provider so fallback gets at least
        # one real attempt when the primary burns all base attempts on 503/429.
        total_attempts = max_attempts + (max(0, len(solvers) - 1) if allow_extra_provider_attempt else 0)
        i = 0
        while i < total_attempts:
            if i > 0 and consecutive_503 == 0:
                time.sleep(INTER_ATTEMPT_DELAY)

            solver = None
            solver_label = ""
            for tag, s in solvers:
                if tag not in rate_limited:
                    solver = s
                    solver_label = f"{s.provider}/{s.model_name}"
                    break
            if solver is None:
                if verbose:
                    _log.info(f"  [L5] All models busy; sleeping {RETRY_503_DELAY}s...")
                time.sleep(RETRY_503_DELAY)
                rate_limited.clear()
                consecutive_503 = 0
                solver = solvers[0][1]
                solver_label = f"{solver.provider}/{solver.model_name}"

            t5 = time.time()
            try:
                solution = solver.solve(prompt, system_prompt=system_prompt)
                consecutive_503 = 0
                fa = self.l6._extract_final_answer(solution)
                elapsed = time.time() - t5

                fa_num = parse_latex_to_value(fa) if fa else None
                fa_key = f"{fa_num:.6f}" if fa_num is not None else (fa if fa else None)

                attempts.append({
                    "solution": solution,
                    "final_answer": fa,
                    "numeric": fa_num,
                    "key": fa_key,
                    "elapsed": elapsed,
                    "model": solver_label,
                    "attempt_count": i + 1,
                })

                if fa_key is not None:
                    answer_counts[fa_key] = answer_counts.get(fa_key, 0) + 1

                if verbose:
                    _log.info(
                        f"  [L5] {solver_label} [{i+1}/{total_attempts}]... "
                        f"-> {fa or '?'} ({elapsed:.1f}s)"
                    )

                if i > 0 and fa_key is not None and answer_counts.get(fa_key, 0) >= 2:
                    if verbose:
                        _log.info(
                            f"  [L5] [OK] Consensus: {fa} ({answer_counts[fa_key]}/{i+1} agree)"
                        )
                    for a in attempts:
                        if a.get("key") == fa_key:
                            a["consensus"] = True
                            a["attempt_count"] = i + 1
                    return [a for a in attempts if a.get("key") == fa_key][:1]

            except Exception as e:
                err = str(e)
                if verbose:
                    _log.info(f"[ERR] {err[:80]}")

                is_transient = any(k in err for k in ["503", "UNAVAILABLE", "overloaded", "high demand"])
                is_rate = any(k in err for k in ["429", "RESOURCE_EXHAUSTED", "rate", "quota"])
                is_fatal = any(k in err for k in ["404", "NOT_FOUND"])
                has_any_valid_answer = any(a.get("key") is not None for a in attempts)

                # If we already have at least one parseable answer, avoid long retry
                # loops on a flaky provider; switch faster to fallback.
                if is_transient and has_any_valid_answer:
                    consecutive_503 = 0
                    for tag, s in solvers:
                        if s is solver:
                            rate_limited.add(tag)
                            if verbose:
                                _log.info(f"  [L5] [WARN] {solver_label}: transient 503 after valid answer; switching")
                            if i + 1 >= total_attempts and len(rate_limited) < len(solvers):
                                total_attempts += 1
                            break
                elif is_transient and consecutive_503 < 2:
                    consecutive_503 += 1
                    delay = RETRY_503_DELAY * consecutive_503
                    if verbose:
                        _log.info(f"  [L5] [RETRY] HTTP 503: same model again in {delay}s...")
                    time.sleep(delay)
                elif is_rate or is_fatal or (is_transient and consecutive_503 >= 2):
                    consecutive_503 = 0
                    for tag, s in solvers:
                        if s is solver:
                            rate_limited.add(tag)
                            if verbose:
                                _log.info(f"  [L5] [WARN] {solver_label}: switching to fallback")
                            # If switch happens on the last slot, grant one extra
                            # attempt so the fallback provider actually runs.
                            if i + 1 >= total_attempts and len(rate_limited) < len(solvers):
                                total_attempts += 1
                            break

                attempts.append({"solution": "", "final_answer": "", "error": err})
            finally:
                i += 1

        if not attempts:
            return []

        valid_counts = {k: v for k, v in answer_counts.items() if k is not None}
        if valid_counts:
            best_key = max(valid_counts, key=valid_counts.get)
            best_count = valid_counts[best_key]
            if verbose and best_count < 2:
                _log.info(
                    f"  [L5] [WARN] Weak consensus; best guess: {best_key} "
                    f"({best_count}/{len(attempts)})"
                )
            result = [a for a in attempts if a.get("key") == best_key]
            if result:
                for a in result:
                    a["attempt_count"] = len(attempts)
                return result

        valid = [a for a in attempts if a.get("final_answer")]
        if valid:
            for a in valid:
                a["attempt_count"] = len(attempts)
            return valid[:1]
        valid = [a for a in attempts if a.get("solution")]
        if valid:
            for a in valid:
                a["attempt_count"] = len(attempts)
            return valid[:1]
        for a in attempts:
            a["attempt_count"] = len(attempts)
        return attempts[:1]

    @staticmethod
    def _looks_intermediate_answer(ans: str) -> bool:
        """Heuristic guard: reject likely intermediate/non-final forms."""
        if not ans:
            return True
        t = ans.strip()
        lo = t.lower()

        if t in {"?", "(could not extract)"}:
            return True
        if any(k in lo for k in ["assuming", "let ", "substitute", "set u=", "set u ="]):
            return True
        if t.endswith(":") or "$" in t:
            return True

        identity_markers = ["1-\\sin^2", "1 - \\sin^2", "\\cos^2", "\\tan", "\\cot", "\\sec", "\\csc"]
        if any(m in lo for m in identity_markers):
            if not any(k in lo for k in ["\\ln", "log", "+ c", "\\int", "\\frac"]):
                return True

        if len(t) < 18 and "+ c" not in lo and "\\ln" not in lo and "\\frac" not in lo:
            return True
        return False

    def _refine_final_answer(self, prompt: str, candidate_solution: str, verbose: bool = True) -> str:
        """Ask one extra strict pass only when current final looks intermediate."""
        refine_timeout_s = 18
        strict_system = (
            "Return ONLY the final mathematical result in closed form. "
            "Do not return intermediate identities or substitutions. "
            "If indefinite integral, include + C. "
            "Output exactly one \\\\boxed{...} answer."
        )
        strict_user = (
            "Original problem:\n"
            f"{prompt}\n\n"
            "Candidate answer (possibly intermediate):\n"
            f"{candidate_solution}\n\n"
            "Now provide only the final closed-form answer."
        )

        for solver in (self.l5_primary, self.l5_fallback):
            if not solver.is_available:
                continue
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(solver.solve, strict_user, strict_system)
                    refined_solution = fut.result(timeout=refine_timeout_s)
                refined_fa = self.l6._extract_final_answer(refined_solution) or ""
                if refined_fa and not self._looks_intermediate_answer(refined_fa):
                    if verbose:
                        _log.info(f"  [L6] [REFINE] final upgraded via {solver.provider}/{solver.model_name}")
                    return refined_fa
            except FuturesTimeoutError:
                if verbose:
                    _log.info(f"  [L6] [REFINE-TIMEOUT] {solver.provider}/{solver.model_name} > {refine_timeout_s}s")
            except Exception as e:
                if verbose:
                    _log.info(f"  [L6] [REFINE-ERR] {str(e)[:80]}")
        return ""

    @staticmethod
    def _needs_category_refine(category: str, ans: str) -> bool:
        """Domain-aware guardrails for obvious final-form mismatches."""
        a = (ans or "").lower()
        c = (category or "").lower()
        if c == "definite_integral" and "+ c" in a:
            return True
        return False

    @staticmethod
    def _sympy_definite_integral_from_prompt(prompt: str) -> str:
        """Try deterministic solving for simple one-variable definite integrals."""
        from latex_parser import parse_latex_to_expr, find_matching_brace

        text = (prompt or "").replace("$", " ").strip()
        idx = text.find("\\int_")
        if idx < 0:
            return ""
        i = idx + len("\\int_")
        if i >= len(text) or text[i] != "{":
            return ""
        a_end = find_matching_brace(text, i)
        a_ltx = text[i + 1:a_end]
        j = a_end + 1
        if j >= len(text) or text[j] != "^":
            return ""
        j += 1
        if j < len(text) and text[j] == "{":
            b_end = find_matching_brace(text, j)
            b_ltx = text[j + 1:b_end]
            k = b_end + 1
        else:
            m_b = re.match(r"([^\s\\]+)", text[j:])
            if not m_b:
                return ""
            b_ltx = m_b.group(1)
            k = j + len(b_ltx)

        tail = text[k:]
        m_var = re.search(r"d([a-zA-Z])", tail)
        if not m_var:
            return ""
        integrand_ltx = tail[:m_var.start()].strip()
        var_name = m_var.group(1)
        if not integrand_ltx:
            return ""

        var = sp.Symbol(var_name)
        a_expr = parse_latex_to_expr(a_ltx)
        b_expr = parse_latex_to_expr(b_ltx)
        f_expr = parse_latex_to_expr(integrand_ltx)
        if a_expr is None or b_expr is None or f_expr is None:
            return ""
        try:
            val = sp.simplify(sp.integrate(f_expr, (var, a_expr, b_expr)))
            # Prefer a standard log-style closed form over acosh/asinh when possible.
            val = sp.simplify(val.rewrite(sp.log))
            return sp.latex(val)
        except Exception:
            return ""

    @staticmethod
    def _extract_llm_summary(solution: str) -> dict:
        """Parse the structured SUMMARY: block from the system prompt template (if present)."""
        summary = {}
        idx = solution.find("SUMMARY:")
        if idx == -1:
            return summary
        block = solution[idx:]
        fields = [
            ("problem_type", r"Problem Type:\s*(.+)"),
            ("method_used", r"Method Used:\s*(.+)"),
            ("surface", r"Surface:\s*(.+)"),
            ("key_steps", r"Key Steps:\s*(.+)"),
            ("difficulty", r"Difficulty:\s*(.+)"),
            ("domain", r"Domain:\s*(.+)"),
        ]
        for key, pattern in fields:
            m = re.search(pattern, block)
            if m:
                summary[key] = m.group(1).strip()
        return summary


def solve_single(pdf_path: str, use_nougat: bool = False, use_vlm: bool = True):
    """Run ``STEPSolver`` and emit a human-readable report via the root logger (stdout)."""
    ensure_dirs()
    configure_logging()
    _log.info("")
    _log.info("  " + "=" * 56)
    _log.info("  STEP Pipeline - math problem solver")
    _log.info("  " + "=" * 56)
    _log.info("")
    _log.info(f"  PDF: {pdf_path}")
    _log.info("  " + "-" * 55)

    solver = STEPSolver(use_nougat=use_nougat, use_vlm=use_vlm)
    result = solver.solve(pdf_path, verbose=True)

    _log.info("\n  " + "=" * 55)
    if "error" in result:
        _log.info(f"  ERROR: {result['error']}")
    else:
        # --- PDF Profiling ---
        _log.info("  PDF PROFILING")
        _log.info(f"  {'-'*55}")
        _log.info(f"  File:            {result['file']}")
        _log.info(f"  Pages:           {result.get('pages', '?')}")
        _log.info(f"  Chars:           {result.get('chars', 0)}")
        _log.info(f"  Category:        {result.get('category', 'unknown')}")
        sec = result.get("secondary_categories") or []
        if sec:
            _log.info(f"  Also signals:    {', '.join(sec)}")
        _log.info(f"  Surface type:    {result.get('surface_type', 'unknown')}")
        domain_label = "surface_integral" if result.get('domain') == "surface_integral" else "general_math"
        _log.info(f"  Domain:          {domain_label}")
        ls = result.get("l5_system")
        if ls and isinstance(ls, dict):
            l5d = ls.get("domain", domain_label)
            l5sec = ls.get("secondary_categories") or []
            sec_txt = f" - hints->{', '.join(l5sec)}" if l5sec else ""
            _log.info(f"  LLM system:      {l5d}{sec_txt}")
        kws = result.get('keywords', [])
        if kws:
            _log.info(f"  Keywords:        {', '.join(kws)}")
        summary_text = result.get('summary', '')
        if summary_text:
            if len(summary_text) > 100:
                _log.info(f"  Summary:         {summary_text[:100]}...")
            else:
                _log.info(f"  Summary:         {summary_text}")

        _log.info("\n  ANSWER")
        _log.info(f"  {'-'*55}")
        _log.info(f"  Source:          {result['source']}")
        _log.info(f"  Final:           {result['final_answer']}")
        if result.get("consensus"):
            _log.info(f"  Consensus:       yes ({result.get('attempts', '?')} runs agreed)")
        elif result.get("attempts", 1) > 1:
            _log.info(f"  Consensus:       no ({result.get('attempts', '?')} runs)")

        llm_sum = result.get('llm_summary', {})
        if llm_sum:
            _log.info("\n  MODEL SUMMARY")
            _log.info(f"  {'-'*55}")
            field_labels = {
                "problem_type": "Problem type",
                "method_used": "Method",
                "surface": "Surface",
                "key_steps": "Key steps",
                "difficulty": "Difficulty",
                "domain": "Domain",
            }
            for key, label in field_labels.items():
                val = llm_sum.get(key)
                if val:
                    _log.info(f"  {label+':':<20} {val}")

        _log.info(f"\n  Total time:      {result['elapsed_s']}s")
    _log.info("  " + "=" * 55)
    _log.info("")
    return result


def solve_batch(pdf_dir: str, count: int = None, use_nougat: bool = False, use_vlm: bool = True):
    """Run every ``*.pdf`` in order; append ``pipeline_log_*.json`` under ``RESULTS_DIR`` via ``PipelineLogger``."""
    ensure_dirs()
    configure_logging()
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        _log.info(f"  [!] No PDFs in {pdf_dir}")
        return

    if count:
        pdfs = pdfs[:count]

    _log.info("")
    _log.info("  " + "=" * 44)
    _log.info("  STEP Pipeline - batch run")
    _log.info("  " + "=" * 44)
    _log.info(f"\n  {len(pdfs)} PDF(s) queued")

    solver = STEPSolver(use_nougat=use_nougat, use_vlm=use_vlm)
    logger = PipelineLogger(RESULTS_DIR)
    logger.log_config({"ensemble": False, "nougat_dpi": NOUGAT_DPI,
                        "total_pdfs": len(pdfs), "pdf_dir": str(pdf_dir)})
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.log_environment({
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": gpu_name,
    })

    results = []
    ok_count = 0

    for i, pdf in enumerate(pdfs):
        _log.info(f"\n  --- [{i+1}/{len(pdfs)}] {pdf.name} ---")
        logger.start_pdf(pdf.name)
        r = solver.solve(pdf, verbose=True)
        results.append(r)

        fin = "ok" if not r.get("error") else "error"
        if fin == "ok":
            ok_count += 1
        logger.finish_pdf(fin)

    log_path = logger.save()
    logger.print_summary()

    _log.info("\n  " + "=" * 50)
    _log.info(f"  BATCH: {ok_count}/{len(pdfs)} completed without error")
    _log.info(f"  Log: {log_path}")
    _log.info("  " + "=" * 50 + "\n")
    return results


def check_system():
    """Log GPU, API keys, VLM status, and PDF folder count (``python run.py --check``)."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ensure_dirs()
    configure_logging()

    _log.info("")
    _log.info("  " + "=" * 44)
    _log.info("  STEP Pipeline - health check")
    _log.info("  " + "=" * 44)
    _log.info("")

    # GPU
    if torch.cuda.is_available():
        _log.info(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
        _log.info(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        _log.info("  [!] No CUDA GPU (Nougat is slow on CPU)")

    # API Keys
    from config import TOGETHER_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
    keys = [("Together", TOGETHER_API_KEY), ("Gemini", GEMINI_API_KEY),
            ("Anthropic", ANTHROPIC_API_KEY), ("OpenAI", OPENAI_API_KEY)]
    for name, key in keys:
        status = "OK" if key else "NO"
        _log.info(f"  [{status}] {name} API key: {'...'+key[-8:] if key else 'missing'}")

    # VLM
    vlm = Layer3_VLM()
    vlm_st = "OK" if vlm.is_available else "NO"
    vlm_name = f"{vlm.provider}/{vlm.model}" if vlm.is_available else "disabled"
    _log.info(f"  [{vlm_st}] VLM: {vlm_name}")

    # PDF count
    pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    pdf_st = "OK" if pdf_count else "!"
    _log.info(f"  [{pdf_st}] PDF folder: {pdf_count} file(s) ({PDF_DIR})")
    _log.info("")

    # Hints after --check (also shown in Turkish for local users).
    _log.info("  " + "-" * 44)
    _log.info("  Sonraki adimlar / Next steps")
    _log.info("  " + "-" * 44)
    if pdf_count == 0:
        _log.info("  [!] Cozecek PDF yok. Ornek dosyalari su klasore koyun:")
        _log.info(f"      {PDF_DIR.resolve()}")
        _log.info("")
    _log.info("  Tek PDF:     python run.py <dosya.pdf>")
    _log.info("  Klasor:      python run.py Surface_Integration/ -n 5")
    _log.info("  Varsayilan:  Nougat kapali (VLM + raw)")
    _log.info("  Nougat ac:   python run.py dosya.pdf --with-nougat")
    _log.info("  Web arayuz:  python web_app.py")
    _log.info("               -> tarayici: http://127.0.0.1:5000")
    _log.info("  Eski batch:  python main.py -n 5")
    _log.info("  (Windows'ta garip karakterler icin: $env:PYTHONIOENCODING=\"utf-8\")")
    _log.info("")


def main():
    parser = argparse.ArgumentParser(
        description="STEP - surface-integral PDF solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to a .pdf file or a folder of PDFs")
    parser.add_argument("-n", "--count", type=int, default=None,
                        help="Folder mode: max number of PDFs to process")
    parser.add_argument("--with-nougat", action="store_true",
                        help="Enable Nougat OCR (default is off)")
    parser.add_argument("--no-nougat", action="store_true",
                        help="Deprecated alias; Nougat is already off by default")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip the vision model (Nougat-only if enabled, else raw fallback)")
    parser.add_argument("--check", action="store_true",
                        help="Print GPU / API key / VLM status")

    args = parser.parse_args()

    if args.check:
        check_system()
        return

    if args.input is None:
        parser.print_help()
        ensure_dirs()
        configure_logging()
        _log.info("\nExamples:")
        _log.info("  python run.py problem.pdf")
        _log.info("  python run.py Surface_Integration/si1.pdf")
        _log.info("  python run.py Surface_Integration/")
        _log.info("  python run.py Surface_Integration/ -n 10")
        _log.info("  python run.py my_problem.pdf            # default: no Nougat")
        _log.info("  python run.py my_problem.pdf --with-nougat")
        _log.info("  python run.py my_problem.pdf --no-vlm")
        _log.info("  python run.py --check")
        return

    input_path = Path(args.input)
    use_nougat = args.with_nougat and not args.no_nougat
    use_vlm = not args.no_vlm

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        solve_single(str(input_path), use_nougat=use_nougat, use_vlm=use_vlm)
    elif input_path.is_dir():
        solve_batch(str(input_path), count=args.count, use_nougat=use_nougat, use_vlm=use_vlm)
    else:
        ensure_dirs()
        configure_logging()
        _log.info(f"\n  [!] Invalid path: {input_path}")
        _log.info("      Pass a .pdf file or a folder that contains PDFs.\n")


if __name__ == "__main__":
    main()
