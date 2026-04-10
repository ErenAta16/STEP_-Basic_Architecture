"""
Layer 5 — LLM solver (Together, Gemini, Claude, or OpenAI-compatible).

Picks the first provider that has a working API key unless ``force_provider`` is set.
Initialization messages go through the standard ``logging`` logger for this module.
"""

import logging

from config import (
    TOGETHER_API_KEY,
    TOGETHER_MODEL,
    TOGETHER_BASE_URL,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_FALLBACK_MODEL,
    CLAUDE_MODEL,
    GPT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_SYSTEM_PROMPT,
)

_log = logging.getLogger(__name__)


class Layer5_LLMSolver:
    """Thin wrapper: send the built prompt to one LLM and return raw text."""

    def __init__(self, force_provider: str | None = None):
        self.provider = None
        self.client = None
        self.model_name = None
        self.gemini_fallback_model = None
        self._setup(force_provider)

    def _setup(self, force_provider: str | None = None):
        """Wire up one client. If `force_provider` is set, only that backend is tried."""
        providers = {
            "together": self._init_together,
            "gemini": self._init_gemini,
            "claude": self._init_claude,
            "openai": self._init_openai,
        }

        if force_provider:
            if force_provider in providers:
                providers[force_provider]()
            else:
                _log.info(f"  [!] Unknown LLM provider: {force_provider}")
            return

        for name, init_fn in providers.items():
            if init_fn():
                return
        _log.info("  [!] No LLM API key found in the environment.")

    def _init_together(self) -> bool:
        if not TOGETHER_API_KEY:
            return False
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)
            self.provider = "together"
            self.model_name = TOGETHER_MODEL
            _log.info(f"  LLM: Together ({TOGETHER_MODEL})")
            return True
        except ImportError:
            _log.info("  [!] `openai` package missing. pip install openai")
            return False

    def _init_gemini(self) -> bool:
        if not GEMINI_API_KEY:
            return False
        try:
            from google import genai
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.provider = "gemini"
            self.model_name = GEMINI_MODEL
            self.gemini_fallback_model = GEMINI_FALLBACK_MODEL
            _log.info(f"  LLM: Gemini ({GEMINI_MODEL})")
            return True
        except ImportError:
            _log.info("  [!] google-genai missing. pip install google-genai")
            return False

    def _init_claude(self) -> bool:
        if not ANTHROPIC_API_KEY:
            return False
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.provider = "claude"
            self.model_name = CLAUDE_MODEL
            _log.info(f"  LLM: Claude ({CLAUDE_MODEL})")
            return True
        except ImportError:
            _log.info("  [!] anthropic package missing. pip install anthropic")
            return False

    def _init_openai(self) -> bool:
        if not OPENAI_API_KEY:
            return False
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            self.model_name = GPT_MODEL
            _log.info(f"  LLM: GPT-4o ({GPT_MODEL})")
            return True
        except ImportError:
            _log.info("  [!] `openai` package missing. pip install openai")
            return False

    @property
    def is_available(self) -> bool:
        return self.client is not None

    _BOXED_FOLLOWUP = (
        "Based on the following solution, state ONLY the final answer "
        "inside \\boxed{{}}. Nothing else.\n\nSolution:\n{tail}"
    )

    def solve(self, problem_latex: str, system_prompt: str | None = None) -> str:
        """Run the model on `problem_latex`; optional `system_prompt` overrides the default.
        If the answer lacks \\boxed{}, a cheap follow-up asks for one (all providers)."""
        if not self.is_available:
            raise RuntimeError("No LLM client configured. Check your API keys in .env")

        prompt = system_prompt or LLM_SYSTEM_PROMPT

        if self.provider in ("together", "openai"):
            text = self._solve_openai_compat(problem_latex, prompt)
        elif self.provider == "gemini":
            text = self._solve_gemini(problem_latex, prompt)
        elif self.provider == "claude":
            text = self._solve_claude(problem_latex, prompt)
        else:
            raise RuntimeError(f"Unknown LLM provider: {self.provider}")

        if "\\boxed" not in text:
            text = self._boxed_followup(text)
        return text

    def _boxed_followup(self, text: str) -> str:
        """Cheap second call: ask the same provider to emit \\boxed{} only."""
        followup_prompt = self._BOXED_FOLLOWUP.format(tail=text[-1500:])
        try:
            if self.provider in ("together", "openai"):
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": followup_prompt}],
                )
                boxed = resp.choices[0].message.content or ""
            elif self.provider == "gemini":
                from google.genai import types
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=followup_prompt,
                    config=types.GenerateContentConfig(max_output_tokens=256, temperature=0.0),
                )
                boxed = resp.text or ""
            elif self.provider == "claude":
                resp = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=256,
                    temperature=0.0,
                    messages=[{"role": "user", "content": followup_prompt}],
                )
                boxed = resp.content[0].text or ""
            else:
                return text

            boxed = boxed.strip()
            if "\\boxed" in boxed:
                return text + "\n\n" + boxed
        except Exception:
            pass
        return text

    def _solve_openai_compat(self, problem_latex: str, system_prompt: str) -> str:
        """Chat completions API (shared by Together and OpenAI)."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem_latex},
            ],
        )
        return response.choices[0].message.content or ""

    def _solve_gemini_once(self, model_name: str, problem_latex: str, system_prompt: str) -> str:
        """Single Gemini call with an explicit model name."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=model_name,
            contents=problem_latex,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            ),
        )
        return response.text or ""

    def _solve_gemini(self, problem_latex: str, system_prompt: str) -> str:
        """Gemini generate_content (follow-up is handled centrally in `solve`)."""
        try:
            return self._solve_gemini_once(self.model_name, problem_latex, system_prompt)
        except Exception:
            fallback = (self.gemini_fallback_model or "").strip()
            if fallback and fallback != self.model_name:
                _log.warning(
                    f"  [L5] Gemini primary '{self.model_name}' failed; retrying with '{fallback}'"
                )
                return self._solve_gemini_once(fallback, problem_latex, system_prompt)
            raise

    def _solve_claude(self, problem_latex: str, system_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": problem_latex,
            }],
        )
        return response.content[0].text
