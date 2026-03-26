"""
Post-process LLM output: pull one display string for the final answer.

The class name ``Layer6_SymPyVerifier`` is historical; there is no reference-answer check here anymore.
"""

import re

# Prefix to search for in model output: backslash + "boxed" + "{" (length 7).
_BOXED_TAG = r"\boxed{"


class Layer6_SymPyVerifier:
    """Heuristic parsers for \\boxed{}, FINAL_ANSWER:, and common math tail patterns."""

    def __init__(self):
        pass

    @staticmethod
    def _extract_boxed(text: str) -> list[str]:
        """Return inner contents of each ``\\boxed{...}``, handling nested braces."""
        results = []
        start = 0
        while True:
            idx = text.find(_BOXED_TAG, start)
            if idx == -1:
                break
            open_brace = idx + len(_BOXED_TAG) - 1  # the ``{`` right after ``\boxed``
            depth = 0
            i = open_brace
            while i < len(text):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        results.append(text[open_brace + 1:i])
                        break
                i += 1
            start = i + 1
        return results

    @staticmethod
    def _clean_boxed_content(text: str) -> str:
        """Models sometimes chain ``a = b = c``; keep the last segment."""
        if "=" in text:
            parts = text.split("=")
            return parts[-1].strip()
        return text

    def _extract_final_answer(self, llm_solution: str) -> str:
        """Pick one line/expression to show as the answer.

        Order: explicit tag, then ``\\boxed{}``, then phrase patterns, then math in the last third
        of the solution (avoids grabbing intermediate steps from the middle of the work).
        """
        work = llm_solution.split("SUMMARY")[0] if "SUMMARY" in llm_solution else llm_solution

        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", work)
        if fa_match:
            text = fa_match.group(1).strip().strip("$").strip()
            if text:
                return self._clean_boxed_content(text)

        boxed = self._extract_boxed(work)
        if boxed:
            return self._clean_boxed_content(boxed[-1].strip())

        phrase_patterns = [
            r"[Tt]he\s+final\s+answer\s+is[:\s]*\$*([^$\n]+?)\$*\s*$",
            r"[Ff]inal\s+[Aa]nswer[:\s]*\$*([^$\n]+?)\$*\s*$",
            r"\*\*(?:Answer|Result|Final Answer)[:\s]*\*\*\s*\$*([^$\n]+?)\$*\s*$",
        ]
        for pat in phrase_patterns:
            m = re.search(pat, work, re.MULTILINE)
            if m:
                text = m.group(1).strip().strip("$").strip()
                if text and len(text) < 120 and re.search(r'[0-9\\]', text):
                    return self._clean_boxed_content(text)

        is_patterns = [
            r"(?:flux|integral|answer|result|value)\s+(?:is|equals?)\s+\$([^$]+)\$",
            r"(?:flux|integral|answer|result|value)\s+(?:is|equals?)\s+\$\$([^$]+)\$\$",
            r"(?:across|over|of)\s+\$?[^$]*\$?\s+is\s+\$([^$]+)\$",
        ]
        for pat in is_patterns:
            matches = re.findall(pat, work, re.IGNORECASE)
            if matches:
                candidate = matches[-1].strip()
                if len(candidate) > 2 and re.search(r'[0-9\\]', candidate):
                    return self._clean_boxed_content(candidate)

        tail_start = max(0, len(work) - len(work) // 3)
        tail = work[tail_start:]

        display_blocks = re.findall(r'\$\$\s*(.+?)\s*\$\$', tail, re.DOTALL)
        if display_blocks:
            for block in reversed(display_blocks):
                block = block.strip()
                if "boxed" in block:
                    inner = self._extract_boxed(block)
                    if inner:
                        return self._clean_boxed_content(inner[-1].strip())
                cleaned = self._clean_boxed_content(block)
                if 2 < len(cleaned) < 80 and re.search(r'[0-9]', cleaned):
                    if not re.search(r'(\\partial|\\nabla|\\int|\\iint|\\iiint)', cleaned):
                        return cleaned

        inline_blocks = re.findall(r'(?<!\$)\$([^$]+)\$(?!\$)', tail)
        if inline_blocks:
            for block in reversed(inline_blocks):
                block = block.strip()
                if len(block) > 60 or len(block) < 2:
                    continue
                if re.search(r'\\(?:frac|pi|sqrt)', block) or re.search(r'\d', block):
                    if not re.search(r'(\\mathbf|\\text|\\partial|\\nabla)', block):
                        return self._clean_boxed_content(block)

        eq_end = re.findall(r'=\s*\$*([^$=\n]{2,60}?)\$*\s*$', tail, re.MULTILINE)
        if eq_end:
            for candidate in reversed(eq_end):
                candidate = candidate.strip().strip("$").strip()
                if re.search(r'[0-9]', candidate) and len(candidate) < 60:
                    if not re.search(r'\\(?:partial|nabla|int)', candidate):
                        return self._clean_boxed_content(candidate)

        return ""
