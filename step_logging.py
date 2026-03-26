"""
Central logging setup for the STEP pipeline.

Attaches a single ``StreamHandler`` to the *root* logger with a message-only format
(``%(message)s``) so log lines stay identical to the old ``print`` output—important
for ``web_app`` SSE classification (e.g. lines starting with ``[L0]``, ``[L5]``).

Call ``configure_logging()`` once from CLI or web startup. In ``web_app``, call it
*after* replacing ``sys.stdout`` with ``_SmartStdout`` so the handler writes through
the capture wrapper.
"""

from __future__ import annotations

import logging
import sys

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger once (idempotent).

    First call: add one stdout handler with plain message format. Later calls only
    adjust the root level (e.g. if you pass ``logging.DEBUG``).

    Args:
        level: Minimum level for the root logger (default ``INFO``).
    """
    global _configured
    if _configured:
        logging.getLogger().setLevel(level)
        return
    root = logging.getLogger()
    root.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    _configured = True
