"""
conftest.py — pytest configuration for masis.infra tests.

Registers the asyncio event loop mode (pytest-asyncio auto mode)
and adds the repository root to sys.path so ``masis.*`` imports resolve
without needing an editable install.
"""

from __future__ import annotations

import os
import sys

# Ensure masis package is importable from the repository root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def pytest_configure(config):
    """Register custom pytest markers used in Phase 2 tests."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as requiring the asyncio event loop (pytest-asyncio)"
    )
