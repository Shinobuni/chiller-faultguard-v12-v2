"""Vercel & ASGI root entrypoint for Chiller FaultGuard v12.0 Digital Twin."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app, create_app

__all__ = ["app", "create_app"]
