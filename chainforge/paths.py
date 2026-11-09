"""Resolve the project root for both source checkouts and installed containers."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("CHAINFORGE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd()
    # Running from a checkout or from the Docker WORKDIR (/app) both match.
    if (cwd / "web").is_dir() or (cwd / "chainforge").is_dir() or (cwd / "corpus").is_dir():
        return cwd
    # Editable installs still resolve to the repository; wheels fall back here.
    return Path(__file__).resolve().parent.parent


ROOT = project_root()
