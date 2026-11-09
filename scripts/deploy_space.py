"""Deploy the ChainForge container to a Hugging Face Space.

Requires HF_TOKEN (repository write token) and HF_SPACE (e.g. "user/chainforge").
When either is missing the script prints a note and exits successfully, so the
GitHub Actions workflow still publishes the GHCR image.
"""

from __future__ import annotations

import os
import sys

SPACE_README = """---
title: ChainForge
emoji: ⚒️
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# ChainForge

A custom mini-GPT trainer with a visible reasoning workspace.

Train a small transformer on your own documents, inspect a token-budgeted
reasoning scratchpad, and export portable `.cgpt` archives. This Space runs the
same FastAPI app that `chainforge serve` runs locally.

Storage on this Space may be ephemeral: download models you want to keep.
"""

IGNORE = [
    ".git/*",
    ".github/*",
    ".venv/*",
    "data/*",
    "runs/*",
    "tests/*",
    "docs/*",
    "**/__pycache__/*",
    "**/*.pyc",
    "*.tmp",
    ".pytest_cache/*",
]


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    space = os.environ.get("HF_SPACE", "").strip()
    if not token or not space:
        print("HF_TOKEN/HF_SPACE not configured; skipping Space deploy.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed", file=sys.stderr)
        return 1

    print(f"Deploying to Hugging Face Space {space} …")
    api = HfApi(token=token)
    api.create_repo(
        repo_id=space,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=".",
        repo_id=space,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy ChainForge",
    )
    api.upload_file(
        path_or_fileobj=SPACE_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=space,
        repo_type="space",
        commit_message="Add Space metadata",
    )
    print(f"Live at https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
