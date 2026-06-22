"""Repository-relative paths used by runtime code."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
SHADER_DIR = REPO_ROOT / "shaders"
GRAPH_DIR = REPO_ROOT / "graphs"
