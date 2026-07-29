"""実験1(インストール速度)の共通ユーティリティ"""
import json
import os
import shutil
import statistics
import subprocess
import time

WORK = "/bench/work"
PIP_CACHE_DIR = os.environ.get("PIP_CACHE_DIR", "/bench/.cache/pip")
POETRY_CACHE_DIR = os.environ.get("POETRY_CACHE_DIR", "/bench/.cache/pypoetry")
UV_CACHE_DIR = os.environ.get("UV_CACHE_DIR", "/bench/.cache/uv")

POETRY_PYPROJECT = """[tool.poetry]
name = "bench-poetry"
version = "0.1.0"
description = ""
authors = ["bench <bench@example.com>"]
package-mode = false

[tool.poetry.dependencies]
python = "^3.12"
streamlit = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""

UV_PYPROJECT = """[project]
name = "bench-uv"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["streamlit"]
"""


def run_cmd(cmd, cwd=None, env=None, timeout=600):
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.perf_counter() - t0
    return elapsed, proc.returncode, proc.stdout, proc.stderr


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def du_bytes(path):
    if not os.path.exists(path):
        return 0
    out = subprocess.run(["du", "-sb", path], capture_output=True, text=True)
    try:
        return int(out.stdout.split()[0])
    except Exception:
        return -1


def median(values):
    return statistics.median(values) if values else None


def fresh_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
