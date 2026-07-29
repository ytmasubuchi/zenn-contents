"""実験4: グローバルキャッシュ+ハードリンク(uv) vs コピー(poetry)を実証する。

2つの別プロジェクトに同じパッケージ(numpy)をインストールし、
- uv: venv内のファイルとuvキャッシュ内のファイルのinodeが同一(ハードリンク、nlink>=2)か
- poetry: venv内のファイルとpoetryキャッシュ内のファイルのinodeが別(コピー)か
を stat で確認する。さらに2プロジェクト分のディスク使用量(du)を比較する。
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_common import POETRY_CACHE_DIR, UV_CACHE_DIR, WORK, du_bytes, fresh_dir, run_cmd

PKG = "numpy"
UV_PROJ_A = os.path.join(WORK, "hardlink_uv_a")
UV_PROJ_B = os.path.join(WORK, "hardlink_uv_b")
POETRY_PROJ_A = os.path.join(WORK, "hardlink_poetry_a")
POETRY_PROJ_B = os.path.join(WORK, "hardlink_poetry_b")

UV_PYPROJECT_NUMPY = """[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy"]
"""

POETRY_PYPROJECT_NUMPY = """[tool.poetry]
name = "{name}"
version = "0.1.0"
description = ""
authors = ["bench <bench@example.com>"]
package-mode = false

[tool.poetry.dependencies]
python = "^3.12"
numpy = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""


def stat_info(path):
    st = os.stat(path)
    return {"path": path, "inode": st.st_ino, "nlink": st.st_nlink, "size": st.st_size, "dev": st.st_dev}


def find_largest_file(root, name_hint=".so"):
    """venv/cache内でnumpyの最大のバイナリ(.so)ファイルを探す(比較対象として使う)"""
    best = None
    best_size = -1
    for dirpath, _, filenames in os.walk(root):
        if "numpy" not in dirpath:
            continue
        for fn in filenames:
            if fn.endswith(name_hint):
                p = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > best_size:
                    best_size = sz
                    best = p
    return best


def setup_uv_project(path, name):
    fresh_dir(path)
    with open(os.path.join(path, "pyproject.toml"), "w") as f:
        f.write(UV_PYPROJECT_NUMPY.format(name=name))


def setup_poetry_project(path, name):
    fresh_dir(path)
    with open(os.path.join(path, "pyproject.toml"), "w") as f:
        f.write(POETRY_PYPROJECT_NUMPY.format(name=name))


def run_uv_experiment():
    subprocess.run(["uv", "cache", "clean"], capture_output=True, text=True)
    fresh_dir(UV_CACHE_DIR)

    setup_uv_project(UV_PROJ_A, "hardlink-uv-a")
    setup_uv_project(UV_PROJ_B, "hardlink-uv-b")

    t_a, rc_a, _, err_a = run_cmd(["uv", "sync"], cwd=UV_PROJ_A, timeout=600)
    t_b, rc_b, _, err_b = run_cmd(["uv", "sync"], cwd=UV_PROJ_B, timeout=600)

    venv_a_file = find_largest_file(os.path.join(UV_PROJ_A, ".venv"))
    venv_b_file = find_largest_file(os.path.join(UV_PROJ_B, ".venv"))
    cache_file = find_largest_file(UV_CACHE_DIR)

    stats = {
        "venv_a_file": stat_info(venv_a_file) if venv_a_file else None,
        "venv_b_file": stat_info(venv_b_file) if venv_b_file else None,
        "cache_file": stat_info(cache_file) if cache_file else None,
    }

    same_inode_a_b = (
        stats["venv_a_file"]["inode"] == stats["venv_b_file"]["inode"]
        if venv_a_file and venv_b_file
        else None
    )
    same_inode_a_cache = (
        stats["venv_a_file"]["inode"] == stats["cache_file"]["inode"]
        if venv_a_file and cache_file
        else None
    )

    du_a = du_bytes(os.path.join(UV_PROJ_A, ".venv"))
    du_b = du_bytes(os.path.join(UV_PROJ_B, ".venv"))
    du_combined = du_bytes(WORK)  # 概算(他ディレクトリも含む可能性があるため参考値)
    du_a_b_only = du_bytes(UV_PROJ_A) + du_bytes(UV_PROJ_B)

    result = {
        "tool": "uv",
        "package": PKG,
        "sync_a_sec": round(t_a, 3),
        "sync_b_sec": round(t_b, 3),
        "returncode_a": rc_a,
        "returncode_b": rc_b,
        "stats": stats,
        "venvA_venvB_same_inode": same_inode_a_b,
        "venvA_cache_same_inode": same_inode_a_cache,
        "venv_a_nlink": stats["venv_a_file"]["nlink"] if venv_a_file else None,
        "du_venv_a_bytes": du_a,
        "du_venv_b_bytes": du_b,
        "du_projA_plus_projB_bytes": du_a_b_only,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return result


def run_poetry_experiment():
    list_proc = subprocess.run(["poetry", "cache", "list"], capture_output=True, text=True)
    for line in list_proc.stdout.splitlines():
        name = line.strip()
        if name:
            subprocess.run(["poetry", "cache", "clear", "--all", "-n", name], capture_output=True, text=True)
    fresh_dir(POETRY_CACHE_DIR)
    subprocess.run(["poetry", "config", "virtualenvs.in-project", "true"], capture_output=True, text=True)

    setup_poetry_project(POETRY_PROJ_A, "hardlink-poetry-a")
    setup_poetry_project(POETRY_PROJ_B, "hardlink-poetry-b")

    t_lock_a, rc_lock_a, _, _ = run_cmd(["poetry", "lock"], cwd=POETRY_PROJ_A, timeout=600)
    t_install_a, rc_install_a, _, err_a = run_cmd(["poetry", "install"], cwd=POETRY_PROJ_A, timeout=600)
    t_lock_b, rc_lock_b, _, _ = run_cmd(["poetry", "lock"], cwd=POETRY_PROJ_B, timeout=600)
    t_install_b, rc_install_b, _, err_b = run_cmd(["poetry", "install"], cwd=POETRY_PROJ_B, timeout=600)

    venv_a_file = find_largest_file(os.path.join(POETRY_PROJ_A, ".venv"))
    venv_b_file = find_largest_file(os.path.join(POETRY_PROJ_B, ".venv"))
    cache_file = find_largest_file(POETRY_CACHE_DIR)

    stats = {
        "venv_a_file": stat_info(venv_a_file) if venv_a_file else None,
        "venv_b_file": stat_info(venv_b_file) if venv_b_file else None,
        "cache_file": stat_info(cache_file) if cache_file else None,
    }

    same_inode_a_b = (
        stats["venv_a_file"]["inode"] == stats["venv_b_file"]["inode"]
        if venv_a_file and venv_b_file
        else None
    )
    same_inode_a_cache = (
        stats["venv_a_file"]["inode"] == stats["cache_file"]["inode"]
        if venv_a_file and cache_file
        else None
    )

    du_a = du_bytes(os.path.join(POETRY_PROJ_A, ".venv"))
    du_b = du_bytes(os.path.join(POETRY_PROJ_B, ".venv"))
    du_a_b_only = du_bytes(POETRY_PROJ_A) + du_bytes(POETRY_PROJ_B)

    result = {
        "tool": "poetry",
        "package": PKG,
        "install_a_sec": round(t_install_a, 3),
        "install_b_sec": round(t_install_b, 3),
        "lock_a_sec": round(t_lock_a, 3),
        "lock_b_sec": round(t_lock_b, 3),
        "returncode_install_a": rc_install_a,
        "returncode_install_b": rc_install_b,
        "stats": stats,
        "venvA_venvB_same_inode": same_inode_a_b,
        "venvA_cache_same_inode": same_inode_a_cache,
        "venv_a_nlink": stats["venv_a_file"]["nlink"] if venv_a_file else None,
        "du_venv_a_bytes": du_a,
        "du_venv_b_bytes": du_b,
        "du_projA_plus_projB_bytes": du_a_b_only,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return result


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    out = {}
    if which in ("uv", "both"):
        out["uv"] = run_uv_experiment()
        print(json.dumps(out["uv"], indent=2, ensure_ascii=False))
    if which in ("poetry", "both"):
        out["poetry"] = run_poetry_experiment()
        print(json.dumps(out["poetry"], indent=2, ensure_ascii=False))

    result_path = "/bench/results/exp4_hardlink_result.json"
    existing = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
    existing.update(out)
    with open(result_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"wrote {result_path}")
