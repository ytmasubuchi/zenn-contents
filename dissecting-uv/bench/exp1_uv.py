"""実験1: uv sync (streamlit依存) のコールド/ウォームキャッシュ計測。
lock(依存解決)とsync(取得・展開)を分けて計測する。
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_common import UV_CACHE_DIR, UV_PYPROJECT, WORK, append_jsonl, du_bytes, fresh_dir, run_cmd

PROJ_DIR = os.path.join(WORK, "uv_proj")
RESULTS_PATH = "/bench/results/exp1_uv_raw.jsonl"


def clear_cache():
    subprocess.run(["uv", "cache", "clean"], capture_output=True, text=True)
    fresh_dir(UV_CACHE_DIR)


def setup_project():
    fresh_dir(PROJ_DIR)
    with open(os.path.join(PROJ_DIR, "pyproject.toml"), "w") as f:
        f.write(UV_PYPROJECT)


def one_run(condition, run_idx):
    setup_project()
    if condition == "cold":
        clear_cache()
    cache_size_before = du_bytes(UV_CACHE_DIR)

    t_lock, rc_lock, out_lock, err_lock = run_cmd(["uv", "lock"], cwd=PROJ_DIR, timeout=900)
    t_sync, rc_sync, out_sync, err_sync = run_cmd(["uv", "sync"], cwd=PROJ_DIR, timeout=900)

    cache_size_after = du_bytes(UV_CACHE_DIR)
    venv_size = du_bytes(os.path.join(PROJ_DIR, ".venv"))

    record = {
        "tool": "uv",
        "condition": condition,
        "run": run_idx,
        "lock_sec": round(t_lock, 3),
        "install_sec": round(t_sync, 3),
        "total_sec": round(t_lock + t_sync, 3),
        "lock_returncode": rc_lock,
        "install_returncode": rc_sync,
        "cache_size_before_bytes": cache_size_before,
        "cache_size_after_bytes": cache_size_after,
        "venv_size_bytes": venv_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if rc_lock != 0:
        record["lock_stderr_tail"] = err_lock[-2000:]
    if rc_sync != 0:
        record["install_stderr_tail"] = err_sync[-2000:]
    append_jsonl(RESULTS_PATH, record)
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["cold", "warm"], required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    if args.condition == "warm":
        setup_project()
        run_cmd(["uv", "lock"], cwd=PROJ_DIR, timeout=900)
        run_cmd(["uv", "sync"], cwd=PROJ_DIR, timeout=900)
        print("warmup done for uv warm condition")

    for i in range(args.runs):
        rec = one_run(args.condition, i)
        print(
            f"[uv/{args.condition}] run={i} lock_sec={rec['lock_sec']} "
            f"install_sec={rec['install_sec']} rc_lock={rec['lock_returncode']} rc_install={rec['install_returncode']}"
        )
