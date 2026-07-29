"""実験1: pip install streamlit のコールド/ウォームキャッシュ計測"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_common import PIP_CACHE_DIR, WORK, append_jsonl, du_bytes, fresh_dir, run_cmd

VENV_DIR = os.path.join(WORK, "pip_venv")
RESULTS_PATH = "/bench/results/exp1_pip_raw.jsonl"


def clear_cache():
    subprocess.run(["pip", "cache", "purge"], capture_output=True, text=True)
    fresh_dir(PIP_CACHE_DIR)


def one_run(condition, run_idx):
    fresh_dir(WORK)
    if condition == "cold":
        clear_cache()
    cache_size_before = du_bytes(PIP_CACHE_DIR)

    t_venv, rc_venv, _, err_venv = run_cmd(["python3", "-m", "venv", VENV_DIR])
    pip_bin = os.path.join(VENV_DIR, "bin", "pip")

    t_install, rc, out, err = run_cmd(
        [pip_bin, "install", "streamlit"], timeout=900
    )

    cache_size_after = du_bytes(PIP_CACHE_DIR)
    venv_size = du_bytes(VENV_DIR)

    record = {
        "tool": "pip",
        "condition": condition,
        "run": run_idx,
        "venv_create_sec": round(t_venv, 3),
        "install_sec": round(t_install, 3),
        "total_sec": round(t_venv + t_install, 3),
        "returncode": rc,
        "cache_size_before_bytes": cache_size_before,
        "cache_size_after_bytes": cache_size_after,
        "venv_size_bytes": venv_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if rc != 0:
        record["stderr_tail"] = err[-2000:]
    append_jsonl(RESULTS_PATH, record)
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["cold", "warm"], required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    if args.condition == "warm":
        # ウォームアップ: キャッシュを温める(計測しない)
        fresh_dir(WORK)
        t, rc, out, err = run_cmd(["python3", "-m", "venv", VENV_DIR])
        pip_bin = os.path.join(VENV_DIR, "bin", "pip")
        run_cmd([pip_bin, "install", "streamlit"], timeout=900)
        print("warmup done for pip warm condition")

    for i in range(args.runs):
        rec = one_run(args.condition, i)
        print(f"[pip/{args.condition}] run={i} install_sec={rec['install_sec']} rc={rec['returncode']}")
