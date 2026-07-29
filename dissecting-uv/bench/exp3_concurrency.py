"""実験3: uv sync のダウンロード並列数 (UV_CONCURRENT_DOWNLOADS) を
1 (直列) と デフォルト(50) で比較する。
lockファイルは事前に一度だけ生成し(計測対象外)、`uv sync` のみを計測することで
「依存解決」ではなく「並列ダウンロード」の寄与を単離する。
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_common import UV_CACHE_DIR, UV_PYPROJECT, WORK, append_jsonl, du_bytes, fresh_dir, run_cmd

PROJ_DIR = os.path.join(WORK, "uv_concurrency_proj")
RESULTS_PATH = "/bench/results/exp3_concurrency_raw.jsonl"


def clear_cache():
    subprocess.run(["uv", "cache", "clean"], capture_output=True, text=True)
    fresh_dir(UV_CACHE_DIR)


def setup_project_with_lock():
    fresh_dir(PROJ_DIR)
    with open(os.path.join(PROJ_DIR, "pyproject.toml"), "w") as f:
        f.write(UV_PYPROJECT)
    run_cmd(["uv", "lock"], cwd=PROJ_DIR, timeout=300)


def one_run(concurrency, run_idx):
    setup_project_with_lock()
    clear_cache()  # 常にコールドキャッシュでダウンロード時間を計測
    fresh_dir(os.path.join(PROJ_DIR, ".venv"))

    env = os.environ.copy()
    if concurrency is not None:
        env["UV_CONCURRENT_DOWNLOADS"] = str(concurrency)
    else:
        env.pop("UV_CONCURRENT_DOWNLOADS", None)

    t_sync, rc, out, err = run_cmd(["uv", "sync"], cwd=PROJ_DIR, env=env, timeout=900)

    record = {
        "tool": "uv",
        "concurrent_downloads": concurrency if concurrency is not None else "default(50)",
        "run": run_idx,
        "sync_sec": round(t_sync, 3),
        "returncode": rc,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if rc != 0:
        record["stderr_tail"] = err[-2000:]
    append_jsonl(RESULTS_PATH, record)
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=None, help="UV_CONCURRENT_DOWNLOADS value; omit for default")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    for i in range(args.runs):
        rec = one_run(args.concurrency, i)
        print(f"[uv concurrency={rec['concurrent_downloads']}] run={i} sync_sec={rec['sync_sec']} rc={rec['returncode']}")
