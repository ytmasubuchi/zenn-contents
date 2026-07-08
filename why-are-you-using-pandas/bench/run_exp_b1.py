"""実験B1のドライバ"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NS = [5_000, 20_000, 50_000, 200_000, 500_000]
LIBS = ["pandas", "polars"]


def run(lib, n, threads=0):
    cmd = [sys.executable, os.path.join(HERE, "exp_b1_single.py"), "--lib", lib, "--n", str(n), "--reps", "7"]
    if threads:
        cmd += ["--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    rows = []
    for n in NS:
        for lib in LIBS:
            data = run(lib, n)
            rows.append(data)
            print(f"done: {lib} n={n} -> median {data['median_sec']*1000:.3f} ms")

    # 公平性チェック: polarsをシングルスレッドに固定した場合の1点(n=500_000)
    data_single_thread = run("polars", 500_000, threads=1)
    data_single_thread["lib"] = "polars_1thread"
    rows.append(data_single_thread)
    print(f"done: polars(1thread) n=500000 -> median {data_single_thread['median_sec']*1000:.3f} ms")

    out_path = os.path.join(ROOT, "results", "exp_b1.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["lib", "n", "reps", "threads", "median_sec", "min_sec", "max_sec"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
