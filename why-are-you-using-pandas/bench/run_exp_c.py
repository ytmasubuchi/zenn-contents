"""実験Cのドライバ: 各条件をサブプロセスとして実行しCSVに集約する"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NS = [5_000, 20_000, 50_000, 200_000, 500_000]
LIBS = ["pandas", "polars_single_chunk", "polars_multi_chunk", "polars_nulls"]


def run(lib, n):
    cmd = [sys.executable, os.path.join(HERE, "exp_c_single.py"), "--lib", lib, "--n", str(n), "--reps", "7"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    rows = []
    for n in NS:
        for lib in LIBS:
            data = run(lib, n)
            rows.append(data)
            print(f"done: {lib} n={n} chunks={data['n_chunks']} -> median {data['median_sec']*1e6:.1f} us")

    out_path = os.path.join(ROOT, "results", "exp_c.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["lib", "n", "reps", "n_chunks", "median_sec", "min_sec", "max_sec"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
