"""実験Dのドライバ: 各条件をサブプロセスとして実行しCSVに集約する"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NS = [5_000, 20_000, 50_000, 200_000, 500_000]
CONDITIONS = [
    ("numeric", False),
    ("numeric", True),
    ("string", False),
    ("string", True),
]


def run(cols, use_pyarrow_ext, n):
    cmd = [
        sys.executable,
        os.path.join(HERE, "exp_d_single.py"),
        "--cols",
        cols,
        "--n",
        str(n),
        "--reps",
        "7",
    ]
    if use_pyarrow_ext:
        cmd.append("--use_pyarrow_ext")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    rows = []
    for n in NS:
        for cols, use_pyarrow_ext in CONDITIONS:
            data = run(cols, use_pyarrow_ext, n)
            rows.append(data)
            print(
                f"done: cols={cols} use_pyarrow_ext={use_pyarrow_ext} n={n} "
                f"-> median {data['median_sec']*1000:.3f} ms"
            )

    out_path = os.path.join(ROOT, "results", "exp_d.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["cols", "use_pyarrow_ext", "n", "length", "reps", "median_sec", "min_sec", "max_sec"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
