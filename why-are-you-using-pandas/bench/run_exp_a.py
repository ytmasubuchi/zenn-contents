"""実験Aのドライバ: 各条件をサブプロセスとして実行しCSVに集約する"""
import csv
import json
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LIBS = ["pandas_object", "pandas_default", "pandas_category", "polars"]
LENGTHS = [5, 10, 20, 50, 100, 200]
N = 200_000


def main():
    rows = []
    for length in LENGTHS:
        for lib in LIBS:
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(HERE, "exp_a_single.py"),
                    "--lib",
                    lib,
                    "--n",
                    str(N),
                    "--length",
                    str(length),
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
                check=True,
            )
            line = proc.stdout.strip().splitlines()[-1]
            data = json.loads(line)
            rows.append(data)
            print(f"done: {lib} L={length} -> {data['api_bytes_per_elem']:.1f} B/elem (API), "
                  f"{data['rss_delta_per_elem']:.1f} B/elem (RSS)")

    out_path = os.path.join(ROOT, "results", "exp_a.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
