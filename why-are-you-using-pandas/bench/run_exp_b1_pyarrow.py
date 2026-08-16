"""実験B1の追加条件(pandas, ArrowDtype/float64[pyarrow])専用のドライバ。

既存2条件(pandas / polars)はpyarrow未インストールのpandas-benchイメージで
計測するが、この条件だけはpyarrowを導入したDockerfile.pyarrowベースの
イメージ(pandas-bench-pyarrow)で実行する必要があるため、run_exp_b1.pyとは
別ドライバに分離している。

計測対象・パラメータ(N系列、reps、gc制御、drop対象列)はrun_exp_b1.py/
exp_b1_single.pyと完全に同一。結果はexp_b1.csvに混ぜず、
results/exp_b1_pandas_pyarrow.json に独立して保存する。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NS = [5_000, 20_000, 50_000, 200_000, 500_000]
LIB = "pandas_pyarrow"


def run(n):
    cmd = [sys.executable, os.path.join(HERE, "exp_b1_single.py"), "--lib", LIB, "--n", str(n), "--reps", "7"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    rows = []
    for n in NS:
        data = run(n)
        rows.append(data)
        print(
            f"done: {LIB} n={n} -> median {data['median_sec']*1000:.3f} ms "
            f"dtype={data['dtype_repr']} blocks={data['block_count']}"
        )

    out_path = os.path.join(ROOT, "results", "exp_b1_pandas_pyarrow.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
