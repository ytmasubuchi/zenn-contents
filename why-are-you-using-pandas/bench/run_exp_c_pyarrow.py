"""実験Cの追加条件(pandas, ArrowDtype/float64[pyarrow])専用のドライバ。

既存4条件(pandas / polars_single_chunk / polars_multi_chunk / polars_nulls)は
pyarrow未インストールのpandas-benchイメージで計測するが、この条件だけは
pyarrowを導入したDockerfile.pyarrowベースのイメージ(pandas-bench-pyarrow)で
実行する必要があるため、run_exp_c.pyとは別ドライバに分離している。

計測対象・パラメータ(N系列、reps、gc制御)はrun_exp_c.py/exp_c_single.pyと
完全に同一。結果はexp_c.csvに混ぜず、results/exp_c_pandas_pyarrow.json に
独立して保存する。
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
    cmd = [sys.executable, os.path.join(HERE, "exp_c_single.py"), "--lib", LIB, "--n", str(n), "--reps", "7"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    rows = []
    for n in NS:
        data = run(n)
        rows.append(data)
        print(
            f"done: {LIB} n={n} dtype={data['dtype_repr']} -> median {data['median_sec']*1e6:.1f} us"
        )

    out_path = os.path.join(ROOT, "results", "exp_c_pandas_pyarrow.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
