"""実験Aの追加条件(pandas 3.0 default, pyarrowインストール環境)専用のドライバ。

既存4条件(pandas_object / pandas_default / pandas_category / polars)は
pyarrow未インストールのpandas-benchイメージで計測するが、この条件だけは
pyarrowを導入したDockerfile.pyarrowベースのイメージ(pandas-bench-pyarrow)で
実行する必要があるため、run_exp_a.pyとは別ドライバに分離している。

計測対象・パラメータ(N, 文字列長の刻み、一意ASCII文字列の生成方法)は
run_exp_a.pyと完全に同一。結果はexp_a.csvに混ぜず、
results/exp_a_pandas_pyarrow.json に独立して保存する。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LENGTHS = [5, 10, 20, 50, 100, 200]
N = 200_000
LIB = "pandas_pyarrow"


def main():
    rows = []
    for length in LENGTHS:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(HERE, "exp_a_single.py"),
                "--lib",
                LIB,
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
        print(
            f"done: {LIB} L={length} -> {data['api_bytes_per_elem']:.1f} B/elem (API), "
            f"{data['rss_delta_per_elem']:.1f} B/elem (RSS), "
            f"dtype={data['dtype_repr']} storage={data['storage_repr']}"
        )

    out_path = os.path.join(ROOT, "results", "exp_a_pandas_pyarrow.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
