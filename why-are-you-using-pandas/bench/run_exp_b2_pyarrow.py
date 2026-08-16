"""実験B2の追加条件(pandas, ArrowDtype/float64[pyarrow])専用のドライバ。

既存2条件(pandas / polars)はpyarrow未インストールのpandas-benchイメージで
計測するが、この条件だけはpyarrowを導入したDockerfile.pyarrowベースの
イメージ(pandas-bench-pyarrow)で実行する必要があるため、run_exp_b2.pyとは
別ドライバに分離している。

計測対象・パラメータ(N、start-cols、add-cols、reps)はrun_exp_b2.py/
exp_b2_single.pyと完全に同一。結果はexp_b2_pandas.json / exp_b2_polars.jsonに
混ぜず、results/exp_b2_pandas_pyarrow.json に独立して保存する。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LIB = "pandas_pyarrow"


def run():
    cmd = [sys.executable, os.path.join(HERE, "exp_b2_single.py"), "--lib", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    data = run()
    out_path = os.path.join(ROOT, "results", "exp_b2_pandas_pyarrow.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"done: {LIB} -> loop_total={data['loop_add_time_total_sec']*1000:.1f}ms "
          f"batch={data['batch_add_time_sec']*1000:.1f}ms "
          f"warning_seen={data.get('fragmentation_warning_seen')} "
          f"dtype={data.get('dtype_repr')} blocks={data.get('n_blocks_after')} "
          f"blocks_after_copy={data.get('n_blocks_after_copy')}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
