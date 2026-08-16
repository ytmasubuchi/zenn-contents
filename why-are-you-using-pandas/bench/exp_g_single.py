"""実験Gの単一条件ランナー: exp_g_worker.pyを子プロセスとして1回だけ起動し、
resource.getrusage(RUSAGE_CHILDREN)でそのworker(=このプロセスの唯一の子)の
ピークRSSを取得する。

ピークRSS計測にgetrusage(RUSAGE_CHILDREN)を使う理由:
- Linuxカーネルはwait4()でハーベストした子プロセスの生涯最大RSS(ru_maxrss)を
  返す。psutilによる定期ポーリングと異なり、ポーリング間隔の間に起きた瞬間的な
  ピークを取りこぼす心配がない、確実な方式。
- 注意点: RUSAGE_CHILDRENは「これまでにこのプロセスが回収した全子孫プロセスの
  中の最大値」を返す。同一プロセスが子プロセスを複数回連続起動すると、2回目以降の
  値は前の子の値を引きずってしまい、条件ごとの値を分離できない。
  そのため、このプロセス自身はworkerをちょうど1回しか起動しない設計にしており、
  reps(繰り返し測定)はrun_exp_g.py側でこのスクリプト自体を毎回新規プロセスとして
  起動することで実現している(=各条件・各repが完全に独立したプロセスツリーになる)。
"""
import argparse
import json
import os
import resource
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--condition",
        required=True,
        choices=["pandas_full", "pandas_chunked", "polars_eager", "polars_streaming"],
    )
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    cmd = [
        sys.executable,
        os.path.join(HERE, "exp_g_worker.py"),
        "--condition",
        args.condition,
        "--data-path",
        args.data_path,
        "--chunksize",
        str(args.chunksize),
    ]

    # 念のため実行前の値も見ておく(このプロセスがまだ子を1つも回収していなければ0のはず)。
    before_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    after_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    # Linux上のru_maxrssの単位はキロバイト(macOSはバイトだが、この実験はDocker/Linux
    # 前提で実行するためキロバイト固定で扱う)。
    peak_rss_bytes = max(before_kb, after_kb) * 1024

    worker_result = json.loads(proc.stdout.strip().splitlines()[-1])
    result = {
        "condition": args.condition,
        "chunksize": args.chunksize,
        "peak_rss_bytes": peak_rss_bytes,
        **worker_result,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
