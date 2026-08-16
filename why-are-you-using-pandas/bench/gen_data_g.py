"""実験G用の入力データ(CSV)をストリーミング生成する。

行数が数千万〜数億規模になることを想定し、一度に全行をメモリ上に構築しない。
gen_chunk_rows行ずつのDataFrameを作ってはto_csv(mode="a")で追記する、を繰り返す。
これにより生成時のピークメモリはgen_chunk_rowsの大きさでほぼ決まり、
n_rowsをいくら増やしても生成プロセス自体のメモリ使用量は増えない。

スキーマ: id(int64, 連番), group(object, "g000"..形式のN種類の低カーディナリティ文字列),
value(float64, 0〜100の一様乱数)。

実験E(bench/gen_exp_e_data.py)と同様、行数ごとにファイルを分けて
why-are-you-using-pandas/data/exp_g/ 配下に置く(このdata/ディレクトリは
.gitignoreでリポジトリ管理対象外)。書き込み中の破損ファイルを他プロセスが
読まないよう、一時ファイルに書いてからos.replace()でアトミックに配置する。
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "exp_g")


def data_path(n_rows: int) -> str:
    return os.path.join(DATA_DIR, f"n_{n_rows}.csv")


def generate(path, n_rows, n_groups, gen_chunk_rows, seed=42, force=False):
    if os.path.exists(path) and not force:
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    idx = 0
    mode = "w"
    header = True

    tmp_path = path + ".tmp"
    while written < n_rows:
        this_chunk = min(gen_chunk_rows, n_rows - written)
        chunk_df = pd.DataFrame(
            {
                "id": np.arange(idx, idx + this_chunk, dtype=np.int64),
                "group": [f"g{g:03d}" for g in rng.integers(0, n_groups, size=this_chunk)],
                "value": rng.random(this_chunk) * 100.0,
            }
        )
        chunk_df.to_csv(tmp_path, mode=mode, header=header, index=False)
        mode = "a"
        header = False
        written += this_chunk
        idx += this_chunk

    os.replace(tmp_path, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--n-groups", type=int, default=200)
    ap.add_argument("--gen-chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="既存ファイルがあっても再生成する")
    args = ap.parse_args()

    path = data_path(args.rows)
    if os.path.exists(path) and not args.force:
        print(f"skip (exists): {path}")
        return
    generate(path, args.rows, args.n_groups, args.gen_chunk_rows, seed=args.seed, force=args.force)
    size_mb = os.path.getsize(path) / 1024**2
    print(f"wrote {path} ({args.rows} rows, {size_mb:.1f} MiB)")


if __name__ == "__main__":
    main()
