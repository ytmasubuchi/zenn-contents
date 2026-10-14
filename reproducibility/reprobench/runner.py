"""プローブ実行ランナー: CLIから呼ばれ、選択したプローブを実行して結果を
JSONへ書き出す。

このランナーは計測ホスト上で「設定1件につき1回」しか走らない。共用4
ホスト(MacBook/anakin/blade/dgxspark)の占有は実測10分以内という制約の
もとで動き、出力ファイルだけが計測の痕跡として残る。比較器・図版・記事の
数値は全てこの出力ファイル経由でしか取り出せないため、ここで何かを
静かに落とすと、その情報はもう無く、再計測は共用ホストをもう一度
占有することを意味する。

このモジュールの責務は「プローブの選択・実行・結果の整形と書き出し」の
みであり、プローブそのもの(Task 4以降)・プローブの登録と欠落/例外の
吸収(registry.py)・結果の比較(別タスクの比較器)は扱わない。
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from reprobench.envinfo import collect_env
from reprobench.registry import ProbeResult, SkippedProbe, all_probes, run_probe


def _select_probe_names(selected: list[str] | None) -> list[str]:
    """`all_probes()` が返す名前(ソート済み)のうち、`selected` のいずれか
    の前方一致にかかるものだけを順序を保って返す。`selected` が `None`
    なら全件を返す。

    選択が何にもマッチしない場合は空リストを返す(それをエラーにするか
    どうかは呼び出し側 `run`/`main` の責務とし、ここでは選択の計算のみを
    行う)。
    """
    names = list(all_probes())
    if selected is None:
        return names
    return [name for name in names if any(name.startswith(prefix) for prefix in selected)]


def _run_one(name: str) -> dict:
    """1プローブを実行し、所要時間(秒)を添えたJSON安全な辞書で返す。

    `ProbeResult`/`SkippedProbe` どちらであっても、`registry.run_probe`
    が例外を外へ送出しない契約(docstring参照)に乗っているため、ここで
    追加の例外処理は行わない。プローブ自身の `meta`/`scalars` が
    JSON化できることは各プローブモジュール側の責務(このランナーの
    テストでも本物のペイロードで一度round-tripを確認している)。
    """
    start = time.perf_counter()
    result = run_probe(name)
    wall_time_s = time.perf_counter() - start

    if isinstance(result, SkippedProbe):
        return {"probe": result.probe, "skipped": result.reason, "wall_time_s": wall_time_s}
    assert isinstance(result, ProbeResult)
    return {**asdict(result), "wall_time_s": wall_time_s}


def run(selected: list[str] | None = None, repeat: int = 1) -> dict:
    """選択したプローブ(前方一致、`None`なら全件)を`repeat`回実行し、
    実行環境メタデータと合わせて1つの辞書に詰めて返す。

    `repeat` が1より大きいときは `rounds` キーに各回の `probes` 相当の
    リストを `repeat` 個並べて持たせる(G1-1a: 同一プロセス内の繰り返しで
    プローブが安定するかを見るため)。`probes` キーは後方互換のため常に
    1回目の結果を指す。`repeat` が1のとき(デフォルト)は `rounds` キー
    自体を持たせない。

    Raises:
        ValueError: 登録済みプローブが1つも無い場合、または `selected`
            がどの登録済みプローブ名にも前方一致しなかった場合。空の
            結果を書き出すのは、共用ホストを占有した上で何も測らずに
            成功したように見える、最悪の結果になるため、ここで必ず
            エラーにする。
    """
    import reprobench.probes  # noqa: F401  (副作用: 全プローブを登録する)

    if not all_probes():
        raise ValueError(
            "run: no probes are registered (did reprobench.probes fail to import?)"
        )

    names = _select_probe_names(selected)
    if not names:
        raise ValueError(f"run: selection {selected!r} matched no registered probe")

    rounds = [[_run_one(name) for name in names] for _ in range(max(1, repeat))]

    out = {
        "schema": 1,
        "env": collect_env(),
        "probe_order": names,
        "probes": rounds[0],
    }
    if repeat > 1:
        out["rounds"] = rounds
    return out


def _default_output_path(env: dict) -> Path:
    """`--out` 未指定時の出力先: `results/{host}__{image}__{config}.json`。

    `host`/`image`/`config_label` のいずれかが取得できない(`None`)場合
    でも、ファイル名自体は常に生成できるよう `"unknown-*"` で埋める。
    """
    host = env.get("host") or "unknown-host"
    image = env.get("image") or "unknown-image"
    config = env.get("config_label") or "unknown-config"
    return Path("results") / f"{host}__{image}__{config}.json"


def main(argv: list[str] | None = None) -> None:
    """CLIエントリポイント。

    `--probes S1,S2` のようにカンマ区切りで複数の前方一致セレクタを指定
    できる(省略時は全プローブ)。`--out` 省略時は
    `results/{host}__{image}__{config}.json` に書き出す。出力先ディレクトリ
    が無ければ作成するが、**出力先に既にファイルが存在する場合は上書き
    せず拒否する**(占有済みの共用ホストで得た結果を無言で失うのは、
    空の結果を書くのと同じ種類の事故であるため)。

    選択が空プローブに終わった場合・出力ファイルが既に存在する場合は、
    標準エラーに理由を書いて非ゼロ終了する(`SystemExit`)。
    """
    parser = argparse.ArgumentParser(description="Run reprobench probes and record results.")
    parser.add_argument(
        "--probes",
        default=None,
        help="カンマ区切りの前方一致セレクタ(例: S1,S2)。省略時は全プローブ。",
    )
    parser.add_argument("--out", default=None, help="出力先JSONパス。省略時はresults/配下。")
    parser.add_argument(
        "--repeat", type=int, default=1, help="同一プロセス内でプローブ群をN回実行する。"
    )
    args = parser.parse_args(argv)

    # 空文字列トークン("S1,"のような末尾カンマ)を素通りさせると、
    # 全ての名前が""に前方一致してしまい、「一部だけ選んだつもり」が
    # 「全件選択」に化ける。ここで取り除き、素通りさせない。
    selected = [p.strip() for p in args.probes.split(",") if p.strip()] if args.probes else None

    try:
        out = run(selected=selected, repeat=args.repeat)
    except ValueError as exc:
        print(f"runner: {exc}", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(args.out) if args.out else _default_output_path(out["env"])

    if out_path.exists():
        print(
            f"runner: refusing to overwrite existing output file: {out_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"runner: wrote {out_path}")


if __name__ == "__main__":
    main()
