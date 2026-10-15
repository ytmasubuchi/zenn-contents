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

このモジュールは個々のプローブのハング(応答が返らない)を検知する
ウォッチドッグを持たない。1つのプローブが戻ってこなければプロセス全体が
そこで止まり、既に完了した分も含めて1つもファイルに書き出されない。
これは意図的な選択である: プロセス内タイムアウト(シグナル等)を足すと
第4のスキップ種別が生まれ、スモークテスト(test_all_probes_smoke.py)や
比較器へ前提として波及するが、実際に起こるコストは「このホストでの
再実行1回」程度に過ぎない。ウォッチドッグは一段外側、Task 13がホスト側
シェルスクリプトで`docker run`を外側の`timeout`コマンドで包む形で持つ。
"""
import argparse
import json
import os
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

    `SkippedProbe.reason` はここで一切加工しない(要約・切り詰め・
    正規化のいずれもしない)。理由文字列は"missing requirement: "/
    "missing env var: "/"raised: "のどの接頭辞であっても、ここを素通りし
    `probes`エントリの`skipped`キーへそのまま入る。ここで何かを変形すると、
    計測ホスト上で実際にクラッシュしたプローブが「設計通りスキップされた」
    かのように見えてしまい、記事がその理由を繰り返してしまう。
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
        ValueError: 登録済みプローブが1つも無い場合、`selected` が
            どの登録済みプローブ名にも前方一致しなかった場合、または
            `repeat` が1未満の場合。前2つは、空の結果を書き出すのが
            共用ホストを占有した上で何も測らずに成功したように見える、
            最悪の結果になるため。最後の1つは、呼び出し側が指定した
            `repeat`を無言で1へ丸めるのが、このプロジェクトが繰り返し
            引き締めてきた「指定した値を無言で変える」のと同じ種類の
            問題であるため(丸めずエラーにする)。
    """
    import reprobench.probes  # noqa: F401  (副作用: 全プローブを登録する)

    if repeat < 1:
        raise ValueError(f"run: repeat must be >= 1, got {repeat!r}")

    if not all_probes():
        raise ValueError(
            "run: no probes are registered (did reprobench.probes fail to import?)"
        )

    names = _select_probe_names(selected)
    if not names:
        raise ValueError(f"run: selection {selected!r} matched no registered probe")

    rounds = [[_run_one(name) for name in names] for _ in range(repeat)]

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
    `results/{host}__{image}__{config}.json` に書き出す。

    出力先に既にファイルが存在する場合は上書きせず拒否する(占有済みの
    共用ホストで得た結果を無言で失うのは、空の結果を書くのと同じ種類の
    事故であるため)。この存在チェックは**プローブの実行(`run()`)より
    前**に行う — 後に置くと、既に結果がある場所へ向けて実測10分を
    使い切ってから書き込みだけ拒否する、という本末転倒になり、保護しよう
    としている資源そのものを二重に消費してしまう。`--out`が無い場合の
    デフォルトパス算出に必要なのは`collect_env()`だけであり、プローブの
    実行は要らない。

    書き出しは一時ファイルへ書いてから`os.replace`で本来の名前へ置き換える
    (途中で強制終了されても、既存ファイルが壊れた(切り詰められた)JSON
    で上書きされないようにするため)。

    選択が空プローブに終わった場合・`repeat`が1未満の場合・出力ファイルが
    既に存在する場合・出力先への書き込みがOS側の理由(権限・ディスク
    フルなど)で失敗した場合は、いずれも標準エラーに理由を一行書いて
    非ゼロ終了する(`SystemExit`)。
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

    # 出力先を、プローブを1つも実行する前に確定・チェックする(モジュール
    # docstring参照)。デフォルトパスの算出だけのために`collect_env()`を
    # ここで呼ぶのは、`run()`内部でも同じものをもう一度集める(=二重に
    # 集める)ことになるが、環境メタデータの収集はプローブの実測そのもの
    # ではなく軽い処理なので、実測前にチェックできることの方を優先する。
    out_path = Path(args.out) if args.out else _default_output_path(collect_env())

    if out_path.exists():
        print(
            f"runner: refusing to overwrite existing output file: {out_path}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        out = run(selected=selected, repeat=args.repeat)
    except ValueError as exc:
        print(f"runner: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        tmp_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        os.replace(tmp_path, out_path)
    except OSError as exc:
        print(f"runner: failed to write output file {out_path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(f"runner: wrote {out_path}")


if __name__ == "__main__":
    main()
