"""reprobench.runner のテスト: プローブ実行CLIの契約を検証する。

このランナーは計測ホスト上で「設定1件につき1回」しか走らず、出力ファイル
だけが生き残る。比較器・図版・記事の数値は全てこのファイル経由でしか
取り出せないため、ここでのテストは「何かを静かに落とさないこと」
(空の結果、落ちたプローブの欠落、JSON化できないフィールド、実行順序の
不記載)を優先して確認する。
"""
import json

import pytest

import reprobench.probes  # noqa: F401  (副作用: 全プローブを登録する)
from reprobench import registry, runner
from reprobench.registry import all_probes
from reprobench.runner import main, run

# --- タスクブリーフ記載のテスト(そのまま) ---


def test_run_includes_env_and_probes():
    out = run(selected=["S1_stdlib_mt19937"])
    assert out["schema"] == 1 and "env" in out and len(out["probes"]) == 1


def test_run_prefix_selection_matches_group():
    out = run(selected=["S1"])
    assert all(p["probe"].startswith("S1") for p in out["probes"]) and len(out["probes"]) >= 4


def test_skipped_probes_are_recorded_not_dropped(monkeypatch):
    monkeypatch.setattr(registry, "check_requirement", lambda r: False)
    out = run(selected=["S8"])
    assert out["probes"] and all("skipped" in p for p in out["probes"])


def test_repeat_produces_independent_rounds():
    out = run(selected=["S0_parallel_completion_order"], repeat=2)
    assert len(out["rounds"]) == 2


# --- 解決事項に対する追加テスト ---


def test_empty_selection_raises_instead_of_producing_empty_file():
    # 解決事項2: 空の選択で空の結果ファイルを書いてしまうのが、共用ホスト
    # 上でもっとも悪い結果になる。マッチしないセレクタは例外にする。
    with pytest.raises(ValueError):
        run(selected=["Z_no_such_probe_prefix"])


def test_no_registered_probes_raises(monkeypatch):
    # 解決事項2: all_probes()が空(=import漏れ等)のときも同様にエラーに
    # する。reprobench.runner が束縛している名前そのものを差し替える
    # (registry.all_probesを差し替えても、runner.py内のグローバル参照には
    # 影響しないため)。
    import reprobench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "all_probes", lambda: {})
    with pytest.raises(ValueError):
        run(selected=None)


def test_skip_reason_is_recorded_verbatim_when_forced(monkeypatch):
    # 解決事項3(fix round 1 finding 3): 「skippedならチェックする」という
    # if分岐の内側に理由文字列の一致を置くと、実行環境にたまたまtorchが
    # 入っているだけでテストの本体が一度も走らず、何も確かめずに
    # 合格してしまう。ここでは`check_requirement`を強制的にFalseへ
    # 差し替えてスキップを無条件に起こし、`_run_one`が理由文字列を一切
    # 加工(要約・切り詰め・正規化)せずそのまま出力へ通すことを確認する。
    # これが崩れると、計測ホスト上で実際にクラッシュしたプローブ
    # ("raised: ...")が「設計通りスキップ」に見えかねない。
    monkeypatch.setattr(registry, "check_requirement", lambda r: False)
    out = run(selected=["S7_mlp_train10"])
    entry = out["probes"][0]
    assert entry["probe"] == "S7_mlp_train10"
    assert entry["skipped"] == "missing requirement: torch"
    assert isinstance(entry["wall_time_s"], float)


def test_full_run_payload_is_json_round_trip_safe():
    # 解決事項4: このプロジェクトは「整数キーの辞書」「ライブラリ製config
    # 辞書の無検証な埋め込み」の2回、JSON化できないペイロードを出荷した
    # 実績がある。合成データではなく、実際に`run()`がデフォルト(全プローブ)
    # で返す本物のペイロード(成功・スキップ両方を含む、CLIを`--probes`
    # 無しで叩いたときと同じもの)でround-tripを確認する。
    out = run(selected=None)
    assert any("skipped" in p for p in out["probes"]), "この選択にスキップが無ければ想定と違う"
    assert any("skipped" not in p for p in out["probes"]), "この選択に成功が無ければ想定と違う"
    round_tripped = json.loads(json.dumps(out))
    assert round_tripped == out


def test_each_probe_entry_records_wall_time():
    # 解決事項5: 10分予算は共用ホストで実測されるため、1プローブごとの
    # 所要時間が無いと遅いホストを後から診断できない。
    out = run(selected=["S1"])
    for entry in out["probes"]:
        assert "wall_time_s" in entry
        assert isinstance(entry["wall_time_s"], float)
        assert entry["wall_time_s"] >= 0.0


def test_probe_execution_order_is_recorded():
    # 解決事項6: all_probes()は名前順を返すが、「実際に何を何順で走らせた
    # か」を出力自体に書いておかないと、後から推測するしかなくなる。
    out = run(selected=["S1"])
    assert out["probe_order"] == [p["probe"] for p in out["probes"]]
    assert out["probe_order"] == sorted(out["probe_order"])


def test_repeat_one_has_no_rounds_key():
    # repeatが1のとき(デフォルト)は、後方互換のための`rounds`キーを
    # わざわざ持たせない(ブリーフ: 「repeatが1より大きいときはrounds
    # キーに...」)。
    out = run(selected=["S1_stdlib_mt19937"])
    assert "rounds" not in out


def test_repeat_keeps_probes_as_first_round_for_backward_compat():
    out = run(selected=["S1_stdlib_mt19937"], repeat=3)
    assert out["probes"] == out["rounds"][0]
    assert len(out["rounds"]) == 3


def test_run_rejects_repeat_less_than_one():
    # fix round 1 small finding: 呼び出し側が指定した`repeat`を無言で1へ
    # 丸めるのは、このプロジェクトが繰り返し引き締めてきた「指定した値を
    # 無言で変える」のと同じ種類の問題なので、丸めずエラーにする。
    with pytest.raises(ValueError):
        run(selected=["S1_stdlib_mt19937"], repeat=0)
    with pytest.raises(ValueError):
        run(selected=["S1_stdlib_mt19937"], repeat=-1)


# --- selected_probe_names()/all_probe_names() のテスト(practice config向け) ---
# Task 1: scripts/run_on_host.shの`practice` configがREPRO_PROBESへ
# カンマ区切りの複数接頭辞(S11,S12,S13,S14,S15)を設定するために、
# main()に埋め込まれていた絞り込みロジックをこの2関数へ切り出した。


def test_repro_probes_unset_runs_all_probes(monkeypatch):
    monkeypatch.delenv("REPRO_PROBES", raising=False)
    assert runner.selected_probe_names(argv_probes=None) == runner.all_probe_names()


def test_repro_probes_accepts_comma_separated_prefixes(monkeypatch):
    # S11は後続タスク(Task 2)で初めて登録されるため、ここでは既存の
    # 接頭辞(S0, S1)でカンマ区切り絞り込みの機構そのものを確認する。
    # S11〜S15が実際に選択されることはdry-run(手動確認)と後続タスクの
    # 実測で検証する。
    monkeypatch.setenv("REPRO_PROBES", "S0,S1")
    names = runner.selected_probe_names(argv_probes=None)
    assert names, "絞り込み結果が空になってはいけない"
    assert all(n.startswith(("S0", "S1")) for n in names)
    assert any(n.startswith("S0") for n in names)
    assert any(n.startswith("S1") for n in names)


# --- main()/CLI のテスト ---


def test_main_writes_default_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REPRO_HOST", "testhost")
    monkeypatch.setenv("REPRO_IMAGE", "testimage")
    monkeypatch.setenv("REPRO_CONFIG", "testconfig")

    main(["--probes", "S1_stdlib_mt19937"])

    out_path = tmp_path / "results" / "testhost__testimage__testconfig.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["schema"] == 1
    assert len(payload["probes"]) == 1


def test_main_falls_back_to_repro_probes_env_var(tmp_path, monkeypatch):
    # scripts/run_on_host.shのs10batch config向けの経路: `--probes`を
    # 省略したとき、REPRO_PROBES環境変数があればそれをセレクタとして使う
    # (run_containerが固定するコマンド行にCLI引数を足さず、-eフラグだけで
    # S10に絞り込むため)。
    monkeypatch.setenv("REPRO_PROBES", "S1")

    out_path = tmp_path / "out.json"
    main(["--out", str(out_path)])

    payload = json.loads(out_path.read_text())
    assert payload["probes"]
    assert all(p["probe"].startswith("S1") for p in payload["probes"])
    assert len(payload["probes"]) >= 4  # test_run_prefix_selection_matches_groupと同じ下限


def test_main_prefers_explicit_probes_flag_over_env_var(tmp_path, monkeypatch):
    # `--probes`が明示されたときはREPRO_PROBESより優先する(環境変数は
    # あくまでCLIが省略されたときのフォールバック)。
    monkeypatch.setenv("REPRO_PROBES", "S1")

    out_path = tmp_path / "out.json"
    main(["--probes", "S0_parallel_completion_order", "--out", str(out_path)])

    payload = json.loads(out_path.read_text())
    assert [p["probe"] for p in payload["probes"]] == ["S0_parallel_completion_order"]


def test_main_accepts_explicit_out_path(tmp_path):
    out_path = tmp_path / "custom" / "nested" / "out.json"
    main(["--probes", "S1_stdlib_mt19937", "--out", str(out_path)])
    assert out_path.exists()


def test_main_refuses_to_overwrite_existing_output(tmp_path):
    out_path = tmp_path / "out.json"
    out_path.write_text("not json, must survive untouched")

    with pytest.raises(SystemExit):
        main(["--probes", "S1_stdlib_mt19937", "--out", str(out_path)])

    assert out_path.read_text() == "not json, must survive untouched"


def test_main_checks_existing_output_before_running_any_probe(tmp_path, monkeypatch):
    # fix round 1 finding 1: 既存ファイルの検出を測定("run()"の呼び出し)
    # より後に置くと、共用ホストの占有時間を使い切ってから最後の一行で
    # 書き込みだけ拒否する、という本末転倒になる(このタスクが保護しよう
    # としている資源そのものを二重に浪費する)。end-stateだけを見るテスト
    # (上の test_main_refuses_to_overwrite_existing_output)はこの順序違反
    # を検出できない — 安いプローブでは「測定してから拒否」でも
    # 「拒否してから測定しない」でも最終的なファイルの中身は同じに見える
    # ため。ここでは`run()`自体を監視し、既存ファイルがある場合には
    # 一度も呼ばれないことを直接確認する。
    import reprobench.runner as runner_mod

    out_path = tmp_path / "out.json"
    out_path.write_text("existing result, must not be touched or re-measured")

    def _must_not_run(*args, **kwargs):
        raise AssertionError("run() must not be called when the output file already exists")

    monkeypatch.setattr(runner_mod, "run", _must_not_run)

    with pytest.raises(SystemExit):
        main(["--probes", "S1_stdlib_mt19937", "--out", str(out_path)])

    assert out_path.read_text() == "existing result, must not be touched or re-measured"


def test_main_exits_nonzero_on_empty_selection(tmp_path):
    out_path = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        main(["--probes", "Z_no_such_probe_prefix", "--out", str(out_path)])
    assert not out_path.exists()


def test_main_supports_repeat_flag(tmp_path):
    out_path = tmp_path / "out.json"
    main(["--probes", "S0_parallel_completion_order", "--repeat", "2", "--out", str(out_path)])
    payload = json.loads(out_path.read_text())
    assert len(payload["rounds"]) == 2


def test_main_exits_nonzero_on_repeat_less_than_one(tmp_path):
    out_path = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        main(["--probes", "S1_stdlib_mt19937", "--repeat", "0", "--out", str(out_path)])
    assert not out_path.exists()


def test_main_leaves_no_temp_file_after_a_successful_write(tmp_path):
    # fix round 1 small finding: 書き出しは一時ファイル+os.replaceで行う
    # (途中終了で壊れたJSONが本来のパスに残らないようにするため)。
    # 一時ファイルが後片付けされずに残らないことも確認する。
    out_path = tmp_path / "out.json"
    main(["--probes", "S1_stdlib_mt19937", "--out", str(out_path)])
    assert out_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_main_reports_oserror_cleanly_instead_of_a_raw_traceback(tmp_path):
    # fix round 1 small finding: mkdir/書き込みがOS側の理由で失敗したとき
    # (権限・壊れた親パスなど)も、空選択/既存ファイルの2経路と同じく
    # 一行のstderrメッセージ+非ゼロ終了にする(生のtracebackを外へ漏らさ
    # ない)。ここでは出力先の親コンポーネントを「ディレクトリではなく
    # 普通のファイル」にすることで、mkdir(parents=True)が
    # NotADirectoryError(OSErrorのサブクラス)で失敗する状況を作る。
    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a directory")
    out_path = blocker / "nested" / "out.json"

    with pytest.raises(SystemExit):
        main(["--probes", "S1_stdlib_mt19937", "--out", str(out_path)])


def test_main_trailing_comma_does_not_silently_select_everything(tmp_path):
    # "S1,"のような末尾カンマの空トークンをそのまま前方一致セレクタに
    # 混ぜると、空文字列は全てのプローブ名に前方一致する
    # (`"foo".startswith("")`は常にTrue)ため、「S1だけのつもり」が
    # 「全件」に静かに化けてしまう。空トークンは除去され、"S1,"は
    # "S1"と同じ(S1グループのみ)に解決されなければならない。
    out_path = tmp_path / "out.json"
    main(["--probes", "S1,", "--out", str(out_path)])
    payload = json.loads(out_path.read_text())
    probes = {p["probe"] for p in payload["probes"]}
    assert probes and all(name.startswith("S1") for name in probes)


def test_main_blank_only_selector_matches_nothing_and_errors(tmp_path):
    # カンマだけ("," )を渡すと空トークンしか残らず、フィルタ後の選択は
    # 空リストになる。これは「何にもマッチしない選択」として明示的に
    # エラーにすべきで、空トークンを素通りさせて全件実行に化けては
    # ならない。
    out_path = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        main(["--probes", ",", "--out", str(out_path)])
    assert not out_path.exists()


def test_main_supports_comma_separated_group_selection(tmp_path):
    out_path = tmp_path / "out.json"
    main(["--probes", "S1,S2", "--out", str(out_path)])
    payload = json.loads(out_path.read_text())
    probes = {p["probe"] for p in payload["probes"]}
    assert all(name.startswith(("S1", "S2")) for name in probes)
    assert any(name.startswith("S1") for name in probes)
    assert any(name.startswith("S2") for name in probes)
