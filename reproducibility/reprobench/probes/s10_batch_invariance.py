"""S10: batch-invariance(バッチ構成が変わると同じ入力の数値が変わる現象)の
再設計実験。設計と事前登録された予測は `BATCH_INVARIANCE_REDESIGN.md`
(本ディレクトリ直下)を参照。本ファイルはその仕様書の実装である。

**S9 (`s9_llm.py`) とは独立したファイル。** S9は既存の記事の記述に紐づいて
おり、本ファイルはS9のモデルキャッシュ・トークナイザ状態を一切共有しない
(`tokenizer.padding_side` を書き換えるため、S9と状態を共有すると
S9プローブの挙動を変えてしまいかねない — 絶対にやってはいけないことの1つ)。

重い依存(`transformers`/`torch`)はalpine等に無いことがあるため、関数本体
の内部でimportする(モジュール先頭には置かない。パッケージdocstring
参照)。純粋関数(`compute_p_logit_position` / `trim_at_eos` /
`filler_content` / `content_texts`)はtorch等に依存しないため、モデル無し
でも単体テストできるよう、モジュール先頭に置く。

## 設計上の要点(再掲)

- 測るのは生成トークン列ではなく、固定した被験プロンプトPの**ロジット行
  そのもの**(`model.generate()`ではなく`model(**inputs).logits`の単一
  順伝播)。
- バッチの中身を変えて(同一プロンプト8本複製/長さの異なるfillerと混在/
  バッチ内の位置/バッチサイズ/パディング比率)、Pのロジットが動くかを見る。
- `padding_side`(left/right)も独立した軸として`b8_mixed_first`で両方
  測る。
- 位置の計算を間違えると全てが無意味になるため、`compute_p_logit_position`
  を純粋関数として切り出し、実行時には`attention_mask`で「その位置が実際に
  Pの最後の実トークンである」ことを毎回assertで確認する。

## 被験プロンプトP

`_P_PROMPT` は `s9_llm._PROMPT`(「List three distinct facts about the
Roman Empire, one short sentence each.」)をそのままimportして使う
(fix round 1: 当初は独自の短いプロンプト「What is the capital of
France? Answer in one word.」を使っていたが、これには2つの問題があった。
(1) S9の既存実測(68〜119トークン生成)と比較できない。(2) 「一言で
答えろ」という指示は2トークン程度でEOSに達するため、`S10_batch_invariance_
greedy`が記録する分岐位置にほとんど情報が無く、かつEOS到達後にバッチの
他行がまだ生成中だと`pad_token_id`で埋められた「実際には生成していない」
トークンが`first_divergence`に混入し、本物の分岐と見分けが付かなく
なっていた。定数として`import`することで、将来s9側がプロンプトを変えても
手で同期し直す必要が無く、2つのプローブが「同じPを見ている」という設計の
前提を壊さない)。`first_divergence`と合わせて同じモジュールからimportする。

## filler(文脈を埋める他系列)の設計

`_FILLER_BASE_SENTENCE` を `n` 回連結したものを filler の内容とする
(`filler_content`)。`n` を変えるだけで長さが単調に(かつ明確に)変わる
決定的な生成器であり、実際の文字列ではなく `(base sentence, repeat回数)`
の組をmetaに記録すれば完全に再現できる。

## EOS到達後のpadding混入を避ける(fix round 1)

`S10_batch_invariance_greedy`はバッチ生成を行うため、Pの行が他の行より
先にEOSへ到達すると、HFの`generate()`はPの行を`pad_token_id`で埋めながら
他の行の生成を続ける。これをそのまま`first_divergence`に渡すと、「Pの内容
自体が分岐した」のか「単にバッチ内の他の行が長く、Pの行がpad_token_idで
水増しされただけ」なのかを区別できない。`trim_at_eos`がPの行をその
**最初のEOS出現位置(含む)で切り詰めてから**digest・first_divergenceの
対象にすることで、この混入を取り除く。切り詰め前の生の token_ids も
`meta["configs"][name]["raw_token_ids"]`に残し、何個のpad_token_idが
切り詰められたかを確認できるようにする。

## pad_token

Qwen2.5-0.5B-Instructのtokenizerは実測時点で既に`pad_token`
(`<|endoftext|>`)を持っていたが、将来のtransformers/tokenizer設定変更で
これが無くなる可能性に備え、`tokenizer.pad_token is None`なら
`tokenizer.eos_token`を明示的に`pad_token`へ設定する。実際に使った
pad_token文字列・ID・それが「tokenizerのデフォルトのまま」か「ここで
明示的に設定したもの」かを両プローブの`meta`に記録する。
"""
import numpy as np

from reprobench.digest import compare_arrays, digest
from reprobench.probes.s9_llm import _PROMPT as _P_PROMPT  # noqa: F401 (S9と同じPを使う、モジュールdocstring参照)
from reprobench.probes.s9_llm import first_divergence  # noqa: F401 (再エクスポートではなく直接利用)
from reprobench.registry import HF_MODEL_ID, ProbeResult, register

_SEED = 20261014

# Qwen2.5-0.5B-Instructはchatテンプレート前提のモデルであるため(s9_llm.py
# と同じ理由)、P・fillerともに `tokenizer.apply_chat_template` を通す。
# `_P_PROMPT`自体はs9_llm._PROMPTのエイリアス(上のimport参照)。

_FILLER_BASE_SENTENCE = (
    "The archive contained a single faded photograph of a harbor at dawn."
)
_FILLER_SUFFIX = "Summarize this passage in one short sentence."

# 各configで使うfillerの「繰り返し回数」。繰り返し回数を変えるだけで
# 長さが単調に伸びる決定的な生成器(`filler_content`)への入力であり、
# 具体的な文字列そのものではなくこのタプルをmetaに記録することで再現性を
# 担保する(どちらで記録しても等価だが、こちらの方が遥かにコンパクト)。
_FILLER_REPEATS_7 = (1, 2, 3, 4, 5, 6, 7)
_FILLER_REPEATS_15 = tuple(range(1, 16))
_FILLER_REPEATS_PADHEAVY = (20, 25, 30)
_FILLER_REPEAT_LONG = (60,)

# バッチ構成(content)の定義。padding_sideには依存しない「何が何本、
# Pがどこにあるか」のみを決める純粋関数。torchに依存しないため、ここに
# 置きモデル無しでテストできるようにする。
_BASE_CONFIG_NAMES = (
    "b1",
    "b8_same",
    "b8_mixed_first",
    "b8_mixed_last",
    "b2_long",
    "b16_mixed",
    "b4_padheavy",
)

# S10_batch_invariance_logits が走らせる実際のconfig群(順序が
# digestの順序付きマッピングを決めるため、ここが唯一の正)。
# `b8_mixed_first`はpadding_side left/rightの両方を別configとして走らせる
# (設計書: 「さらに b8_mixed_first について padding_side を left / right
# の両方で測る」)。(content_config_name, padding_side)のタプルで表す。
LOGIT_CONFIG_ORDER = (
    "b1",
    "b8_same",
    "b8_mixed_first_right",
    "b8_mixed_first_left",
    "b8_mixed_last",
    "b2_long",
    "b16_mixed",
    "b4_padheavy",
)

_LOGIT_CONFIG_SPECS = {
    "b1": ("b1", "right"),
    "b8_same": ("b8_same", "right"),
    "b8_mixed_first_right": ("b8_mixed_first", "right"),
    "b8_mixed_first_left": ("b8_mixed_first", "left"),
    "b8_mixed_last": ("b8_mixed_last", "right"),
    "b2_long": ("b2_long", "right"),
    "b16_mixed": ("b16_mixed", "right"),
    "b4_padheavy": ("b4_padheavy", "right"),
}
assert set(LOGIT_CONFIG_ORDER) == set(_LOGIT_CONFIG_SPECS)

# S10_batch_invariance_greedy は decoder-only モデルのバッチ生成作法通り
# left paddingのみで走らせる(brief参照)。b8_mixed_firstの2変種はlogits
# プローブだけの軸なので、greedyは`_BASE_CONFIG_NAMES`をそのまま使う。
GREEDY_CONFIG_ORDER = _BASE_CONFIG_NAMES

_MAX_NEW_TOKENS = 128
# s9_llm.pyと同じ128(fix round 1: 一度64に下げたが、理由は未測定の
# 憶測による安全マージンに過ぎず、(a) S9の既存実測(68/90/87/119/65
# トークン生成)と比較できなくなる、(b) Pのプロンプトを短い指示に変えた
# ことと合わせて生成が2〜24トークンで終わってしまい分岐位置の分布を見る
# のにほぼ意味が無くなる、という2つの問題があった。Pをs9_llm._PROMPTに
# 揃えた上で128に戻し、実機(anakin RTX 4090)で実際に時間を測り直して
# 480秒に収まることを確認した(呼び出し元の報告参照)。


def filler_content(n_repeats: int) -> str:
    """`_FILLER_BASE_SENTENCE`を`n_repeats`回連結したfillerの内容を返す。

    決定的な生成器であり、`n_repeats`を増やすほど長さが単調に伸びる。
    実際の文字列ではなく`(_FILLER_BASE_SENTENCE, n_repeats)`の組を記録
    すれば完全に再現できるため、meta には repeat回数のタプルだけを残す。
    """
    body = " ".join([_FILLER_BASE_SENTENCE] * n_repeats)
    return body + " " + _FILLER_SUFFIX


def content_texts(name: str) -> tuple[list[str], int]:
    """config名から(バッチ各行の内容文字列のリスト, Pが入っている位置)を返す。

    `padding_side`には一切依存しない(どちらを使うかは呼び出し側が決める)。
    純粋関数であり、torch/transformersに依存しないためモデル無しでテスト
    できる。
    """
    if name == "b1":
        return [_P_PROMPT], 0
    if name == "b8_same":
        return [_P_PROMPT] * 8, 0
    if name == "b8_mixed_first":
        fillers = [filler_content(n) for n in _FILLER_REPEATS_7]
        return [_P_PROMPT] + fillers, 0
    if name == "b8_mixed_last":
        fillers = [filler_content(n) for n in _FILLER_REPEATS_7]
        return fillers + [_P_PROMPT], len(fillers)
    if name == "b2_long":
        fillers = [filler_content(n) for n in _FILLER_REPEAT_LONG]
        return [_P_PROMPT] + fillers, 0
    if name == "b16_mixed":
        fillers = [filler_content(n) for n in _FILLER_REPEATS_15]
        return [_P_PROMPT] + fillers, 0
    if name == "b4_padheavy":
        fillers = [filler_content(n) for n in _FILLER_REPEATS_PADHEAVY]
        return [_P_PROMPT] + fillers, 0
    raise ValueError(f"content_texts: unknown config name {name!r}")


def compute_p_logit_position(padding_side: str, p_real_len: int, padded_seq_len: int) -> int:
    """Pの最終実トークン位置の、パディング後シーケンス次元でのインデックスを返す。

    - `padding_side == "right"`: 各行の実トークンは常に位置0から始まり、
      パディングはその後ろに付く。したがってPの最終実トークンは
      Pの実トークン長に関わらずバッチ内の他系列の影響を一切受けず
      `p_real_len - 1` に固定される(Pがバッチ内のどこにいても、だ)。
    - `padding_side == "left"`: 各行の実トークンはパディング後シーケンスの
      末尾に右詰めされる。したがってPの最終実トークンは常に
      `padded_seq_len - 1`(最後のインデックス)であり、Pの実トークン長にも
      他系列の長さにも依存しない。

    どちらの式も「バッチ内の他系列が何であれ、Pの最終実トークン位置は
    this行の実トークン長とpadding_sideだけで決まる」という設計(brief)
    通りの性質を体現している。呼び出し側はこの位置が実際にPの最後の実
    トークンであることを`attention_mask`で必ず検証すること(実装は
    `_run_logit_config`内のassert文を参照)。
    """
    if padding_side == "right":
        if not (1 <= p_real_len <= padded_seq_len):
            raise ValueError(
                f"compute_p_logit_position: p_real_len={p_real_len} out of range "
                f"for padded_seq_len={padded_seq_len}"
            )
        return p_real_len - 1
    if padding_side == "left":
        if padded_seq_len < 1:
            raise ValueError(f"compute_p_logit_position: padded_seq_len={padded_seq_len} < 1")
        return padded_seq_len - 1
    raise ValueError(f"compute_p_logit_position: unknown padding_side {padding_side!r}")


def trim_at_eos(token_ids: list[int], eos_token_ids: int | list[int]) -> list[int]:
    """`token_ids`を最初のEOS出現位置(それ自身を含む)で切り詰める。

    `eos_token_ids`はint 1個、またはその複数形(Qwen2.5-0.5B-Instructの
    `model.generation_config.eos_token_id`は実際に`[151645, 151643]`という
    2要素のリストであり、後者`151643`は`<|endoftext|>`、本プローブの
    `pad_token_id`でもある — つまりHFのbatch generateが「このEOSのどれかに
    到達したらこの行は完了」と判定する基準そのもの)。どちらも出現しなければ
    `token_ids`をそのまま返す。純粋関数でありtorch/transformersに依存しない
    ためモデル無しでテストできる。

    バッチ生成では、Pの行が他の行より先にEOSへ到達しても、HFの
    `generate()`はバッチ全体が終わるまでPの行を`pad_token_id`で埋め続ける
    (モジュールdocstring「EOS到達後のpadding混入を避ける」参照)。この
    埋め草をdigest/first_divergenceの比較対象に含めると、「Pの内容自体が
    バッチ構成で変わった」のか「単に他の行が長くPの行が水増しされた
    だけ」なのかを区別できなくなるため、呼び出し側は必ずこの関数を通した
    後の列を比較に使うこと。`eos_token_ids`に`pad_token_id`が含まれる
    場合、埋め草の先頭そのものが「停止位置」として扱われるため、この
    区別が自動的に付く。
    """
    ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids)
    for i, t in enumerate(token_ids):
        if t in ids:
            return token_ids[: i + 1]
    return list(token_ids)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


_MODEL_CACHE: dict[str, tuple] = {}


def _load_model(dtype_name: str = "float32"):
    """`HF_MODEL_ID`をロードしプロセス内キャッシュする。

    S9の`_MODEL_CACHE`とは完全に独立したキャッシュ・独立したtokenizer/model
    インスタンスを持つ(モジュールdocstring参照: `tokenizer.padding_side`を
    書き換えるため、S9と同じオブジェクトを共有してはならない)。
    """
    if dtype_name not in _MODEL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, dtype_name)
        device = _device()
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        pad_token_source = "tokenizer_default"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_token_source = "set_to_eos_token"
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, dtype=dtype)
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[dtype_name] = (tokenizer, model, device, pad_token_source)
    return _MODEL_CACHE[dtype_name]


def _to_chat_text(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _tokenize_batch(tokenizer, device: str, contents: list[str], padding_side: str):
    """`contents`をchatテンプレート適用後、`padding_side`でパディングしてtokenizeする。

    `tokenizer.padding_side`はtokenizerオブジェクトの可変状態であり、
    呼ぶたびに明示的に設定し直す(他のconfigの設定を引き継がないため)。
    """
    tokenizer.padding_side = padding_side
    chat_texts = [_to_chat_text(tokenizer, c) for c in contents]
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True)
    return inputs.to(device)


def _canonical_p_real_len(tokenizer) -> int:
    """`_P_PROMPT`単体(パディング無し)をtokenizeした実トークン長。

    バッチ構成に関わらず「Pそのものの実トークン長」は不変のはずであり
    (padding_sideにもfillerの有無にも依存しない)、この値を各configで
    観測した値と突き合わせることで、位置計算の前提そのものを検証する。
    """
    chat_text = _to_chat_text(tokenizer, _P_PROMPT)
    ids = tokenizer(chat_text, return_tensors="pt")["input_ids"]
    return int(ids.shape[1])


def _run_logit_config(tokenizer, model, device: str, name: str, canonical_p_real_len: int) -> dict:
    """1つのlogit configを実行し、Pの最終実トークン位置のロジット行等を返す。"""
    import torch

    content_name, padding_side = _LOGIT_CONFIG_SPECS[name]
    contents, p_index = content_texts(content_name)

    inputs = _tokenize_batch(tokenizer, device, contents, padding_side)
    attn = inputs["attention_mask"]
    seq_len = int(attn.shape[1])

    # Pの実トークン長はバッチ構成に関わらず不変であるはず(brief:
    # 「b1 と b8_same で P の実トークン長が同じであることを assert で
    # 確認すること」)。ここでは全configに対して同じ不変条件を検証する
    # (b1/b8_sameに限定する理由が無く、他のconfigでも同じ不変条件が
    # 成り立つはずだから)。
    observed_p_real_len = int(attn[p_index].sum().item())
    assert observed_p_real_len == canonical_p_real_len, (
        f"{name}: observed P real token length {observed_p_real_len} != "
        f"canonical {canonical_p_real_len} (position calculation would be wrong)"
    )

    pos = compute_p_logit_position(padding_side, canonical_p_real_len, seq_len)

    # 位置が実際にPの「最後の」実トークンであることを確認する。
    assert int(attn[p_index, pos].item()) == 1, (
        f"{name}/{padding_side}: position {pos} is padding, not a real token"
    )
    if padding_side == "right" and pos + 1 < seq_len:
        assert int(attn[p_index, pos + 1].item()) == 0, (
            f"{name}/{padding_side}: position {pos} is not P's LAST real token "
            f"(position {pos + 1} is also real)"
        )

    with torch.no_grad():
        logits = model(**inputs).logits
    row = logits[p_index, pos, :].float().cpu().numpy()

    sorted_desc = np.sort(row)[::-1]
    return {
        "row_logits": row,
        "digest": digest(row),
        "argmax_token_id": int(np.argmax(row)),
        "margin_top1_top2": float(sorted_desc[0] - sorted_desc[1]),
        "p_real_len": observed_p_real_len,
        "padded_seq_len": seq_len,
        "padding_side": padding_side,
        "batch_size": len(contents),
        "p_index": p_index,
    }


@register("S10_batch_invariance_logits", requires=("transformers", "torch", "hf_model"))
def _batch_invariance_logits() -> ProbeResult:
    import torch

    torch.manual_seed(_SEED)
    tokenizer, model, device, pad_token_source = _load_model("float32")
    canonical_p_real_len = _canonical_p_real_len(tokenizer)

    configs: dict[str, dict] = {}
    for name in LOGIT_CONFIG_ORDER:
        # 設計書の予測5: 同一configを同一プロセス内で2回実行し、プロセス内
        # 決定性を確認する(config間の差が測定ノイズでないことを示すため)。
        first = _run_logit_config(tokenizer, model, device, name, canonical_p_real_len)
        second = _run_logit_config(tokenizer, model, device, name, canonical_p_real_len)
        stable = first["digest"] == second["digest"]
        configs[name] = {**first, "stable_within_process": stable, "digest_rerun": second["digest"]}

    b1_logits = configs["b1"]["row_logits"]
    for name, entry in configs.items():
        cmp = compare_arrays(b1_logits, entry["row_logits"])
        entry["bit_equal_vs_b1"] = cmp["bit_equal"]
        entry["max_abs_vs_b1"] = cmp["max_abs"]
        entry["max_ulp_vs_b1"] = cmp["max_ulp"]

    # digestの順序付きマッピング: (config名, ロジットdigest)のタプルの
    # リストとして持つ(宣言順=LOGIT_CONFIG_ORDER)。dictではなくlistにする
    # のは、`canonical_bytes`のdict実装がキーでソートし直すため
    # (digest.py参照)、ここで意図している「宣言順」そのものを保持したい
    # から。どちらでも「ホスト間で同じ入力なら同じdigest」という性質は
    # 変わらないが、list-of-tuplesの方がこのモジュールが定義する順序を
    # そのまま反映する。
    digest_map_ordered = [(name, configs[name]["digest"]) for name in LOGIT_CONFIG_ORDER]

    meta = {
        "seed": _SEED,
        "model": HF_MODEL_ID,
        "device": device,
        "dtype": "float32",
        "p_prompt": _P_PROMPT,
        "p_real_token_length": canonical_p_real_len,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "pad_token_source": pad_token_source,
        "filler_base_sentence": _FILLER_BASE_SENTENCE,
        "filler_suffix": _FILLER_SUFFIX,
        "filler_repeats": {
            "b8_mixed_first_and_last": list(_FILLER_REPEATS_7),
            "b16_mixed": list(_FILLER_REPEATS_15),
            "b4_padheavy": list(_FILLER_REPEATS_PADHEAVY),
            "b2_long": list(_FILLER_REPEAT_LONG),
        },
        "config_order": list(LOGIT_CONFIG_ORDER),
        "p1_b1_vs_b8_same_p_real_len_equal": (
            configs["b1"]["p_real_len"] == configs["b8_same"]["p_real_len"]
        ),
        "configs": {
            name: {
                "digest": entry["digest"],
                "digest_rerun": entry["digest_rerun"],
                "stable_within_process": entry["stable_within_process"],
                "argmax_token_id": entry["argmax_token_id"],
                "margin_top1_top2": entry["margin_top1_top2"],
                "p_real_len": entry["p_real_len"],
                "padded_seq_len": entry["padded_seq_len"],
                "padding_side": entry["padding_side"],
                "batch_size": entry["batch_size"],
                "p_index": entry["p_index"],
                "bit_equal_vs_b1": entry["bit_equal_vs_b1"],
                "max_abs_vs_b1": entry["max_abs_vs_b1"],
                "max_ulp_vs_b1": entry["max_ulp_vs_b1"],
            }
            for name, entry in configs.items()
        },
    }
    return ProbeResult(
        probe="S10_batch_invariance_logits",
        digest=digest(digest_map_ordered),
        scalars={},
        sample_b64=None,
        meta=meta,
    )


def _run_greedy_config(tokenizer, model, device: str, name: str) -> dict:
    import torch

    contents, p_index = content_texts(name)
    inputs = _tokenize_batch(tokenizer, device, contents, padding_side="left")
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False)
    raw_gen_ids = output[p_index][prompt_len:].tolist()
    # バッチ内の他行がPより長く生成を続けた場合、Pの行は`pad_token_id`で
    # 埋められる。それをdigest/first_divergenceの対象から除くため、
    # 実際にgenerate()が使う停止基準(model.generation_config.eos_token_id、
    # Qwen2.5-0.5B-Instructでは[151645, 151643]の2値)の最初の出現位置で
    # 切り詰める(モジュールdocstring「EOS到達後のpadding混入を避ける」
    # 参照)。
    gen_ids = trim_at_eos(raw_gen_ids, model.generation_config.eos_token_id)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "token_ids": gen_ids,
        "raw_token_ids": raw_gen_ids,
        "text": text,
        "n_tokens": len(gen_ids),
        "n_tokens_raw": len(raw_gen_ids),
        "batch_size": len(contents),
        "padded_prompt_len": prompt_len,
        "p_index": p_index,
    }


@register("S10_batch_invariance_greedy", requires=("transformers", "torch", "hf_model"))
def _batch_invariance_greedy() -> ProbeResult:
    import torch

    torch.manual_seed(_SEED)
    tokenizer, model, device, pad_token_source = _load_model("float32")

    configs: dict[str, dict] = {}
    for name in GREEDY_CONFIG_ORDER:
        configs[name] = _run_greedy_config(tokenizer, model, device, name)

    b1_ids = configs["b1"]["token_ids"]
    for name, entry in configs.items():
        entry["first_divergence_vs_b1"] = first_divergence(b1_ids, entry["token_ids"])
        entry["digest"] = digest(entry["token_ids"])

    digest_map_ordered = [(name, configs[name]["digest"]) for name in GREEDY_CONFIG_ORDER]

    meta = {
        "seed": _SEED,
        "model": HF_MODEL_ID,
        "device": device,
        "dtype": "float32",
        "do_sample": False,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "padding_side": "left",
        "p_prompt": _P_PROMPT,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "pad_token_source": pad_token_source,
        "filler_base_sentence": _FILLER_BASE_SENTENCE,
        "filler_suffix": _FILLER_SUFFIX,
        "filler_repeats": {
            "b8_mixed_first_and_last": list(_FILLER_REPEATS_7),
            "b16_mixed": list(_FILLER_REPEATS_15),
            "b4_padheavy": list(_FILLER_REPEATS_PADHEAVY),
            "b2_long": list(_FILLER_REPEAT_LONG),
        },
        "eos_token_ids": model.generation_config.eos_token_id,
        "config_order": list(GREEDY_CONFIG_ORDER),
        "configs": {
            name: {
                "digest": entry["digest"],
                "token_ids": entry["token_ids"],
                "raw_token_ids": entry["raw_token_ids"],
                "text": entry["text"],
                "n_tokens": entry["n_tokens"],
                "n_tokens_raw": entry["n_tokens_raw"],
                "batch_size": entry["batch_size"],
                "padded_prompt_len": entry["padded_prompt_len"],
                "p_index": entry["p_index"],
                "first_divergence_vs_b1": entry["first_divergence_vs_b1"],
            }
            for name, entry in configs.items()
        },
    }
    return ProbeResult(
        probe="S10_batch_invariance_greedy",
        digest=digest(digest_map_ordered),
        scalars={},
        sample_b64=None,
        meta=meta,
    )
