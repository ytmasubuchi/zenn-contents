"""S9: LLM推論(`HF_MODEL_ID`固定)とtemperatureの意味論。

重い依存(`transformers`/`torch`)はalpine等に無いことがあるため、関数本体
の内部でimportする(モジュール先頭には置かない。パッケージdocstring
参照)。`first_divergence`だけは純粋関数でありtorch等に依存しないため、
モジュール先頭に置く(比較器がこれをそのままimportして使えるように
するため)。`warnings`は標準ライブラリであり重い依存ではないため、通常
通りモジュール先頭でimportする。

**`temperature=0`は使わない**: transformersに「temperature=0」という
決定的モードは存在しない。`do_sample=False`が公式の決定的パス
(greedy)であり、`temperature=0` + `do_sample=True`はstrictly positiveを
要求して弾かれる。`S9_greedy_*`系はすべて`do_sample=False`で行う。

`S9_temperature_zero_api_semantics`だけは例外で、`temperature=0`を
実際に渡すと何が起きるか(例外/警告/無視)を**そのまま**記録する測定
であり、ワークアラウンドしてはいけない。

モデルは関数内で`transformers.AutoModelForCausalLM`/`AutoTokenizer`を
使ってロードするが、1プロセス内で複数のS9プローブを実行しても毎回
ロードし直さないよう、dtype文字列をキーにしたモジュールレベルの
キャッシュ(`_MODEL_CACHE`)を使う。**キャッシュするのはモデルだけ**
であり、生成結果はキャッシュしない。resolution 3が要求する通り、
各プローブ関数は呼ばれるたびに`torch.manual_seed(20261014)`で明示的に
reseedする(モデルのキャッシュと「プロセス内で2回呼んでも同じ結果に
なる」ことは独立な要求であり、片方を満たしても他方は自動的には
満たされない)。

**fix round 1 finding 7**: モデルは`torch.cuda.is_available()`なら
明示的にCUDAへ移動する(`_device`)。以前はモデル・入力をCPU上に置いた
まま生成しており、GPU計測ホスト上でも`envinfo`が`cuda.available=True`を
報告する傍らでS9だけが静かにCPU推論していた——これではGPUホスト間の比較が
成立しない。全S9プローブの`meta["device"]`に実際に使ったデバイスを記録
する。

**fix round 1 finding 8**: `from_pretrained`の`torch_dtype=`引数は非推奨
であり(`-W error`下では将来のtransformersバージョンで例外になりうる)、
後継の`dtype=`を使う。この変更はtransformers 5.19.0(本プローブの検証に
使ったバージョン)で確認済み。`dtype=`を受け付けない古いtransformersを
Task 12が固定ピン留めする場合は、ここを`torch_dtype=`に戻す必要がある。
同じtransformers 5.19.0で、`do_sample=False`時に`top_k`/`top_p`を渡すと
`[transformers] The following generation flags are not valid and may be
ignored: ['top_k']`という(Pythonの`warnings`経由ではなくライブラリの
ログ経由の)メッセージが出ることも確認した——`-W error`には引っかからない
ため実害は無いが、Task 12が別バージョンをピン留めした際にこの出力が
変わる(あるいは`warnings`経由の例外に変わる)可能性があるので、
`dtype=`の件と合わせてここに記録しておく(fix round 2、deferred)。

**fix round 1 finding 4**: Qwen2.5-0.5B-Instructの`generation_config.json`
は既定で`top_p=0.8, top_k=20, repetition_penalty=1.05`を出荷しており、
これらは(greedyでも適用される`repetition_penalty`を除き)`do_sample=True`
のときtemperatureの後段でトークン分布を切り詰める。`S9_sampling_t0_7_seed`
/`S9_sampling_t1_2_seed`は**温度だけを単離していない**(モデル出荷設定を
そのまま使う「現実的な」構成)。この事実を記録から読めるようにするため、
全S9プローブの`meta`に`model.generation_config.to_dict()`を残し、
サンプリング系プローブには温度以外を無効化する`_ISOLATING_KWARGS`
(`top_p=1.0, top_k=0, repetition_penalty=1.0`)を明示的に渡す`_isolated`
版を追加した。`S9_logit_margin`(greedyだがrepetition_penaltyは適用される)
も単離設定を使う。
"""
import warnings

from reprobench.digest import digest
from reprobench.registry import HF_MODEL_ID, ProbeResult, register

_SEED = 20261014
_MAX_NEW_TOKENS = 128
# Qwen2.5-0.5B-Instructは素のテキスト続きではなくchatテンプレートを前提に
# 調整されたモデルであるため、生のプロンプト文字列をそのままトークナイズ
# するのではなく、tokenizer.apply_chat_templateを通す(そうしないと
# 出力が早期にEOSへ落ちたり、サンプリング温度の効果が見えにくい退化した
# 生成になりやすい)。
_PROMPT = "List three distinct facts about the Roman Empire, one short sentence each."

# 温度以外のサンプリングパラメータを無効化し、temperatureだけを単離する
# (fix round 1 finding 4)。do_sample=Falseの経路ではtop_p/top_kはそもそも
# warperとして使われないため実質no-opだが、repetition_penaltyはgreedyでも
# 適用されるLogitsProcessorなので、S9_logit_marginにとってはこれが本質的。
_ISOLATING_KWARGS = {"top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}

_MODEL_CACHE: dict[str, tuple] = {}


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """2つのトークン列`a`/`b`が最初に食い違う位置を返す(皆無なら`None`)。

    比較器が2環境の生成トークン列を突き合わせ、「どこまでは一致していて
    どこから割れたか」を報告するための共通ヘルパー。長さが違う場合、
    共通の接頭辞が完全に一致していれば短い方の長さ(=一方が終わった
    位置)を食い違いの位置として返す。
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(dtype_name: str = "float32"):
    """`HF_MODEL_ID`をdtype別にロードしてプロセス内キャッシュする。

    資格要件`"hf_model"`がHuggingFaceキャッシュへの存在を既に確認して
    いるため、ここでのロードはオフライン(ネットワーク不要)で完了する
    はずである。CUDAが使える環境ではモデルをCUDAへ移動する(fix round 1
    finding 7)。返り値は`(tokenizer, model, device)`。
    """
    if dtype_name not in _MODEL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, dtype_name)
        device = _device()
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        # fix round 1 finding 8: `torch_dtype=`は非推奨、`dtype=`が後継
        # (モジュールdocstring参照)。
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, dtype=dtype)
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[dtype_name] = (tokenizer, model, device)
    return _MODEL_CACHE[dtype_name]


def _encode_prompt(tokenizer, device: str, batch: int = 1):
    messages = [{"role": "user", "content": _PROMPT}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text] * batch, return_tensors="pt")
    return inputs.to(device)


def _greedy_generate(tokenizer, model, device: str, batch: int):
    import torch

    torch.manual_seed(_SEED)
    inputs = _encode_prompt(tokenizer, device, batch=batch)
    output = model.generate(
        **inputs,
        max_new_tokens=_MAX_NEW_TOKENS,
        do_sample=False,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output[0][prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_ids, text


def _greedy_result(name: str, dtype_name: str, batch: int) -> ProbeResult:
    tokenizer, model, device = _load_model(dtype_name)
    gen_ids, text = _greedy_generate(tokenizer, model, device, batch)
    meta = {
        "seed": _SEED,
        "model": HF_MODEL_ID,
        "prompt": _PROMPT,
        "device": device,
        "batch_size": batch,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "do_sample": False,
        "dtype": dtype_name,
        "token_ids": gen_ids,
        "text": text,
        "n_tokens": len(gen_ids),
        "generation_config": model.generation_config.to_dict(),
    }
    return ProbeResult(probe=name, digest=digest(gen_ids), scalars={}, sample_b64=None, meta=meta)


@register("S9_greedy_batch1", requires=("transformers", "torch", "hf_model"))
def _greedy_batch1() -> ProbeResult:
    return _greedy_result("S9_greedy_batch1", "float32", batch=1)


@register("S9_greedy_batch8", requires=("transformers", "torch", "hf_model"))
def _greedy_batch8() -> ProbeResult:
    # batch8は同じプロンプトを8本並べ、先頭1本の出力だけを取る(brief記載)。
    # fix round 1 small finding: 8本とも**同一**プロンプトなので、トークン
    # 長は全系列で等しく、paddingは発生しない。このプローブが調べているのは
    # padding効果ではなく、バッチ内に他系列が存在すること自体がattention・
    # カーネル選択(バッチサイズに応じて異なるCUDA/CPUカーネルが選ばれうる)
    # に影響し、batch1と結果が変わるかである。
    r = _greedy_result("S9_greedy_batch8", "float32", batch=8)
    r.meta["took_row"] = 0
    return r


@register("S9_greedy_fp32", requires=("transformers", "torch", "hf_model"))
def _greedy_fp32() -> ProbeResult:
    return _greedy_result("S9_greedy_fp32", "float32", batch=1)


@register("S9_greedy_bf16", requires=("transformers", "torch", "hf_model"))
def _greedy_bf16() -> ProbeResult:
    return _greedy_result("S9_greedy_bf16", "bfloat16", batch=1)


@register("S9_greedy_fp16", requires=("transformers", "torch", "hf_model"))
def _greedy_fp16() -> ProbeResult:
    return _greedy_result("S9_greedy_fp16", "float16", batch=1)


@register("S9_temperature_zero_api_semantics", requires=("transformers", "torch", "hf_model"))
def _temperature_zero_api_semantics() -> ProbeResult:
    import torch

    torch.manual_seed(_SEED)
    tokenizer, model, device = _load_model("float32")
    inputs = _encode_prompt(tokenizer, device, batch=1)

    exception_type = None
    exception_message = None
    output = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            output = model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.0,
            )
        except Exception as exc:  # 測定対象そのもの。ワークアラウンドしない。
            exception_type = type(exc).__name__
            exception_message = str(exc)
        finally:
            # fix round 1 small finding: generate()が例外を送出した場合、
            # `with`ブロックを抜けた後のコードは実行されない(例外が
            # そのまま外側へ伝播するため)。それまでに記録された警告を
            # 読み落とさないよう、`with`ブロックの外ではなく`finally`
            # (=例外の有無に関わらず必ず実行される)で読む。
            warning_messages = [str(w.message) for w in caught]

    token_ids = None
    text = None
    if exception_type is None:
        prompt_len = inputs["input_ids"].shape[1]
        token_ids = output[0][prompt_len:].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)

    if exception_type is not None:
        summary = f"exception:{exception_type}:{exception_message}"
    elif warning_messages:
        summary = "warning:" + "|".join(warning_messages)
    else:
        summary = "ignored:no_exception_no_warning"

    meta = {
        "seed": _SEED,
        "model": HF_MODEL_ID,
        "prompt": _PROMPT,
        "device": device,
        "do_sample": True,
        "temperature_requested": 0.0,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "warning_messages": warning_messages,
        "token_ids": token_ids,
        "text": text,
        "summary": summary,
        "generation_config": model.generation_config.to_dict(),
    }
    return ProbeResult(
        probe="S9_temperature_zero_api_semantics",
        digest=digest(summary),
        scalars={},
        sample_b64=None,
        meta=meta,
    )


def _sampling_result(name: str, temperature: float, isolate: bool) -> ProbeResult:
    import torch

    torch.manual_seed(_SEED)
    tokenizer, model, device = _load_model("float32")
    inputs = _encode_prompt(tokenizer, device, batch=1)
    gen_kwargs = dict(
        max_new_tokens=_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
    )
    if isolate:
        gen_kwargs.update(_ISOLATING_KWARGS)
    output = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output[0][prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    meta = {
        "seed": _SEED,
        "model": HF_MODEL_ID,
        "prompt": _PROMPT,
        "device": device,
        "do_sample": True,
        "temperature": temperature,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "dtype": "float32",
        "token_ids": gen_ids,
        "text": text,
        "n_tokens": len(gen_ids),
        # fix round 1 finding 4: "model_shipped_defaults"はgeneration_config.json
        # のtop_p/top_k/repetition_penaltyがそのまま効いた「現実的な」構成、
        # "temperature_only"はそれらを無効化しtemperatureだけを単離した構成。
        "isolation": "temperature_only" if isolate else "model_shipped_defaults",
        "generation_config": model.generation_config.to_dict(),
    }
    if isolate:
        meta["isolating_kwargs"] = dict(_ISOLATING_KWARGS)
    return ProbeResult(probe=name, digest=digest(gen_ids), scalars={}, sample_b64=None, meta=meta)


@register("S9_sampling_t0_7_seed", requires=("transformers", "torch", "hf_model"))
def _sampling_t0_7_seed() -> ProbeResult:
    return _sampling_result("S9_sampling_t0_7_seed", 0.7, isolate=False)


@register("S9_sampling_t0_7_seed_isolated", requires=("transformers", "torch", "hf_model"))
def _sampling_t0_7_seed_isolated() -> ProbeResult:
    return _sampling_result("S9_sampling_t0_7_seed_isolated", 0.7, isolate=True)


@register("S9_sampling_t1_2_seed", requires=("transformers", "torch", "hf_model"))
def _sampling_t1_2_seed() -> ProbeResult:
    return _sampling_result("S9_sampling_t1_2_seed", 1.2, isolate=False)


@register("S9_sampling_t1_2_seed_isolated", requires=("transformers", "torch", "hf_model"))
def _sampling_t1_2_seed_isolated() -> ProbeResult:
    return _sampling_result("S9_sampling_t1_2_seed_isolated", 1.2, isolate=True)


@register("S9_logit_margin", requires=("transformers", "torch", "hf_model"))
def _logit_margin() -> ProbeResult:
    import torch

    torch.manual_seed(_SEED)
    tokenizer, model, device = _load_model("float32")
    inputs = _encode_prompt(tokenizer, device, batch=1)
    outputs = model.generate(
        **inputs,
        max_new_tokens=_MAX_NEW_TOKENS,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        **_ISOLATING_KWARGS,  # fix round 1 finding 4: repetition_penaltyを無効化する
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs.sequences[0][prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    margins = []
    for step_scores in outputs.scores:
        top2 = torch.topk(step_scores[0], k=2).values
        margins.append(float((top2[0] - top2[1]).item()))

    scalars = {"mean_margin": float(sum(margins) / len(margins)) if margins else 0.0}
    meta = {
        "seed": _SEED,
        "model": HF_MODEL_ID,
        "prompt": _PROMPT,
        "device": device,
        "do_sample": False,
        "max_new_tokens": _MAX_NEW_TOKENS,
        "dtype": "float32",
        "token_ids": gen_ids,
        "text": text,
        "n_tokens": len(gen_ids),
        "margins": margins,
        "margins_hex": [m.hex() for m in margins],
        "mean_margin_hex": scalars["mean_margin"].hex(),
        "isolation": "temperature_only",
        "isolating_kwargs": dict(_ISOLATING_KWARGS),
        "generation_config": model.generation_config.to_dict(),
    }
    return ProbeResult(
        probe="S9_logit_margin",
        digest=digest(gen_ids),
        scalars=scalars,
        sample_b64=None,
        meta=meta,
    )
