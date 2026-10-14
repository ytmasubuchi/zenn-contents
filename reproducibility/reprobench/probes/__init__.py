"""プローブ実装を集めるパッケージ(Task 4以降が中身を追加する)。

各モジュールは `reprobench.registry.register` でプローブ関数を登録する。
登録はプローブモジュールのimport時に副作用として起こるため、このパッケージ
自身に責務はない。

重い依存(torch/lightgbm/sklearn/transformers等)のimportは、各プローブ
モジュールのトップレベルではなく関数本体の内部で行うこと。モジュール
トップレベルで`import torch`すると、torchの無いalpineイメージでその
モジュール自体がimportできなくなり、同じモジュール内の他のプローブも
道連れになる。
"""
from reprobench.probes import s0_hidden_inputs  # 副作用: S0プローブ群を登録する
from reprobench.probes import s1_prng  # 副作用: S1プローブ群を登録する
from reprobench.probes import s2_reduction  # 副作用: S2プローブ群を登録する
from reprobench.probes import s3_blas  # 副作用: S3プローブ群を登録する
from reprobench.probes import s4_libm  # 副作用: S4プローブ群を登録する
from reprobench.probes import s5_sklearn  # 副作用: S5プローブ群を登録する
from reprobench.probes import s6_lightgbm  # 副作用: S6プローブ群を登録する
from reprobench.probes import s7_torch_cpu  # 副作用: S7プローブ群を登録する
from reprobench.probes import s8_torch_gpu  # 副作用: S8プローブ群を登録する
from reprobench.probes import s9_llm  # 副作用: S9プローブ群を登録する
