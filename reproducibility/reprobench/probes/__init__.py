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
