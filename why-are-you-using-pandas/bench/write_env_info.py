import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import env_info  # noqa: E402

import pandas as pd
import polars as pl
import numpy as np

try:
    import pyarrow

    pyarrow_version = pyarrow.__version__
except ImportError:
    pyarrow_version = None

info = env_info()
info["pandas_version"] = pd.__version__
info["polars_version"] = pl.__version__
info["pyarrow_version"] = pyarrow_version
info["numpy_version"] = np.__version__

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "env_info.json"), "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(json.dumps(info, indent=2, ensure_ascii=False))
