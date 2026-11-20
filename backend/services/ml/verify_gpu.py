"""Verify that this Python environment actually executes XGBoost and PyTorch on CUDA."""
import json
import numpy as np
import torch
import xgboost as xgb
from xgboost import XGBClassifier

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA unavailable")
print("PyTorch:", torch.__version__, torch.cuda.get_device_name(0))
tensor = torch.randn(256, 256, device="cuda")
torch.cuda.synchronize()
assert (tensor @ tensor).device.type == "cuda"
rng = np.random.default_rng(42)
X = rng.random((512, 12), dtype=np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(np.int32)
model = XGBClassifier(n_estimators=8, max_depth=3, tree_method="hist", device="cuda")
model.fit(X, y)
actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
if not actual.startswith("cuda"):
    raise RuntimeError(f"XGBoost silently fell back to {actual}")
print("XGBoost:", xgb.__version__, "actual device:", actual)
print("TRADEMIND GPU VERIFICATION PASSED")
