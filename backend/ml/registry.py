"""
Model registry: deserialize once, reuse forever.

There are ~2031 artifacts in models/. Loading three of them per request costs
roughly 800ms of pure disk I/O, so handles are cached by absolute path. The
artifacts are small (DL hidden sizes 32-64), which makes a generous cache cheap.

Every handle also reports the feature width its model actually expects, read
from the artifact itself rather than assumed. See architectures.infer_input_width
for why that matters.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch

from backend.config import settings
from backend.ml import architectures as arch

logger = logging.getLogger(__name__)

Kind = Literal["tree", "sequence", "tabular"]

# Maps a model name to its constructor. Commodity nets take (width, hidden);
# crypto nets take (width,); tabular nets take (width,).
_COMMODITY_NETS = {
    "LSTM": lambda w: arch.LSTMModel(w, settings.commodity_hidden_size),
    "GRU": lambda w: arch.GRUModel(w, settings.commodity_hidden_size),
    "TRANSFORMER": lambda w: arch.TransformerModel(w, settings.commodity_hidden_size),
    "N-BEATS": lambda w: arch.NBeatsModel(settings.commodity_seq_length, w),
    "NBEATS": lambda w: arch.NBeatsModel(settings.commodity_seq_length, w),
    "TFT": lambda w: arch.TFTModel(w, settings.commodity_hidden_size),
}

_CRYPTO_NETS = {
    "LSTM": arch.CryptoLSTM,
    "GRU": arch.CryptoGRU,
    "TRANSFORMER": arch.CryptoTransformer,
    "NBEATS_LITE": lambda w: arch.NBEATS_Lite(w, settings.crypto_seq_length),
    "TFT_LITE": arch.TFT_Lite,
}


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing or cannot be deserialized."""


@dataclass
class ModelHandle:
    """A deserialized, ready-to-predict model plus its input contract."""

    name: str
    kind: Kind
    input_width: int
    raw: Any
    path: Path

    def predict(self, X: np.ndarray) -> float:
        """
        Run inference on a single sample.

        X arrives shaped for the widest available feature set. If this model was
        trained on fewer features (a reverted champion), the trailing sentiment
        columns are dropped here. That truncation is only valid because
        StandardScaler normalizes each column independently and the sentiment
        block occupies the final columns - so slicing post-scaling is equivalent
        to having scaled the narrower set.
        """
        X = self._fit_width(X)

        if self.kind == "tree":
            return float(np.asarray(self.raw.predict(X)).ravel()[0])

        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            if self.kind == "sequence" and tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            return float(self.raw(tensor).squeeze().item())

    def _fit_width(self, X: np.ndarray) -> np.ndarray:
        have = X.shape[-1]
        want = self.input_width
        if have == want:
            return X
        if have > want:
            return X[..., :want]
        raise ArtifactError(
            f"{self.name}: needs {want} features, received {have}. "
            "The scaler is narrower than the model: artifacts are inconsistent."
        )


class ModelRegistry:
    """Thread-safe LRU cache over deserialized artifacts."""

    def __init__(self, max_size: int | None = None):
        self._cache: OrderedDict[str, ModelHandle] = OrderedDict()
        self._max = max_size or settings.model_cache_size
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # -- public -----------------------------------------------------------
    def get(self, path: Path, *, model_name: str, family: str = "commodity") -> ModelHandle:
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]

        handle = self._load(path, model_name, family)  # outside lock: I/O

        with self._lock:
            self._cache[key] = handle
            self._cache.move_to_end(key)
            self.misses += 1
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return handle

    def stats(self) -> dict:
        with self._lock:
            return {
                "cached": len(self._cache),
                "capacity": self._max,
                "hits": self.hits,
                "misses": self.misses,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # -- loading ----------------------------------------------------------
    def _load(self, path: Path, model_name: str, family: str) -> ModelHandle:
        if not path.exists():
            raise ArtifactError(f"Artifact not found: {path}")

        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                return self._load_xgboost(path, model_name)
            if suffix == ".txt":
                return self._load_lightgbm(path, model_name)
            if suffix == ".cbm":
                return self._load_catboost(path, model_name)
            if suffix == ".pkl":
                return self._load_pickle(path, model_name)
            if suffix == ".pt":
                return self._load_torch(path, model_name, family)
        except ArtifactError:
            raise
        except Exception as e:
            raise ArtifactError(f"Failed to load {path.name}: {type(e).__name__}: {e}") from e

        raise ArtifactError(f"Unsupported artifact format: {path.name}")

    def _load_xgboost(self, path: Path, name: str) -> ModelHandle:
        from xgboost import XGBRegressor

        m = XGBRegressor()
        m.load_model(str(path))
        return ModelHandle(name, "tree", int(m.n_features_in_), m, path)

    def _load_lightgbm(self, path: Path, name: str) -> ModelHandle:
        from lightgbm import Booster

        b = Booster(model_file=str(path))
        return ModelHandle(name, "tree", int(b.num_feature()), b, path)

    def _load_catboost(self, path: Path, name: str) -> ModelHandle:
        from catboost import CatBoostRegressor

        m = CatBoostRegressor()
        m.load_model(str(path))
        return ModelHandle(name, "tree", len(m.feature_names_), m, path)

    def _load_pickle(self, path: Path, name: str) -> ModelHandle:
        """
        Deserialize a pickled estimator and recover its input width.

        Width has to be probed across several attributes because the libraries
        disagree. CatBoostRegressor in particular reports n_features_in_ == 0
        while carrying a fully populated feature_names_, so a naive read of the
        sklearn attribute yields zero and truncates the input to nothing.
        """
        obj = joblib.load(path)
        width = self._probe_width(obj)
        if not width:
            raise ArtifactError(f"{path.name}: cannot determine feature width")
        return ModelHandle(name, "tree", int(width), obj, path)

    @staticmethod
    def _probe_width(obj: Any) -> int | None:
        for attr in ("n_features_in_", "n_features_"):
            value = getattr(obj, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        names = getattr(obj, "feature_names_", None)
        if names:
            return len(names)

        num_feature = getattr(obj, "num_feature", None)
        if callable(num_feature):
            try:
                value = num_feature()
                if value:
                    return int(value)
            except Exception:
                pass

        booster = getattr(obj, "get_booster", None)
        if callable(booster):
            try:
                names = booster().feature_names
                if names:
                    return len(names)
            except Exception:
                pass
        return None

    @staticmethod
    def _normalize_nbeats_keys(state: dict) -> dict:
        """
        Accept both N-BEATS key layouts for the same network.

        The original pipeline declared its three blocks as separate attributes
        (block1/block2/block3). training-scripts-new/common/architectures.py
        holds them in an nn.ModuleList, which serializes as blocks.0/blocks.1/
        blocks.2. Layer shapes, activation, dropout placement and the
        doubly-residual forward pass are identical between the two - loading a
        ModuleList checkpoint into the attribute-style module after this remap
        reproduces the training model's output exactly (verified max|diff| ==
        0.0) - so only the names need translating.
        """
        if not any(k.startswith("blocks.") for k in state):
            return state
        return {
            re.sub(r"^blocks\.(\d+)\.", lambda m: f"block{int(m.group(1)) + 1}.", k): v
            for k, v in state.items()
        }

    def _load_torch(self, path: Path, name: str, family: str) -> ModelHandle:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = self._normalize_nbeats_keys(state)

        key = name.upper().replace("-", "").replace("_", "")
        if family == "crypto":
            table, seq, kind = _CRYPTO_NETS, settings.crypto_seq_length, "sequence"
            lookup = {k.replace("-", "").replace("_", ""): v for k, v in _CRYPTO_NETS.items()}
        elif family == "tabular":
            table, seq, kind = None, 1, "tabular"
            lookup = None
        else:
            table, seq, kind = _COMMODITY_NETS, settings.commodity_seq_length, "sequence"
            lookup = {k.replace("-", "").replace("_", ""): v for k, v in _COMMODITY_NETS.items()}

        width = arch.infer_input_width(state, seq)
        if width is None:
            raise ArtifactError(f"{path.name}: cannot infer input width from state_dict")

        if kind == "tabular":
            model = arch.TabularLSTM(width)
        else:
            ctor = lookup.get(key)
            if ctor is None:
                raise ArtifactError(f"Unknown architecture '{name}' for family '{family}'")
            model = ctor(width)

        model.load_state_dict(state)
        model.eval()
        return ModelHandle(name, kind, width, model, path)


registry = ModelRegistry()
