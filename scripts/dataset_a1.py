from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .common import (
    DEFAULT_CATEGORICAL_ENCODER,
    DEFAULT_CV_SPLITS,
    DEFAULT_RANDOM_STATE,
    OracleBundle,
    OracleProblemSpec,
    build_oracle_bundle,
    make_problem_spec,
)


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_a1_aspuru_guzik_suzuki_numeric.csv"
DEFAULT_TARGET = "yield_percent"


def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    dataset_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    return pd.read_csv(dataset_path)


def build_default_spec(df: pd.DataFrame) -> OracleProblemSpec:
    return make_problem_spec(
        df,
        features=[
            "temperature_C",
            "Pd_mol_percent",
            "ArBpin_equiv",
            "K3PO4_equiv",
        ],
        targets=["yield_percent"],
        continuous=[
            "temperature_C",
            "Pd_mol_percent",
            "ArBpin_equiv",
            "K3PO4_equiv",
        ],
        categorical=[],
    )


def build_oracle(
    data_path: Optional[str] = None,
    spec: Optional[OracleProblemSpec] = None,
    categorical_encoder: Any = DEFAULT_CATEGORICAL_ENCODER,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    models: Optional[Dict[str, Any]] = None,
) -> OracleBundle:
    df = load_dataset(data_path=data_path)
    return build_oracle_bundle(
        df,
        build_default_spec=build_default_spec,
        spec=spec,
        categorical_encoder=categorical_encoder,
        cv_splits=cv_splits,
        random_state=random_state,
        models=models,
    )


_DEFAULT_BUNDLE: Optional[OracleBundle] = None


def get_default_oracle() -> OracleBundle:
    global _DEFAULT_BUNDLE
    if _DEFAULT_BUNDLE is None:
        _DEFAULT_BUNDLE = build_oracle()
    return _DEFAULT_BUNDLE


def oracle(x):
    return get_default_oracle().oracle(x)


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_TARGET",
    "build_default_spec",
    "build_oracle",
    "get_default_oracle",
    "load_dataset",
    "oracle",
]
