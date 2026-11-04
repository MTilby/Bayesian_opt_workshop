from __future__ import annotations

import re
from pathlib import Path
from pprint import pformat
from typing import Optional, Sequence


def normalize_module_name(module_name: str) -> str:
    name = module_name.strip()
    if name.endswith(".py"):
        name = name[:-3]
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
    if not name:
        raise ValueError("module_name must contain at least one letter, number, or underscore")
    if name[0].isdigit():
        name = f"oracle_{name}"
    return name


def _validate_columns(
    *,
    features: Sequence[str],
    targets: Sequence[str],
    continuous: Sequence[str],
    categorical: Sequence[str],
    default_target: Optional[str],
) -> str:
    feature_list = list(features)
    target_list = list(targets)
    continuous_list = list(continuous)
    categorical_list = list(categorical)

    if not feature_list:
        raise ValueError("features cannot be empty")
    if not target_list:
        raise ValueError("targets cannot be empty")
    if set(continuous_list) & set(categorical_list):
        overlap = sorted(set(continuous_list) & set(categorical_list))
        raise ValueError(f"continuous and categorical features overlap: {overlap}")
    if set(continuous_list) | set(categorical_list) != set(feature_list):
        raise ValueError("features must equal continuous + categorical")

    for sequence_name, values in {
        "features": feature_list,
        "targets": target_list,
        "continuous": continuous_list,
        "categorical": categorical_list,
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{sequence_name} contains duplicate entries")

    resolved_default = default_target or target_list[0]
    if resolved_default not in target_list:
        raise ValueError("default_target must be one of targets")
    return resolved_default


def build_oracle_module_text(
    *,
    module_name: str,
    data_filename: str,
    features: Sequence[str],
    targets: Sequence[str],
    continuous: Sequence[str],
    categorical: Sequence[str],
    default_target: Optional[str] = None,
) -> str:
    normalized_module_name = normalize_module_name(module_name)
    resolved_default_target = _validate_columns(
        features=features,
        targets=targets,
        continuous=continuous,
        categorical=categorical,
        default_target=default_target,
    )
    data_path_fragment = Path(data_filename).as_posix()

    return f"""from __future__ import annotations

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


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / {data_path_fragment!r}
DEFAULT_TARGET = {resolved_default_target!r}


def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    dataset_path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH
    return pd.read_csv(dataset_path)


def build_default_spec(df: pd.DataFrame) -> OracleProblemSpec:
    return make_problem_spec(
        df,
        features={pformat(list(features), width=88, sort_dicts=False)},
        targets={pformat(list(targets), width=88, sort_dicts=False)},
        continuous={pformat(list(continuous), width=88, sort_dicts=False)},
        categorical={pformat(list(categorical), width=88, sort_dicts=False)},
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
"""


def preview_oracle_module(
    *,
    module_name: str,
    data_filename: str,
    features: Sequence[str],
    targets: Sequence[str],
    continuous: Sequence[str],
    categorical: Sequence[str],
    default_target: Optional[str] = None,
) -> str:
    return build_oracle_module_text(
        module_name=module_name,
        data_filename=data_filename,
        features=features,
        targets=targets,
        continuous=continuous,
        categorical=categorical,
        default_target=default_target,
    )


def write_oracle_module(
    *,
    module_name: str,
    data_filename: str,
    features: Sequence[str],
    targets: Sequence[str],
    continuous: Sequence[str],
    categorical: Sequence[str],
    default_target: Optional[str] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    normalized_module_name = normalize_module_name(module_name)
    module_text = build_oracle_module_text(
        module_name=normalized_module_name,
        data_filename=data_filename,
        features=features,
        targets=targets,
        continuous=continuous,
        categorical=categorical,
        default_target=default_target,
    )

    destination_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / f"{normalized_module_name}.py"

    if destination_path.exists() and not overwrite:
        raise FileExistsError(
            f"{destination_path} already exists; set overwrite=True to replace it"
        )

    destination_path.write_text(module_text, encoding="utf-8")
    return destination_path


__all__ = [
    "build_oracle_module_text",
    "normalize_module_name",
    "preview_oracle_module",
    "write_oracle_module",
]
