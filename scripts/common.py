from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CATEGORICAL_ENCODER = "onehot"
DEFAULT_CV_SPLITS = 5
DEFAULT_RANDOM_STATE = 42


@dataclass
class OracleProblemSpec:
    features: List[str]
    targets: List[str]
    continuous: List[str]
    categorical: List[str]
    bounds: Dict[str, tuple]
    categorical_levels: Optional[Dict[str, list]] = None


@dataclass
class OracleBundle:
    oracle: Any
    models: Dict[str, Any]
    model_names: Dict[str, str]
    performance: Dict[str, pd.DataFrame]
    spec: OracleProblemSpec
    dataframe: Optional[pd.DataFrame] = None


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


SCORING = {
    "r2": "r2",
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(rmse, greater_is_better=False),
}


def make_problem_spec(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    targets: Sequence[str],
    continuous: Sequence[str],
    categorical: Sequence[str],
) -> OracleProblemSpec:
    categorical_levels = {
        feature: sorted(df[feature].dropna().unique().tolist())
        for feature in categorical
    }
    bounds = {
        feature: (
            float(pd.to_numeric(df[feature], errors="raise").min()),
            float(pd.to_numeric(df[feature], errors="raise").max()),
        )
        for feature in continuous
    }
    return OracleProblemSpec(
        features=list(features),
        targets=list(targets),
        continuous=list(continuous),
        categorical=list(categorical),
        bounds=bounds,
        categorical_levels=categorical_levels,
    )


def validate_spec(df: pd.DataFrame, spec: OracleProblemSpec) -> None:
    missing = [
        column
        for column in list(spec.features) + list(spec.targets)
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Spec references missing columns: {missing}")

    feature_set = set(spec.features)
    continuous_set = set(spec.continuous)
    categorical_set = set(spec.categorical)

    if continuous_set & categorical_set:
        overlap = sorted(continuous_set & categorical_set)
        raise ValueError(f"Features cannot be both continuous and categorical: {overlap}")

    if continuous_set | categorical_set != feature_set:
        raise ValueError("Spec.features must equal continuous + categorical features")

    for feature in spec.continuous:
        if feature not in spec.bounds:
            raise ValueError(f"Missing bounds for continuous feature '{feature}'")

    levels = spec.categorical_levels or {}
    for feature in spec.categorical:
        if feature not in levels or not levels[feature]:
            raise ValueError(f"Missing categorical levels for feature '{feature}'")


def build_preprocessor(
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    categorical_encoder: Any = DEFAULT_CATEGORICAL_ENCODER,
):
    transformers = []

    if continuous_features:
        transformers.append(("num", StandardScaler(), list(continuous_features)))

    if categorical_features:
        if categorical_encoder == "onehot":
            cat_encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        elif hasattr(categorical_encoder, "fit"):
            cat_encoder = categorical_encoder
        else:
            raise ValueError("Invalid categorical_encoder")

        transformers.append(("cat", cat_encoder, list(categorical_features)))

    if not transformers:
        raise ValueError("At least one feature is required to build a preprocessor")

    return ColumnTransformer(transformers=transformers)


def get_default_models(random_state: int = DEFAULT_RANDOM_STATE):
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_oracle_model(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    categorical_encoder: Any = DEFAULT_CATEGORICAL_ENCODER,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    models: Optional[Dict[str, Any]] = None,
):
    if len(df) < 2:
        raise ValueError("Need at least two rows to build an oracle model")

    X = df.loc[:, list(feature_cols)]
    y = pd.to_numeric(df[target_col], errors="raise").to_numpy(dtype=float)

    preprocessor = build_preprocessor(
        continuous_features,
        categorical_features,
        categorical_encoder=categorical_encoder,
    )

    if models is None:
        models = get_default_models(random_state)

    n_splits = min(int(cv_splits), len(df))
    if n_splits < 2:
        raise ValueError("Need at least two CV splits")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    fitted_models = {}

    for name, regressor in models.items():
        pipe = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", regressor),
            ]
        )

        result = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            # Forests already parallelise tree fitting. Serial folds avoid
            # nested worker pools and Windows temporary-file cleanup errors.
            n_jobs=1,
        )

        rows.append(
            {
                "model": name,
                "R2_mean": float(np.mean(result["test_r2"])),
                "R2_std": float(np.std(result["test_r2"])),
                "MAE_mean": float(-np.mean(result["test_mae"])),
                "RMSE_mean": float(-np.mean(result["test_rmse"])),
            }
        )
        fitted_models[name] = pipe

    results_df = (
        pd.DataFrame(rows)
        .sort_values("R2_mean", ascending=False)
        .reset_index(drop=True)
    )

    best_name = str(results_df.iloc[0]["model"])
    best_model = fitted_models[best_name]
    best_model.fit(X, y)

    return best_model, best_name, results_df


def _validated_row_from_input(
    x: Sequence[Any],
    feature_order: Sequence[str],
    bounds: Dict[str, tuple],
    categorical_levels: Dict[str, list],
) -> Dict[str, Any]:
    if len(x) != len(feature_order):
        raise ValueError("Incorrect input length")

    row: Dict[str, Any] = {}
    for value, feature in zip(x, feature_order):
        if feature in bounds:
            numeric_value = float(value)
            lower, upper = bounds[feature]
            if not (lower <= numeric_value <= upper):
                raise ValueError(f"{feature} outside bounds")
            row[feature] = numeric_value
        else:
            allowed = categorical_levels.get(feature, [])
            if value not in allowed:
                raise ValueError(f"Invalid category for {feature}")
            row[feature] = value
    return row


def build_multi_target_oracle(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    bounds: Dict[str, tuple],
    categorical_levels: Dict[str, list],
    categorical_encoder: Any = DEFAULT_CATEGORICAL_ENCODER,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    models: Optional[Dict[str, Any]] = None,
):
    oracle_models = {}
    model_names = {}
    performance = {}

    for target in target_cols:
        model, name, perf = build_oracle_model(
            df=df,
            feature_cols=feature_cols,
            target_col=target,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
            categorical_encoder=categorical_encoder,
            cv_splits=cv_splits,
            random_state=random_state,
            models=models,
        )
        oracle_models[target] = model
        model_names[target] = name
        performance[target] = perf

    def oracle(x):
        row = _validated_row_from_input(
            x,
            feature_order=feature_cols,
            bounds=bounds,
            categorical_levels=categorical_levels,
        )
        X_df = pd.DataFrame([row])
        return {
            target: float(model.predict(X_df)[0])
            for target, model in oracle_models.items()
        }

    return oracle, oracle_models, model_names, performance


def make_oracle(
    model,
    *,
    bounds: Dict[str, tuple],
    categorical_levels: Dict[str, list],
    feature_order: Sequence[str],
):
    def oracle(x):
        row = _validated_row_from_input(
            x,
            feature_order=feature_order,
            bounds=bounds,
            categorical_levels=categorical_levels,
        )
        X_df = pd.DataFrame([row])
        return float(model.predict(X_df)[0])

    return oracle


def build_oracle_from_spec(
    df: pd.DataFrame,
    spec: OracleProblemSpec,
    categorical_encoder: Any = DEFAULT_CATEGORICAL_ENCODER,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    models: Optional[Dict[str, Any]] = None,
):
    validate_spec(df, spec)

    if len(spec.targets) > 1:
        return build_multi_target_oracle(
            df=df,
            feature_cols=spec.features,
            target_cols=spec.targets,
            continuous_features=spec.continuous,
            categorical_features=spec.categorical,
            bounds=spec.bounds,
            categorical_levels=spec.categorical_levels or {},
            categorical_encoder=categorical_encoder,
            cv_splits=cv_splits,
            random_state=random_state,
            models=models,
        )

    model, name, perf = build_oracle_model(
        df=df,
        feature_cols=spec.features,
        target_col=spec.targets[0],
        continuous_features=spec.continuous,
        categorical_features=spec.categorical,
        categorical_encoder=categorical_encoder,
        cv_splits=cv_splits,
        random_state=random_state,
        models=models,
    )
    oracle = make_oracle(
        model,
        bounds=spec.bounds,
        categorical_levels=spec.categorical_levels or {},
        feature_order=spec.features,
    )
    return oracle, {spec.targets[0]: model}, {spec.targets[0]: name}, {spec.targets[0]: perf}


def build_oracle_bundle(
    df: pd.DataFrame,
    *,
    build_default_spec,
    spec: Optional[OracleProblemSpec] = None,
    categorical_encoder: Any = DEFAULT_CATEGORICAL_ENCODER,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    models: Optional[Dict[str, Any]] = None,
) -> OracleBundle:
    if spec is None:
        spec = build_default_spec(df)

    oracle_fn, oracle_models, model_names, performance = build_oracle_from_spec(
        df=df,
        spec=spec,
        categorical_encoder=categorical_encoder,
        cv_splits=cv_splits,
        random_state=random_state,
        models=models,
    )
    return OracleBundle(
        oracle=oracle_fn,
        models=oracle_models,
        model_names=model_names,
        performance=performance,
        spec=spec,
        dataframe=df.copy(),
    )


__all__ = [
    "DEFAULT_CATEGORICAL_ENCODER",
    "DEFAULT_CV_SPLITS",
    "DEFAULT_RANDOM_STATE",
    "OracleBundle",
    "OracleProblemSpec",
    "build_oracle_bundle",
    "build_oracle_from_spec",
    "build_oracle_model",
    "build_preprocessor",
    "get_default_models",
    "make_oracle",
    "make_problem_spec",
    "validate_spec",
]
