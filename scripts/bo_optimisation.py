import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    ExpSineSquared,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.preprocessing import OneHotEncoder

from .common import OracleBundle, OracleProblemSpec


@dataclass
class BORunResult:
    target: str
    seed: Optional[int]
    search_space: Dict[str, Any]
    history: List[tuple]
    log: pd.DataFrame
    best_conditions: Dict[str, Any]
    best_value: float
    gp: Optional[GaussianProcessRegressor] = None


COMPARISON_OVERRIDE_KEYS = (
    "ENCODING_MODE",
    "CUSTOM_CATEGORICAL_MAPS",
    "INIT_STRATEGY",
    "KERNEL_NAME",
    "ACQ_FUNCTION",
    "EI_XI",
    "UCB_BETA",
    "N_INITIAL",
    "N_ITER",
    "BATCH_SIZE",
    "N_CANDIDATES",
)


class ConditionEncoder:
    def __init__(
        self,
        spec: OracleProblemSpec,
        search_space: Dict[str, Any],
        *,
        categorical_mode: str = "onehot",
        custom_categorical_maps: Optional[Dict[str, Dict[Any, Sequence[float]]]] = None,
    ):
        self.feature_order = list(spec.features)
        self.continuous = [feature for feature in self.feature_order if feature in spec.continuous]
        self.categorical = [feature for feature in self.feature_order if feature in spec.categorical]
        self.categorical_mode = categorical_mode
        self.encoder: Optional[OneHotEncoder] = None
        self.ordinal_maps: Dict[str, Dict[Any, float]] = {}
        self.custom_categorical_maps: Dict[str, Dict[Any, np.ndarray]] = {}

        if self.categorical_mode not in {"onehot", "ordinal", "custom"}:
            raise ValueError("categorical_mode must be 'onehot', 'ordinal', or 'custom'")

        if self.categorical:
            categories = {feature: list(search_space[feature]) for feature in self.categorical}
            if self.categorical_mode == "onehot":
                self.encoder = OneHotEncoder(
                    categories=[categories[feature] for feature in self.categorical],
                    sparse_output=False,
                    handle_unknown="ignore",
                )
                dummy_row = pd.DataFrame(
                    [{feature: categories[feature][0] for feature in self.categorical}]
                )
                self.encoder.fit(dummy_row[self.categorical])
            elif self.categorical_mode == "ordinal":
                self.ordinal_maps = {
                    feature: {
                        level: float(index)
                        for index, level in enumerate(categories[feature])
                    }
                    for feature in self.categorical
                }
            else:
                provided_maps = custom_categorical_maps or {}
                for feature in self.categorical:
                    if feature not in provided_maps:
                        raise ValueError(
                            f"Missing custom_categorical_maps entry for '{feature}'"
                        )
                    feature_map = provided_maps[feature]
                    missing = [
                        level for level in categories[feature]
                        if level not in feature_map
                    ]
                    if missing:
                        raise ValueError(
                            f"Custom categorical map for '{feature}' is missing levels: {missing}"
                        )
                    encoded_vectors = {}
                    vector_lengths = set()
                    for level in categories[feature]:
                        vector = np.asarray(feature_map[level], dtype=float).reshape(-1)
                        if vector.size == 0:
                            raise ValueError(
                                f"Custom categorical map for '{feature}' has an empty vector for '{level}'"
                            )
                        encoded_vectors[level] = vector
                        vector_lengths.add(int(vector.size))
                    if len(vector_lengths) != 1:
                        raise ValueError(
                            f"Custom categorical map for '{feature}' must use a consistent vector length"
                        )
                    self.custom_categorical_maps[feature] = encoded_vectors

    def transform_one(self, conditions: Dict[str, Any]) -> np.ndarray:
        numeric = np.array(
            [float(conditions[feature]) for feature in self.continuous],
            dtype=float,
        ) if self.continuous else np.empty(0, dtype=float)

        if not self.categorical:
            return numeric

        if self.categorical_mode == "onehot":
            cat_frame = pd.DataFrame([{feature: conditions[feature] for feature in self.categorical}])
            categorical = self.encoder.transform(cat_frame[self.categorical])[0]
        elif self.categorical_mode == "ordinal":
            categorical = np.array(
                [
                    self.ordinal_maps[feature][conditions[feature]]
                    for feature in self.categorical
                ],
                dtype=float,
            )
        else:
            categorical = np.concatenate(
                [
                    self.custom_categorical_maps[feature][conditions[feature]]
                    for feature in self.categorical
                ]
            )
        return np.concatenate([numeric, categorical])

    def transform_many(self, conditions_list: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not conditions_list:
            return np.empty((0, 0), dtype=float)
        return np.vstack([self.transform_one(conditions) for conditions in conditions_list])


def build_search_space(
    spec: OracleProblemSpec,
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    levels = spec.categorical_levels or {}
    search_space: Dict[str, Any] = {}

    for feature in spec.features:
        if feature in spec.categorical:
            search_space[feature] = list(levels[feature])
        else:
            lower, upper = spec.bounds[feature]
            search_space[feature] = (float(lower), float(upper))

    override = override or {}
    for feature, value in override.items():
        if feature not in search_space:
            raise ValueError(f"Unknown feature in SEARCH_SPACE_OVERRIDE: '{feature}'")

        if feature in spec.categorical:
            if isinstance(value, str):
                raise ValueError(f"Categorical override for '{feature}' must be a list-like of levels")
            normalized = list(dict.fromkeys(list(value)))
            if not normalized:
                raise ValueError(f"Categorical override for '{feature}' cannot be empty")
            allowed = set(levels[feature])
            invalid = [item for item in normalized if item not in allowed]
            if invalid:
                raise ValueError(f"Invalid categorical levels for '{feature}': {invalid}")
            search_space[feature] = normalized
        else:
            if len(value) != 2:
                raise ValueError(f"Continuous override for '{feature}' must have two values")
            lower = float(value[0])
            upper = float(value[1])
            base_lower, base_upper = spec.bounds[feature]
            if lower < base_lower or upper > base_upper or lower > upper:
                raise ValueError(
                    f"Override for '{feature}' must stay within original bounds {spec.bounds[feature]}"
                )
            search_space[feature] = (lower, upper)

    return search_space


def _random_continuous_samples(
    n: int,
    bounds: Sequence[tuple],
    rng: np.random.Generator,
) -> np.ndarray:
    if not bounds:
        return np.empty((n, 0), dtype=float)
    samples = np.empty((n, len(bounds)), dtype=float)
    for index, (lower, upper) in enumerate(bounds):
        if lower == upper:
            samples[:, index] = float(lower)
        else:
            samples[:, index] = rng.uniform(lower, upper, size=n)
    return samples


def _latin_hypercube_samples(
    n: int,
    bounds: Sequence[tuple],
    rng: np.random.Generator,
) -> np.ndarray:
    if not bounds:
        return np.empty((n, 0), dtype=float)
    samples = np.empty((n, len(bounds)), dtype=float)
    for index, (lower, upper) in enumerate(bounds):
        if lower == upper:
            samples[:, index] = float(lower)
            continue
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        samples[:, index] = lower + u * (upper - lower)
    return samples


def _grid_samples(
    n: int,
    bounds: Sequence[tuple],
    rng: np.random.Generator,
) -> np.ndarray:
    if not bounds:
        return np.empty((n, 0), dtype=float)
    grid_size = int(np.ceil(n ** (1.0 / max(len(bounds), 1))))
    axes = []
    for lower, upper in bounds:
        if lower == upper:
            axes.append(np.array([float(lower)]))
        else:
            axes.append(np.linspace(lower, upper, grid_size))
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack(mesh, axis=-1).reshape(-1, len(bounds))
    if len(grid) > n:
        indices = rng.choice(len(grid), size=n, replace=False)
        grid = grid[np.atleast_1d(indices)]
    return grid


def _maximin_samples(
    pool: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(pool) <= n:
        return pool
    chosen = [int(rng.integers(len(pool)))]
    while len(chosen) < n:
        chosen_points = pool[chosen]
        distances = np.linalg.norm(
            pool[:, None, :] - chosen_points[None, :, :],
            axis=2,
        )
        min_distance = np.min(distances, axis=1)
        min_distance[chosen] = -np.inf
        next_index = int(np.argmax(min_distance))
        chosen.append(next_index)
    return pool[chosen]


def _assemble_conditions(
    feature_order: Sequence[str],
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    continuous_samples: np.ndarray,
    search_space: Dict[str, Any],
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = []
    for row in continuous_samples:
        condition: Dict[str, Any] = {}
        continuous_map = {
            feature: float(row[index])
            for index, feature in enumerate(continuous_features)
        }
        for feature in feature_order:
            if feature in categorical_features:
                sampled = rng.choice(search_space[feature])
                condition[feature] = sampled.item() if hasattr(sampled, "item") else sampled
            else:
                condition[feature] = continuous_map[feature]
        conditions.append(condition)
    return conditions


def _condition_key(conditions: Dict[str, Any], feature_order: Sequence[str]):
    return tuple(conditions[feature] for feature in feature_order)


def _all_categorical(search_space: Dict[str, Any]) -> bool:
    return all(isinstance(value, list) for value in search_space.values())


def _random_condition(
    search_space: Dict[str, Any],
    feature_order: Sequence[str],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    condition = {}
    for feature in feature_order:
        value = search_space[feature]
        if isinstance(value, list):
            sampled = rng.choice(value)
            condition[feature] = sampled.item() if hasattr(sampled, "item") else sampled
        else:
            lower, upper = value
            condition[feature] = float(lower) if lower == upper else float(rng.uniform(lower, upper))
    return condition


def _sample_conditions(
    search_space: Dict[str, Any],
    feature_order: Sequence[str],
    n: int,
    rng: np.random.Generator,
    *,
    seen_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    # Proposing a candidate must not mark it as an evaluated experiment.
    seen_keys = set() if seen_keys is None else set(seen_keys)

    if _all_categorical(search_space):
        levels = [search_space[feature] for feature in feature_order]
        all_keys = list(product(*levels))
        remaining = [key for key in all_keys if key not in seen_keys]
        if not remaining:
            return []
        if len(remaining) > n:
            indices = rng.choice(len(remaining), size=n, replace=False)
            remaining = [remaining[index] for index in np.atleast_1d(indices)]
        return [dict(zip(feature_order, key)) for key in remaining]

    conditions = []
    attempts = 0
    max_attempts = max(1000, n * 200)
    while len(conditions) < n and attempts < max_attempts:
        candidate = _random_condition(search_space, feature_order, rng)
        key = _condition_key(candidate, feature_order)
        attempts += 1
        if key in seen_keys:
            continue
        seen_keys.add(key)
        conditions.append(candidate)
    return conditions


def _sample_initial_conditions(
    spec: OracleProblemSpec,
    search_space: Dict[str, Any],
    feature_order: Sequence[str],
    n: int,
    rng: np.random.Generator,
    *,
    initial_strategy: str = "random",
    seen_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if initial_strategy == "random":
        return _sample_conditions(search_space, feature_order, n, rng, seen_keys=seen_keys)

    if initial_strategy not in {"latin_hypercube", "space_filling", "grid"}:
        raise ValueError(
            "initial_strategy must be 'random', 'latin_hypercube', 'space_filling', or 'grid'"
        )

    if seen_keys is None:
        seen_keys = set()

    if _all_categorical(search_space):
        return _sample_conditions(search_space, feature_order, n, rng, seen_keys=seen_keys)

    continuous_features = [feature for feature in feature_order if feature in spec.continuous]
    categorical_features = [feature for feature in feature_order if feature in spec.categorical]
    bounds = [search_space[feature] for feature in continuous_features]

    if not bounds:
        return _sample_conditions(search_space, feature_order, n, rng, seen_keys=seen_keys)

    conditions: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = 10

    while len(conditions) < n and attempts < max_attempts:
        batch_size = max(n - len(conditions), 4)
        if initial_strategy == "latin_hypercube":
            continuous_samples = _latin_hypercube_samples(batch_size, bounds, rng)
        elif initial_strategy == "space_filling":
            pool_size = max(batch_size * 10, 200)
            pool = _latin_hypercube_samples(pool_size, bounds, rng)
            continuous_samples = _maximin_samples(pool, batch_size, rng)
        else:
            continuous_samples = _grid_samples(batch_size, bounds, rng)

        proposed = _assemble_conditions(
            feature_order,
            continuous_features,
            categorical_features,
            continuous_samples,
            search_space,
            rng,
        )

        for candidate in proposed:
            key = _condition_key(candidate, feature_order)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            conditions.append(candidate)
            if len(conditions) == n:
                break
        attempts += 1

    if len(conditions) < n:
        conditions.extend(
            _sample_conditions(
                search_space,
                feature_order,
                n - len(conditions),
                rng,
                seen_keys=seen_keys,
            )
        )

    return conditions[:n]


def _make_objective(bundle: OracleBundle, target: str):
    if target not in bundle.spec.targets:
        raise ValueError(f"Target '{target}' not available; choose from {bundle.spec.targets}")

    def objective(conditions: Dict[str, Any]) -> float:
        x = [conditions[feature] for feature in bundle.spec.features]
        outputs = bundle.oracle(x)
        if isinstance(outputs, dict):
            return float(outputs[target])
        return float(outputs)

    return objective


def _build_gp(random_state: Optional[int], kernel_name: str = "matern52") -> GaussianProcessRegressor:
    base_kernels = {
        "matern52": Matern(nu=2.5),
        "matern32": Matern(nu=1.5),
        "rbf": RBF(length_scale=1.0),
        "rq": RationalQuadratic(length_scale=1.0, alpha=1.0),
        "periodic": ExpSineSquared(length_scale=1.0, periodicity=1.0),
    }
    if kernel_name not in base_kernels:
        raise ValueError(
            "kernel_name must be one of: 'matern52', 'matern32', 'rbf', 'rq', 'periodic'"
        )

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) * base_kernels[kernel_name]
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=random_state,
    )


def make_default_acquisition_scorer(
    acquisition_function: str = "ei",
    *,
    ei_xi: float = 0.0,
    ucb_beta: float = 2.0,
) -> Callable[[GaussianProcessRegressor, np.ndarray, np.ndarray, Optional[np.random.Generator]], np.ndarray]:
    if acquisition_function not in {"ei", "pi", "ucb", "greedy", "thompson"}:
        raise ValueError(
            "acquisition_function must be one of: 'ei', 'pi', 'ucb', 'greedy', 'thompson'"
        )

    def score_candidates(
        gp: GaussianProcessRegressor,
        X_candidates: np.ndarray,
        y_train: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        mean, std = gp.predict(X_candidates, return_std=True)
        best_y = float(np.max(y_train))
        safe_std = np.maximum(std, 1e-9)

        if acquisition_function == "ei":
            improvement = mean - best_y - ei_xi
            z = improvement / safe_std
            scores = improvement * norm.cdf(z) + safe_std * norm.pdf(z)
            return np.where(std > 0, scores, 0.0)

        if acquisition_function == "pi":
            improvement = mean - best_y - ei_xi
            z = improvement / safe_std
            scores = norm.cdf(z)
            return np.where(std > 0, scores, 0.0)

        if acquisition_function == "ucb":
            return mean + ucb_beta * std

        if acquisition_function == "greedy":
            return mean

        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(mean, safe_std)

    return score_candidates


def _log_record(
    experiment_index: int,
    source: str,
    conditions: Dict[str, Any],
    feature_order: Sequence[str],
    pred_mean: float,
    pred_std: float,
    oracle_value: float,
    batch: int = 0,
    batch_position: int = 1,
) -> Dict[str, Any]:
    row = {
        "experiment": experiment_index,
        "source": source,
        "batch": batch,
        "batch_position": batch_position,
    }
    for feature in feature_order:
        row[feature] = conditions[feature]
    row["pred_mean"] = pred_mean
    row["pred_std"] = pred_std
    row["oracle_value"] = oracle_value
    return row


def run_bo(
    bundle: OracleBundle,
    *,
    target: str,
    search_space_override: Optional[Dict[str, Any]] = None,
    categorical_mode: str = "onehot",
    custom_categorical_maps: Optional[Dict[str, Dict[Any, Sequence[float]]]] = None,
    encoder_factory: Optional[Callable[[OracleProblemSpec, Dict[str, Any]], Any]] = None,
    initial_strategy: str = "random",
    initial_design_sampler: Optional[
        Callable[
            [OracleProblemSpec, Dict[str, Any], Sequence[str], int, np.random.Generator, Optional[set]],
            List[Dict[str, Any]],
        ]
    ] = None,
    kernel_name: str = "matern52",
    gp_builder: Optional[Callable[[Optional[int]], GaussianProcessRegressor]] = None,
    score_candidates: Optional[
        Callable[[GaussianProcessRegressor, np.ndarray, np.ndarray, Optional[np.random.Generator]], np.ndarray]
    ] = None,
    acquisition_function: str = "ei",
    ei_xi: float = 0.0,
    ucb_beta: float = 2.0,
    n_initial: int = 12,
    n_iter: int = 30,
    batch_size: int = 1,
    n_candidates: int = 2000,
    random_state: Optional[int] = 42,
    save_logs: bool = False,
    log_dir: str = "bo_experiment_logs",
    log_name: Optional[str] = None,
) -> BORunResult:
    """Run synchronous BO using posterior-mean fantasies within each batch.

    n_iter counts BO batches; the maximum budget is n_initial + n_iter *
    batch_size. All batch members are selected before any of their oracle
    responses are obtained. A finite search space can end with a smaller batch.
    Log predictions come from the model fitted to real data before the batch.
    """
    if n_initial < 1:
        raise ValueError("N_INITIAL must be at least 1")
    if n_iter < 0:
        raise ValueError("N_ITER must be at least 0")
    if n_candidates < 1:
        raise ValueError("N_CANDIDATES must be at least 1")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("BATCH_SIZE must be a positive integer")
    if n_candidates < batch_size:
        raise ValueError("N_CANDIDATES must be at least BATCH_SIZE")

    if score_candidates is None:
        score_candidates = make_default_acquisition_scorer(
            acquisition_function,
            ei_xi=ei_xi,
            ucb_beta=ucb_beta,
        )

    spec = bundle.spec
    feature_order = list(spec.features)
    search_space = build_search_space(spec, search_space_override)
    if encoder_factory is None:
        encoder = ConditionEncoder(
            spec,
            search_space,
            categorical_mode=categorical_mode,
            custom_categorical_maps=custom_categorical_maps,
        )
    else:
        encoder = encoder_factory(spec, search_space)
    objective = _make_objective(bundle, target)
    rng = np.random.default_rng(random_state)

    history: List[tuple] = []
    log_records: List[Dict[str, Any]] = []
    seen_keys = set()

    if initial_design_sampler is None:
        initial_conditions = _sample_initial_conditions(
            spec,
            search_space,
            feature_order,
            n_initial,
            rng,
            initial_strategy=initial_strategy,
            seen_keys=seen_keys,
        )
    else:
        initial_conditions = initial_design_sampler(
            spec,
            search_space,
            feature_order,
            n_initial,
            rng,
            seen_keys,
        )
    if not initial_conditions:
        raise ValueError("Unable to sample an initial design from the search space")

    # Sampler hooks may mutate their set; only returned experiments count.
    seen_keys = set()
    for position, conditions in enumerate(initial_conditions, start=1):
        key = _condition_key(conditions, feature_order)
        if key in seen_keys:
            continue
        value = objective(conditions)
        seen_keys.add(key)
        history.append((conditions, value))
        log_records.append(
            _log_record(
                experiment_index=len(log_records) + 1,
                source="initial_set",
                conditions=conditions,
                feature_order=feature_order,
                pred_mean=np.nan,
                pred_std=np.nan,
                oracle_value=value,
                batch_position=position,
            )
        )

    gp = None
    for batch_index in range(1, n_iter + 1):
        X_train = encoder.transform_many([conditions for conditions, _ in history])
        y_train = np.array([value for _, value in history], dtype=float)

        gp = _build_gp(random_state, kernel_name=kernel_name) if gp_builder is None else gp_builder(random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(X_train, y_train)

        candidates = _sample_conditions(
            search_space,
            feature_order,
            n_candidates,
            rng,
            seen_keys=seen_keys,
        )
        if not candidates:
            break

        X_candidates = encoder.transform_many(candidates)
        pred_mean, pred_std = gp.predict(X_candidates, return_std=True)
        available = list(range(len(candidates)))
        selected = []
        fantasy_gp = gp
        fantasy_X, fantasy_y = X_train.copy(), y_train.copy()
        size = min(batch_size, len(candidates))
        for position in range(size):
            scores = score_candidates(fantasy_gp, X_candidates[available], fantasy_y, rng)
            next_index = available.pop(int(np.argmax(scores)))
            selected.append(next_index)
            if position + 1 < size:
                # Kriging believer: condition on the predicted mean, without
                # querying the oracle or re-optimising kernel hyperparameters.
                point = X_candidates[next_index:next_index + 1]
                fantasy_value = float(fantasy_gp.predict(point)[0])
                fantasy_X = np.vstack([fantasy_X, point])
                fantasy_y = np.append(fantasy_y, fantasy_value)
                fantasy_gp = clone(gp).set_params(kernel=clone(gp.kernel_), optimizer=None)
                fantasy_gp.fit(fantasy_X, fantasy_y)

        # The entire batch is now fixed. These serial oracle calls simulate
        # experiments whose results become available together.
        for position, next_index in enumerate(selected, start=1):
            next_conditions = candidates[next_index]
            next_value = objective(next_conditions)
            seen_keys.add(_condition_key(next_conditions, feature_order))
            history.append((next_conditions, next_value))
            log_records.append(
                _log_record(
                    experiment_index=len(log_records) + 1,
                    source="bo",
                    conditions=next_conditions,
                    feature_order=feature_order,
                    pred_mean=float(pred_mean[next_index]),
                    pred_std=float(pred_std[next_index]),
                    oracle_value=next_value,
                    batch=batch_index,
                    batch_position=position,
                )
            )

    log_df = pd.DataFrame(log_records)
    best_index = int(np.argmax([value for _, value in history]))
    best_conditions, best_value = history[best_index]

    if save_logs:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        filename = log_name or f"bo_run_seed_{random_state}.csv"
        output_path = log_path / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(output_path, index=False)

    return BORunResult(
        target=target,
        seed=random_state,
        search_space=search_space,
        history=history,
        log=log_df,
        best_conditions=best_conditions,
        best_value=float(best_value),
        gp=gp,
    )


def run_multiple_bo(
    bundle: OracleBundle,
    *,
    target: str,
    search_space_override: Optional[Dict[str, Any]] = None,
    categorical_mode: str = "onehot",
    custom_categorical_maps: Optional[Dict[str, Dict[Any, Sequence[float]]]] = None,
    encoder_factory: Optional[Callable[[OracleProblemSpec, Dict[str, Any]], Any]] = None,
    initial_strategy: str = "random",
    initial_design_sampler: Optional[
        Callable[
            [OracleProblemSpec, Dict[str, Any], Sequence[str], int, np.random.Generator, Optional[set]],
            List[Dict[str, Any]],
        ]
    ] = None,
    kernel_name: str = "matern52",
    gp_builder: Optional[Callable[[Optional[int]], GaussianProcessRegressor]] = None,
    score_candidates: Optional[
        Callable[[GaussianProcessRegressor, np.ndarray, np.ndarray, Optional[np.random.Generator]], np.ndarray]
    ] = None,
    acquisition_function: str = "ei",
    ei_xi: float = 0.0,
    ucb_beta: float = 2.0,
    n_initial: int = 12,
    n_iter: int = 30,
    batch_size: int = 1,
    n_candidates: int = 2000,
    n_runs: int = 5,
    base_seed: Optional[int] = 42,
    save_logs: bool = False,
    log_dir: str = "bo_experiment_logs",
    log_name_template: str = "run_{run_index:02d}_experiment_log.csv",
) -> List[BORunResult]:
    if n_runs < 1:
        raise ValueError("N_RUNS must be at least 1")

    results = []
    for run_index in range(1, n_runs + 1):
        if base_seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        else:
            seed = base_seed + run_index - 1
        result = run_bo(
            bundle,
            target=target,
            search_space_override=search_space_override,
            categorical_mode=categorical_mode,
            custom_categorical_maps=custom_categorical_maps,
            encoder_factory=encoder_factory,
            initial_strategy=initial_strategy,
            initial_design_sampler=initial_design_sampler,
            kernel_name=kernel_name,
            gp_builder=gp_builder,
            score_candidates=score_candidates,
            acquisition_function=acquisition_function,
            ei_xi=ei_xi,
            ucb_beta=ucb_beta,
            n_initial=n_initial,
            n_iter=n_iter,
            batch_size=batch_size,
            n_candidates=n_candidates,
            random_state=seed,
            save_logs=save_logs,
            log_dir=log_dir,
            log_name=log_name_template.format(run_index=run_index, seed=seed),
        )
        results.append(result)
    return results


def best_so_far(log_df: pd.DataFrame, value_col: str = "oracle_value") -> np.ndarray:
    values = log_df[value_col].to_numpy(dtype=float)
    return np.maximum.accumulate(values)


def stack_best_so_far(run_results: Sequence[BORunResult]) -> np.ndarray:
    if not run_results:
        return np.empty((0, 0), dtype=float)

    curves = [best_so_far(result.log) for result in run_results]
    max_length = max(len(curve) for curve in curves)
    stacked = np.full((len(curves), max_length), np.nan, dtype=float)
    for index, curve in enumerate(curves):
        stacked[index, : len(curve)] = curve
    return stacked


def summarize_runs(run_results: Sequence[BORunResult]) -> pd.DataFrame:
    rows = []
    for run_index, result in enumerate(run_results, start=1):
        rows.append(
            {
                "run_index": run_index,
                "seed": result.seed,
                "target": result.target,
                "best_value": result.best_value,
                "n_experiments": len(result.log),
            }
        )
    return pd.DataFrame(rows)


def best_value_after_experiment(
    log_df: pd.DataFrame,
    experiment_index: int,
    value_col: str = "oracle_value",
) -> float:
    curve = best_so_far(log_df, value_col=value_col)
    if curve.size == 0:
        return float("nan")
    step = max(1, int(experiment_index))
    step = min(step, int(curve.size))
    return float(curve[step - 1])


def validate_comparison_setups(
    comparison_setups: Optional[Dict[str, Dict[str, Any]]],
    *,
    allowed_keys: Sequence[str] = COMPARISON_OVERRIDE_KEYS,
) -> Dict[str, Dict[str, Any]]:
    if comparison_setups is None:
        return {}
    if not isinstance(comparison_setups, dict):
        raise ValueError("COMPARISON_SETUPS must be a dictionary of named override dictionaries")

    allowed = set(allowed_keys)
    normalized: Dict[str, Dict[str, Any]] = {}

    for setup_name, overrides in comparison_setups.items():
        if not isinstance(setup_name, str) or not setup_name.strip():
            raise ValueError("Each comparison setup must have a non-empty string name")
        if not isinstance(overrides, dict):
            raise ValueError(f"Comparison setup '{setup_name}' must map to a dictionary of overrides")

        invalid_keys = [key for key in overrides if key not in allowed]
        if invalid_keys:
            raise ValueError(
                f"Comparison setup '{setup_name}' uses invalid override keys: {invalid_keys}. "
                f"Allowed keys are: {sorted(allowed)}"
            )

        cleaned_name = setup_name.strip()
        if cleaned_name in normalized:
            raise ValueError(f"Duplicate comparison setup name '{cleaned_name}'")
        normalized[cleaned_name] = deepcopy(overrides)

    return normalized


def merge_comparison_overrides(
    base_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = deepcopy(base_config)
    for key, value in (overrides or {}).items():
        merged[key] = deepcopy(value)
    return merged


def _format_changes_from_base(overrides: Dict[str, Any]) -> str:
    if not overrides:
        return "base settings"
    return ", ".join(f"{key}={value!r}" for key, value in overrides.items())


def _make_score_candidates_from_config(
    config: Dict[str, Any],
) -> Callable[[GaussianProcessRegressor, np.ndarray, np.ndarray, Optional[np.random.Generator]], np.ndarray]:
    explicit_scorer = config.get("SCORE_CANDIDATES")
    if explicit_scorer is not None:
        return explicit_scorer
    return make_default_acquisition_scorer(
        acquisition_function=str(config.get("ACQ_FUNCTION", "ei")),
        ei_xi=float(config.get("EI_XI", 0.0)),
        ucb_beta=float(config.get("UCB_BETA", 2.0)),
    )


def run_named_setup_comparisons(
    bundle: OracleBundle,
    *,
    target: str,
    base_config: Dict[str, Any],
    comparison_setups: Optional[Dict[str, Dict[str, Any]]] = None,
    base_setup_name: str = "main_setup",
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(base_setup_name, str) or not base_setup_name.strip():
        raise ValueError("base_setup_name must be a non-empty string")

    normalized_base_name = base_setup_name.strip()
    validated_setups = validate_comparison_setups(comparison_setups)
    if normalized_base_name in validated_setups:
        raise ValueError(
            f"COMPARISON_SETUPS already contains '{normalized_base_name}'. "
            "Choose a different base_setup_name or rename the comparison setup."
        )

    named_results: Dict[str, Dict[str, Any]] = {}
    all_setups = [(normalized_base_name, {})] + list(validated_setups.items())

    for setup_name, overrides in all_setups:
        config = merge_comparison_overrides(base_config, overrides)
        score_candidates = _make_score_candidates_from_config(config)
        save_logs = bool(config.get("SAVE_EXPERIMENT_LOGS", False))
        log_dir = str(Path(str(config.get("LOG_DIR", "bo_experiment_logs"))) / setup_name)

        run_results = run_multiple_bo(
            bundle,
            target=target,
            search_space_override=config.get("SEARCH_SPACE_OVERRIDE"),
            categorical_mode=str(config.get("ENCODING_MODE", "onehot")),
            custom_categorical_maps=config.get("CUSTOM_CATEGORICAL_MAPS"),
            initial_strategy=str(config.get("INIT_STRATEGY", "random")),
            kernel_name=str(config.get("KERNEL_NAME", "matern52")),
            score_candidates=score_candidates,
            n_initial=int(config.get("N_INITIAL", 12)),
            n_iter=int(config.get("N_ITER", 30)),
            batch_size=config.get("BATCH_SIZE", 1),
            n_candidates=int(config.get("N_CANDIDATES", 2000)),
            n_runs=int(config.get("N_RUNS", 5)),
            base_seed=config.get("BASE_SEED", 42),
            save_logs=save_logs,
            log_dir=log_dir,
            log_name_template="run_{run_index:02d}_seed_{seed}.csv",
        )

        named_results[setup_name] = {
            "settings": config,
            "overrides": deepcopy(overrides),
            "run_results": run_results,
            "run_summary": summarize_runs(run_results),
        }

    return named_results


def summarize_named_setup_comparisons(
    named_results: Dict[str, Dict[str, Any]],
    *,
    compare_after_experiment: Optional[int] = None,
) -> pd.DataFrame:
    if not named_results:
        columns = [
            "setup_name",
            "changes_from_base",
            "kernel_name",
            "acq_function",
            "encoding_mode",
            "n_initial",
            "n_iter",
            "batch_size",
            "max_experiment_budget",
            "n_runs",
            "mean_final_best",
            "std_final_best",
            "mean_final_regret",
        ]
        if compare_after_experiment is not None:
            columns.extend(
                [
                    "compare_after_experiment",
                    "mean_best_by_experiment",
                    "std_best_by_experiment",
                ]
            )
        return pd.DataFrame(columns=columns)

    global_best = max(
        result.best_value
        for payload in named_results.values()
        for result in payload["run_results"]
    )

    rows = []
    for setup_name, payload in named_results.items():
        settings = payload["settings"]
        run_results = payload["run_results"]
        final_bests = np.array([result.best_value for result in run_results], dtype=float)
        final_regrets = global_best - final_bests

        row = {
            "setup_name": setup_name,
            "changes_from_base": _format_changes_from_base(payload["overrides"]),
            "kernel_name": settings.get("KERNEL_NAME", "matern52"),
            "acq_function": settings.get("ACQ_FUNCTION", "ei"),
            "encoding_mode": settings.get("ENCODING_MODE", "onehot"),
            "n_initial": int(settings.get("N_INITIAL", 12)),
            "n_iter": int(settings.get("N_ITER", 30)),
            "batch_size": int(settings.get("BATCH_SIZE", 1)),
            "max_experiment_budget": int(settings.get("N_INITIAL", 12))
            + int(settings.get("N_ITER", 30)) * int(settings.get("BATCH_SIZE", 1)),
            "n_runs": int(settings.get("N_RUNS", len(run_results))),
            "mean_final_best": float(np.mean(final_bests)),
            "std_final_best": float(np.std(final_bests)),
            "mean_final_regret": float(np.mean(final_regrets)),
        }
        if compare_after_experiment is not None:
            values_at_step = np.array(
                [
                    best_value_after_experiment(result.log, compare_after_experiment)
                    for result in run_results
                ],
                dtype=float,
            )
            row["compare_after_experiment"] = int(compare_after_experiment)
            row["mean_best_by_experiment"] = float(np.mean(values_at_step))
            row["std_best_by_experiment"] = float(np.std(values_at_step))
        rows.append(row)

    return pd.DataFrame(rows)


def build_named_setup_plot_frame(
    named_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    if not named_results:
        return pd.DataFrame(
            columns=[
                "setup_name",
                "experiment_index",
                "mean_best",
                "std_best",
                "mean_regret",
                "std_regret",
            ]
        )

    global_best = max(
        result.best_value
        for payload in named_results.values()
        for result in payload["run_results"]
    )

    rows = []
    for setup_name, payload in named_results.items():
        curves = stack_best_so_far(payload["run_results"])
        if curves.size == 0:
            continue

        mean_best = np.nanmean(curves, axis=0)
        std_best = np.nanstd(curves, axis=0)
        regret_curves = global_best - curves
        mean_regret = np.nanmean(regret_curves, axis=0)
        std_regret = np.nanstd(regret_curves, axis=0)

        for experiment_index, (best_value, best_std, regret_value, regret_std) in enumerate(
            zip(mean_best, std_best, mean_regret, std_regret),
            start=1,
        ):
            rows.append(
                {
                    "setup_name": setup_name,
                    "experiment_index": experiment_index,
                    "mean_best": float(best_value),
                    "std_best": float(best_std),
                    "mean_regret": float(regret_value),
                    "std_regret": float(regret_std),
                }
            )

    return pd.DataFrame(rows)


__all__ = [
    "BORunResult",
    "COMPARISON_OVERRIDE_KEYS",
    "ConditionEncoder",
    "best_so_far",
    "best_value_after_experiment",
    "build_named_setup_plot_frame",
    "build_search_space",
    "make_default_acquisition_scorer",
    "merge_comparison_overrides",
    "run_bo",
    "run_multiple_bo",
    "run_named_setup_comparisons",
    "stack_best_so_far",
    "summarize_named_setup_comparisons",
    "summarize_runs",
    "validate_comparison_setups",
]
