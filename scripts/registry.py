from __future__ import annotations

from copy import deepcopy
from importlib import import_module


ORACLE_REGISTRY = {
    "dataset_a1": {
        "label": "Dataset_A1_Aspuru-Guzik_Suzuki_Numeric",
        "module": "scripts.dataset_a1",
        "default_target": "yield_percent",
        "targets": ["yield_percent"],
    },
    "dataset_b1": {
        "label": "Dataset_B1_Jensen_Case_I",
        "module": "scripts.dataset_b1",
        "default_target": "yield_pct",
        "targets": ["yield_pct", "ton"],
    },
    "dataset_b2": {
        "label": "Dataset_B2_Jensen_Case_II",
        "module": "scripts.dataset_b2",
        "default_target": "yield_pct",
        "targets": ["yield_pct", "ton"],
    },
    "dataset_b3": {
        "label": "Dataset_B3_Jensen_Case_III",
        "module": "scripts.dataset_b3",
        "default_target": "yield_pct",
        "targets": ["yield_pct", "ton"],
    },
    "dataset_b4": {
        "label": "Dataset_B4_Jensen_Case_IV",
        "module": "scripts.dataset_b4",
        "default_target": "yield",
        "targets": ["yield", "turnover"],
    },
    "dataset_c1": {
        "label": "Dataset_C1_Newman_BH_Plate_Replicates",
        "module": "scripts.dataset_c1",
        "default_target": "yield_pct",
        "targets": ["yield_pct"],
    },
    "dataset_d1": {
        "label": "Dataset_D1_Doyle_Direct_Arylation",
        "module": "scripts.dataset_d1",
        "default_target": "yield_pct",
        "targets": ["yield_pct"],
    },
    "dataset_e1": {
        "label": "Dataset_E1_Bourne_SNAr",
        "module": "scripts.dataset_e1",
        "default_target": "ortho_yield_pct",
        "targets": ["ortho_yield_pct", "para_yield_pct"],
    },
    "dataset_e2": {
        "label": "Dataset_E2_Bourne_Sonogashira",
        "module": "scripts.dataset_e2",
        "default_target": "rme_pct",
        "targets": ["rme_pct", "sty_kg_m3_h"],
    },
}


def get_oracle_meta(key: str):
    if key not in ORACLE_REGISTRY:
        raise KeyError(f"Unknown oracle key '{key}'")
    meta = deepcopy(ORACLE_REGISTRY[key])
    meta["key"] = key
    return meta


def list_oracles():
    return list(ORACLE_REGISTRY.keys())


def load_oracle(key: str, **build_kwargs):
    meta = get_oracle_meta(key)
    module = import_module(meta["module"])
    return module.build_oracle(**build_kwargs)


__all__ = [
    "ORACLE_REGISTRY",
    "get_oracle_meta",
    "list_oracles",
    "load_oracle",
]
