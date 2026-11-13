"""Churn prediction service: loads the model bundle saved by the notebook and scores new data.

Standalone copy of load_model() / predict_churn() from the notebook ("Model Deployment" section),
so the app does not need the notebook to run.
"""
import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.getenv("MODEL_DIR", "model_artifacts")
REGISTRY_PATH = os.path.join(MODEL_DIR, "registry.json")


def _load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"current": None, "versions": []}


@lru_cache(maxsize=2)
def load_model(version=None):
    """Load a bundle by version; default = 'current' in registry.json.
    MODEL_PATH env var (a .joblib file) overrides the registry."""
    explicit = os.getenv("MODEL_PATH")
    if explicit and version is None:
        return joblib.load(explicit)
    reg = _load_registry()
    version = version or reg["current"]
    if version is None:
        raise FileNotFoundError(
            f"No registered model in {REGISTRY_PATH}. Copy the model_artifacts/ folder "
            "from the notebook, or set MODEL_PATH to a .joblib file."
        )
    entry = next(v for v in reg["versions"] if v["version"] == version)
    path = entry["path"]
    if not os.path.exists(path):  # registry stores paths from the training machine
        path = os.path.join(MODEL_DIR, os.path.basename(path))
    return joblib.load(path)


def predict_churn(new_data, bundle=None):
    """Return churn_score, churn_probability_calibrated, churn_flag, risk_tier, warnings (list)."""
    bundle = bundle or load_model()
    df_new = pd.DataFrame(new_data).copy().drop(columns=["Churn"], errors="ignore")

    expected = bundle["feature_columns"]
    missing = [c for c in expected if c not in df_new.columns]
    extra = [c for c in df_new.columns if c not in expected]
    if missing or extra:
        raise ValueError(f"Columns must match training data. Missing: {missing} | Extra: {extra}")
    df_new = df_new[expected]
    df_new["PreferedOrderCat"] = df_new["PreferedOrderCat"].replace({"Mobile": "Mobile Phone"})

    warnings = []
    for col, rng in bundle["training_ranges"].items():
        out = df_new[col].notna() & ((df_new[col] < rng["min"]) | (df_new[col] > rng["max"]))
        if out.any():
            warnings.append(f"{int(out.sum())} row(s) of '{col}' outside training range [{rng['min']}, {rng['max']}]")
    for col, vals in bundle["category_values"].items():
        unseen = df_new[col].notna() & ~df_new[col].isin(vals)
        if unseen.any():
            warnings.append(f"{int(unseen.sum())} row(s) of '{col}' have unseen categories (ignored by encoder)")

    score = bundle["pipeline"].predict_proba(df_new)[:, 1]
    p = np.clip(score, 1e-6, 1 - 1e-6)
    cal = bundle["calibrator"].predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    out = pd.DataFrame({
        "churn_score": score,
        "churn_probability_calibrated": cal,
        "churn_flag": (score >= bundle["threshold"]).astype(int),
        "risk_tier": pd.cut(score, bins=bundle["risk_cuts"], labels=bundle["risk_labels"],
                            right=False, include_lowest=True).astype(str),
    }, index=df_new.index)
    out.attrs["warnings"] = warnings
    return out
