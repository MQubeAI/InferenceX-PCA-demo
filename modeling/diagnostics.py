"""Aggregate-only diagnostics and deterministic TabFM context selection."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CONTEXT_CATEGORICAL = ("config_framework", "config_hardware", "config_model", "config_precision", "config_spec_method", "config_disagg", "config_is_multinode")
WORKLOAD_COLUMNS = ("isl", "osl", "conc")

def target_transform(y: pd.Series, name: str) -> pd.Series:
    if name == "raw": return y.astype(float)
    if name == "log1p":
        if (y < 0).any(): raise ValueError("log1p is only valid for nonnegative targets")
        return np.log1p(y.astype(float))
    raise ValueError(f"Unknown transform: {name}")

def inverse_target(values: np.ndarray, name: str) -> np.ndarray:
    return np.expm1(values) if name == "log1p" else values

def workload_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in WORKLOAD_COLUMNS:
        if col in frame:
            numeric = pd.to_numeric(frame[col], errors="coerce")
            out[f"{col}_bin"] = numeric.map(lambda x: "missing" if pd.isna(x) else f"{int(x)}")
    return out

def _values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        if col in frame: out[col] = frame[col].astype("string").fillna("__MISSING__")
    return out

def context_signature(index: pd.Index, strategy: str, seed: int) -> str:
    return hashlib.sha256(f"{strategy}:{seed}:".encode() + ",".join(map(str, sorted(index))).encode()).hexdigest()

def select_context(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], cap: int, strategy: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select only training rows; uses predictor values, never target labels."""
    cap = min(int(cap), len(train))
    if cap <= 0: raise ValueError("Context cap must be positive")
    rng = np.random.RandomState(seed)
    candidates = train.copy()
    categorical = [c for c in CONTEXT_CATEGORICAL if c in candidates]
    coverage = pd.concat([_values(candidates, categorical), workload_bins(candidates)], axis=1)
    if strategy == "random":
        chosen = candidates.sample(n=cap, random_state=seed)
    elif strategy == "stratified":
        strata = coverage.astype(str).agg("|".join, axis=1) if len(coverage.columns) else pd.Series("all", index=candidates.index)
        selected: list[Any] = []
        groups = sorted(strata.groupby(strata).groups.items(), key=lambda x: x[0])
        # Deterministic round-robin ensures every represented stratum gets a chance.
        queues = [sorted(list(indices), key=lambda x: str(x)) for _, indices in groups]
        while len(selected) < cap and any(queues):
            for queue in queues:
                if queue and len(selected) < cap: selected.append(queue.pop(0))
        chosen = candidates.loc[selected]
    elif strategy == "coverage":
        remaining = list(candidates.index)
        selected = []
        seen: set[tuple[str, str]] = set()
        while remaining and len(selected) < cap:
            def gain(idx: Any) -> tuple[int, str]:
                vals = coverage.loc[idx]
                return (sum((col, str(value)) not in seen for col, value in vals.items()), str(idx))
            best = max(remaining, key=gain)
            selected.append(best); remaining.remove(best)
            seen.update((col, str(value)) for col, value in coverage.loc[best].items())
        chosen = candidates.loc[selected]
    elif strategy == "nearest":
        numeric = [c for c in features if c in candidates and pd.api.types.is_numeric_dtype(candidates[c])]
        categorical_features = [c for c in features if c in candidates and c not in numeric]
        # Fold-local medians/scales; score against a deterministic, bounded validation panel.
        panel = valid.sort_index().iloc[:min(64, len(valid))]
        if numeric:
            medians = candidates[numeric].median(); scaler = StandardScaler().fit(candidates[numeric].fillna(medians))
            a = scaler.transform(candidates[numeric].fillna(medians)); b = scaler.transform(panel[numeric].fillna(medians))
            numeric_distance = ((a[:, None, :] - b[None, :, :]) ** 2).mean(axis=2)
        else: numeric_distance = np.zeros((len(candidates), len(panel)))
        if categorical_features:
            a = _values(candidates, categorical_features).to_numpy(); b = _values(panel, categorical_features).to_numpy()
            categorical_distance = (a[:, None, :] != b[None, :, :]).mean(axis=2)
        else: categorical_distance = np.zeros_like(numeric_distance)
        # Mean over a bounded panel prevents a few validation rows controlling all context.
        scores = (numeric_distance + categorical_distance).mean(axis=1)
        chosen = candidates.iloc[np.argsort(scores, kind="stable")[:cap]]
    else: raise ValueError(f"Unknown context strategy: {strategy}")
    chosen_coverage = pd.concat([_values(chosen, categorical), workload_bins(chosen)], axis=1)
    summary = {"strategy": strategy, "available_training_rows": len(train), "context_rows_used": len(chosen), "unique_groups": int(chosen["config_id"].nunique(dropna=False)), "categorical_coverage": {c: int(chosen_coverage[c].nunique()) for c in categorical if c in chosen_coverage}, "workload_bin_coverage": {c: int(chosen_coverage[c].nunique()) for c in chosen_coverage if c.endswith("_bin")}, "signature": context_signature(chosen.index, strategy, seed)}
    return chosen, summary

def distribution(series: pd.Series) -> dict[str, Any]:
    y = pd.to_numeric(series, errors="coerce").dropna()
    if not len(y):
        return {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "p1": np.nan, "p5": np.nan, "p25": np.nan, "p75": np.nan, "p95": np.nan, "p99": np.nan, "percentiles": {}, "skewness": np.nan, "outlier_rule": "1.5*IQR", "outlier_count": 0}
    q = y.quantile([.01, .05, .25, .5, .75, .95, .99]).to_dict()
    q1, q3 = q[0.25], q[0.75]
    iqr = q3-q1
    return {"count": int(len(y)), "mean": float(y.mean()), "median": float(y.median()), "std": float(y.std(ddof=0)), "min": float(y.min()), "max": float(y.max()), "p1": float(q[0.01]), "p5": float(q[0.05]), "p25": float(q1), "p75": float(q3), "p95": float(q[0.95]), "p99": float(q[0.99]), "percentiles": {str(k): float(v) for k,v in q.items()}, "skewness": float(y.skew()), "outlier_rule": "1.5*IQR", "outlier_count": int(((y < q1-1.5*iqr)|(y > q3+1.5*iqr)).sum())}

def category_distribution(series: pd.Series) -> list[dict[str, Any]]:
    """Full aggregate category counts, with deterministic ordering."""
    values = series.astype("string").fillna("__MISSING__")
    counts = values.value_counts(dropna=False).rename_axis("value").reset_index(name="count")
    counts["value"] = counts["value"].astype(str)
    counts = counts.sort_values(["count", "value"], ascending=[False, True], kind="stable")
    total = max(len(values), 1)
    return [{"value": str(row["value"]), "count": int(row["count"]), "fraction": float(row["count"] / total)} for row in counts.to_dict(orient="records")]

def error_attribution(valid: pd.DataFrame, target: str, prediction: np.ndarray, columns: list[str]) -> dict[str, Any]:
    """Aggregate residual attribution; predictions never leave process memory."""
    actual = pd.to_numeric(valid[target], errors="coerce").to_numpy(dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError("Prediction length must match validation rows.")
    residual = actual - predicted  # positive means the model underpredicted.
    absolute = np.abs(residual)
    q1, q3 = np.quantile(actual, [.25, .75])
    large_threshold = float(1.5 * (q3 - q1))
    large = absolute > large_threshold
    under = residual > 0
    over = residual < 0
    result: dict[str, Any] = {
        "residual_definition": "actual_minus_prediction",
        "large_residual_rule": "absolute residual > 1.5 * validation target IQR",
        "large_residual_threshold": large_threshold,
        "fold_totals": {
            "rows": int(len(valid)),
            "total_absolute_error": float(absolute.sum()),
            "mean_absolute_error": float(absolute.mean()),
            "large_residual_count": int(large.sum()),
            "underprediction_count": int(under.sum()),
            "underprediction_absolute_error": float(absolute[under].sum()),
            "overprediction_count": int(over.sum()),
            "overprediction_absolute_error": float(absolute[over].sum()),
        },
        "by_feature": {},
    }
    for col in columns:
        values = valid[col].astype("string").fillna("__MISSING__").astype(str)
        grouped = pd.DataFrame({"value": values, "absolute": absolute, "large": large, "under": under, "over": over})
        summary = grouped.groupby("value", sort=True).agg(
            rows=("absolute", "size"),
            total_absolute_error=("absolute", "sum"),
            mean_absolute_error=("absolute", "mean"),
            large_residual_count=("large", "sum"),
            underprediction_count=("under", "sum"),
            overprediction_count=("over", "sum"),
        )
        under_error = grouped.loc[grouped["under"]].groupby("value")["absolute"].sum()
        over_error = grouped.loc[grouped["over"]].groupby("value")["absolute"].sum()
        summary["underprediction_absolute_error"] = under_error.reindex(summary.index, fill_value=0.0)
        summary["overprediction_absolute_error"] = over_error.reindex(summary.index, fill_value=0.0)
        summary = summary.reset_index().sort_values(["total_absolute_error", "value"], ascending=[False, True], kind="stable")
        result["by_feature"][col] = [
            {key: (int(value) if key in {"rows", "large_residual_count", "underprediction_count", "overprediction_count"} else float(value) if key != "value" else str(value)) for key, value in row.items()}
            for row in summary.to_dict(orient="records")
        ]
    return result

def fold_difficulty(train: pd.DataFrame, valid: pd.DataFrame, target: str, prediction: np.ndarray | None = None) -> dict[str, Any]:
    cols = [c for c in (*CONTEXT_CATEGORICAL, *WORKLOAD_COLUMNS) if c in train]
    category = {}
    for col in cols:
        tr = train[col].astype("string").fillna("__MISSING__"); va = valid[col].astype("string").fillna("__MISSING__")
        counts = tr.value_counts(); validation_counts = va.value_counts()
        unseen = sorted(str(v) for v in set(va) - set(tr))
        rare = [{"value": str(value), "training_count": int(counts.get(value, 0)), "validation_count": int(validation_counts[value])} for value in sorted(validation_counts.index, key=str) if counts.get(value, 0) <= 2]
        category[col] = {"training_distribution": category_distribution(tr), "validation_distribution": category_distribution(va), "validation_unseen": unseen, "rare_validation": rare, "rare_validation_rule": "training count <= 2", "validation_coverage_in_training": float(va.isin(counts.index).mean())}
    result = {"train_target": distribution(train[target]), "validation_target": distribution(valid[target]), "category_coverage": category}
    if prediction is not None:
        result["error_attribution"] = error_attribution(valid, target, prediction, cols)
    return result

def known_config_folds(work: pd.DataFrame, folds: int, seed: int) -> list[tuple[np.ndarray,np.ndarray]]:
    """Deterministic interpolation split: each validation configuration remains in train."""
    rng = np.random.RandomState(seed); validation=[]
    for _, group in work.groupby("config_id", dropna=False, sort=True):
        if len(group) >= 2: validation.append(rng.choice(group.index.to_numpy()))
    validation = np.array(validation)
    chunks = np.array_split(validation[rng.permutation(len(validation))], min(folds, len(validation)))
    positions = {idx:i for i,idx in enumerate(work.index)}
    return [(np.array([i for i in range(len(work)) if i not in {positions[x] for x in chunk}]), np.array([positions[x] for x in chunk])) for chunk in chunks if len(chunk)]
