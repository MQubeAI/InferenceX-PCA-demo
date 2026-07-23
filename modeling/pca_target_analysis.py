"""Leakage-safe PCA and descriptive target overlays for the July 2026 snapshot.

The PCA basis is fitted only on the frozen configuration/workload feature schema.
Outcome metrics are joined to scores after fitting and never enter preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUTPUT_TARGET = "metrics_tput_per_gpu"
ENERGY_TARGET = "metrics_joules_per_output_token"
LATENCY_TARGET = "metrics_median_tpot"
OUTPUT_TARGET_LABEL = "Throughput per GPU"
OUTPUT_TARGET_UNIT = "tokens/second/GPU"
ENERGY_TARGET_LABEL = "Observed joules per output token"
ENERGY_TARGET_UNIT = "joules/output token"
LATENCY_TARGET_LABEL = "Median time per output token (TPOT)"
LATENCY_TARGET_UNIT = "seconds/output token"
SHARED_COHORT_FILTERS = {"benchmark_type": "single_turn"}
NUMERIC_FEATURES = (
    "isl",
    "osl",
    "conc",
    "config_prefill_tp",
    "config_prefill_ep",
    "config_prefill_dp_attention",
    "config_prefill_num_workers",
    "config_decode_tp",
    "config_decode_ep",
    "config_decode_dp_attention",
    "config_decode_num_workers",
    "config_num_prefill_gpu",
    "config_disagg",
    "config_is_multinode",
)
CATEGORICAL_FEATURES = (
    "config_hardware",
    "config_framework",
    "config_model",
    "config_precision",
    "config_spec_method",
)
# Preserve the original source-feature order even though boolean topology fields
# are normalized numerically, matching the June preprocessing behavior.
PCA_FEATURES = NUMERIC_FEATURES[:-2] + CATEGORICAL_FEATURES + NUMERIC_FEATURES[-2:]
OUTCOME_PREFIXES = ("metrics_",)
OUTCOME_TERMS = ("latency", "throughput", "power", "energy", "tput", "tpot", "ttft", "itl", "e2el")


@dataclass
class SharedPCAResult:
    cohort: pd.DataFrame
    scores: pd.DataFrame
    preprocessor: ColumnTransformer
    pca: PCA
    encoded_feature_names: list[str]
    source_features: list[str]


def validate_pca_feature_schema(features: list[str] | tuple[str, ...]) -> None:
    """Fail closed if an outcome or target-derived column enters PCA."""
    unexpected = [feature for feature in features if feature not in PCA_FEATURES]
    leakage = [
        feature
        for feature in features
        if feature.startswith(OUTCOME_PREFIXES) or any(term in feature.lower() for term in OUTCOME_TERMS)
    ]
    if unexpected or leakage:
        raise ValueError(
            "PCA inputs must use only the frozen configuration/workload schema: "
            + ", ".join(sorted(set(unexpected + leakage)))
        )
    if list(features) != list(PCA_FEATURES):
        raise ValueError("PCA feature order does not match the frozen schema.")


def shared_basis_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PCA_FEATURES if column not in frame]
    if missing:
        raise ValueError("Missing PCA features: " + ", ".join(missing))
    cohort = frame.copy()
    for column, expected in SHARED_COHORT_FILTERS.items():
        if column not in cohort:
            raise ValueError(f"Missing shared-cohort field: {column}")
        cohort = cohort.loc[cohort[column].eq(expected)]
    return cohort.reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, list(NUMERIC_FEATURES)), ("cat", categorical, list(CATEGORICAL_FEATURES))],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def normalize_pca_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dump representation while preserving the established feature semantics."""
    normalized = frame.copy()
    boolean_values = {True: 1.0, False: 0.0, "t": 1.0, "f": 0.0, "true": 1.0, "false": 0.0}
    for feature in NUMERIC_FEATURES:
        series = normalized[feature]
        if feature.endswith("dp_attention") or feature in {"config_disagg", "config_is_multinode"}:
            series = series.map(lambda value: boolean_values.get(value, value))
        normalized[feature] = pd.to_numeric(series, errors="coerce")
    for feature in CATEGORICAL_FEATURES:
        normalized[feature] = normalized[feature].astype("string")
    return normalized


def fit_shared_pca(frame: pd.DataFrame, seed: int = 42) -> SharedPCAResult:
    validate_pca_feature_schema(PCA_FEATURES)
    cohort = normalize_pca_inputs(shared_basis_cohort(frame))
    preprocessor = make_preprocessor()
    matrix = preprocessor.fit_transform(cohort[list(PCA_FEATURES)])
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise ValueError("The shared PCA cohort is too small.")
    pca = PCA(n_components=min(matrix.shape), random_state=seed)
    coordinates = pca.fit_transform(matrix)
    encoded = list(preprocessor.get_feature_names_out())
    score_columns = [f"PC{index + 1}" for index in range(coordinates.shape[1])]
    scores = pd.DataFrame(coordinates, columns=score_columns, index=cohort.index)
    return SharedPCAResult(
        cohort=cohort,
        scores=scores,
        preprocessor=preprocessor,
        pca=pca,
        encoded_feature_names=encoded,
        source_features=list(PCA_FEATURES),
    )


def source_feature_for_encoded(encoded_name: str) -> str:
    raw = encoded_name.split("__", 1)[-1]
    for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if raw == feature or raw.startswith(feature + "_"):
            return feature
    return raw


def explained_variance_table(result: SharedPCAResult) -> pd.DataFrame:
    values = result.pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(len(values))],
            "explained_variance_ratio": values,
            "cumulative_explained_variance": np.cumsum(values),
        }
    )


def component_thresholds(result: SharedPCAResult) -> dict[str, int]:
    cumulative = np.cumsum(result.pca.explained_variance_ratio_)
    return {
        f"{int(threshold * 100)}%": int(np.searchsorted(cumulative, threshold, side="left") + 1)
        for threshold in (0.50, 0.70, 0.80, 0.90, 0.95)
    }


def loading_table(result: SharedPCAResult, components: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component_index in range(min(components, len(result.pca.components_))):
        for encoded, loading in zip(
            result.encoded_feature_names, result.pca.components_[component_index], strict=True
        ):
            rows.append(
                {
                    "component": f"PC{component_index + 1}",
                    "encoded_feature": encoded,
                    "source_feature": source_feature_for_encoded(encoded),
                    "loading": float(loading),
                    "absolute_loading": abs(float(loading)),
                }
            )
    return pd.DataFrame(rows)


def source_loading_table(result: SharedPCAResult, components: int = 5) -> pd.DataFrame:
    details = loading_table(result, components)
    grouped = (
        details.groupby(["component", "source_feature"], as_index=False)
        .agg(
            signed_loading=("loading", "sum"),
            loading_magnitude=("absolute_loading", "sum"),
        )
    )
    return grouped.sort_values(["component", "loading_magnitude"], ascending=[True, False])


def target_overlay(result: SharedPCAResult, target: str, bins: int = 5) -> dict[str, Any]:
    if target not in result.cohort:
        raise ValueError(f"Target is unavailable: {target}")
    values = pd.to_numeric(result.cohort[target], errors="coerce")
    valid = values.notna() & np.isfinite(values)
    overlay = result.scores.loc[valid].copy()
    overlay[target] = values.loc[valid].astype(float)
    for optional in ("config_id", "benchmark_type", "date", *PCA_FEATURES):
        if optional in result.cohort:
            overlay[optional] = result.cohort.loc[valid, optional].to_numpy()

    associations = []
    component_bins: list[dict[str, Any]] = []
    for component in result.scores.columns[:5]:
        associations.append(
            {
                "component": component,
                "pearson": float(overlay[component].corr(overlay[target], method="pearson")),
                "spearman": float(overlay[component].corr(overlay[target], method="spearman")),
            }
        )
        quantiles = pd.qcut(overlay[component], q=bins, duplicates="drop")
        summary = (
            overlay.assign(component_bin=quantiles.astype(str))
            .groupby("component_bin", observed=True, sort=False)[target]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )
        for row in summary.to_dict("records"):
            component_bins.append({"component": component, **row})

    described = overlay[target].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    distribution = {key: float(value) for key, value in described.to_dict().items()}
    return {
        "frame": overlay,
        "usable_rows": int(len(overlay)),
        "unique_configurations": int(overlay["config_id"].nunique()) if "config_id" in overlay else None,
        "distribution": distribution,
        "associations": pd.DataFrame(associations),
        "component_bins": pd.DataFrame(component_bins),
    }


def preprocessing_state(result: SharedPCAResult) -> dict[str, Any]:
    numeric = result.preprocessor.named_transformers_["num"]
    categorical = result.preprocessor.named_transformers_["cat"]
    encoder = categorical.named_steps["encoder"]
    return {
        "feature_order": result.source_features,
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_imputer_statistics": numeric.named_steps["imputer"].statistics_.tolist(),
        "numeric_scaler_mean": numeric.named_steps["scaler"].mean_.tolist(),
        "numeric_scaler_scale": numeric.named_steps["scaler"].scale_.tolist(),
        "category_vocabularies": {
            feature: [str(value) for value in values.tolist()]
            for feature, values in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True)
        },
        "encoded_feature_names": result.encoded_feature_names,
        "pca_components": result.pca.components_.tolist(),
        "pca_mean": result.pca.mean_.tolist(),
    }


def compare_bases(old: SharedPCAResult, new: SharedPCAResult, components: int = 5) -> dict[str, Any]:
    union = sorted(set(old.encoded_feature_names) | set(new.encoded_feature_names))

    def aligned_matrix(result: SharedPCAResult) -> np.ndarray:
        positions = {name: index for index, name in enumerate(result.encoded_feature_names)}
        matrix = np.zeros((components, len(union)), dtype=float)
        for row in range(components):
            for column, name in enumerate(union):
                if name in positions:
                    matrix[row, column] = result.pca.components_[row, positions[name]]
        return matrix

    old_matrix, new_matrix = aligned_matrix(old), aligned_matrix(new)
    rows = []
    sign_aligned_new = new_matrix.copy()
    for index in range(components):
        left, right = old_matrix[index], new_matrix[index]
        sign = -1.0 if float(np.dot(left, right)) < 0 else 1.0
        sign_aligned_new[index] *= sign
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = float(np.dot(left, sign_aligned_new[index]) / denominator) if denominator else np.nan
        correlation = float(np.corrcoef(left, sign_aligned_new[index])[0, 1])
        rows.append(
            {
                "component": f"PC{index + 1}",
                "sign_alignment": int(sign),
                "loading_correlation": correlation,
                "cosine_similarity": cosine,
                "old_explained_variance": float(old.pca.explained_variance_ratio_[index]),
                "new_explained_variance": float(new.pca.explained_variance_ratio_[index]),
                "explained_variance_change": float(new.pca.explained_variance_ratio_[index] - old.pca.explained_variance_ratio_[index]),
            }
        )
    angles = np.degrees(subspace_angles(old_matrix.T, new_matrix.T)).tolist()
    singular_values = np.linalg.svd(old_matrix @ new_matrix.T, compute_uv=False).tolist()
    return {
        "components": pd.DataFrame(rows),
        "principal_angles_degrees": [float(value) for value in angles],
        "subspace_cosines": [float(value) for value in singular_values],
        "encoded_feature_union_count": len(union),
    }


def sparse_group_warnings(frame: pd.DataFrame, target: str, minimum_rows: int = 30) -> list[str]:
    warnings: list[str] = []
    for column in ("config_hardware", "config_framework", "config_model", "config_precision"):
        if column not in frame:
            continue
        counts = frame.loc[pd.to_numeric(frame[target], errors="coerce").notna(), column].value_counts()
        sparse = counts[counts < minimum_rows]
        if len(sparse):
            warnings.append(f"{column}: {len(sparse)} measured categories have fewer than {minimum_rows} rows")
    return warnings
