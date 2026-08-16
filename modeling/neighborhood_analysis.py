"""Algorithm-independent neighborhood and source-distance primitives.

The functions in this module deliberately operate on structural representations
only.  They neither fit UMAP/t-SNE nor accept outcome-driven selection criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors

from modeling.pca_target_analysis import (
    OUTCOME_PREFIXES,
    OUTCOME_TERMS,
    PCA_FEATURES,
    NUMERIC_FEATURES,
    validate_pca_feature_schema,
)
from modeling.representation_analysis import BOOLEAN_FEATURES


DEFAULT_NEIGHBORHOOD_K_VALUES = (10, 30, 50)
SENSITIVITY_NEIGHBORHOOD_K = 100
DEFAULT_GROUPED_BOOTSTRAP_REPLICATES = 200
DEFAULT_GROUPED_BOOTSTRAP_SEED = 42


def _reject_outcome_columns(columns: Iterable[Any], *, name: str) -> None:
    """Reject labelled outcome inputs before they can become a structural matrix.

    Canonical encoded matrices are NumPy arrays and carry their audited feature
    order separately.  A labelled matrix is inspectable, so accepting an outcome
    label here would be an avoidable structural-leakage route.
    """

    leakage = [
        str(column)
        for column in columns
        if str(column).startswith(OUTCOME_PREFIXES)
        or any(term in str(column).lower() for term in OUTCOME_TERMS)
    ]
    if leakage:
        raise ValueError(
            f"{name} must not contain outcome columns: " + ", ".join(sorted(set(leakage)))
        )


def _matrix(values: Any, *, name: str) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        _reject_outcome_columns(values.columns, name=name)
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional numeric matrix.")
    if not len(array):
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or infinite values.")
    return array


def _validate_k(k: int, reference_rows: int, *, self_excluded: bool) -> None:
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    available = reference_rows - int(self_excluded)
    if k > available:
        raise ValueError(
            f"k={k} is invalid for {reference_rows} reference rows "
            f"with self_excluded={self_excluded}; at most {available} neighbors are available."
        )


def exact_knn_indices(
    reference: Any,
    k: int,
    *,
    query: Any | None = None,
    metric: str = "euclidean",
    exclude_self: bool | None = None,
) -> np.ndarray:
    """Return deterministic exact/reference kNN indices.

    ``query=None`` means that every reference row queries itself, so its own
    index is removed.  For a distinct query set, self removal is disabled unless
    explicitly requested.  Exact brute-force sklearn search with a single worker
    avoids approximate-neighbor variation; returned neighbors are ordered by
    distance with reference index as the deterministic tie breaker where the
    backend returns the same candidates.
    """

    reference_array = _matrix(reference, name="reference")
    if query is None:
        query_array = reference_array
        same_rows = True
    else:
        query_array = _matrix(query, name="query")
        if query_array.shape[1] != reference_array.shape[1]:
            raise ValueError("query and reference must have the same number of columns.")
        # Treat equal aligned matrices as the same set even when callers passed
        # a copy, while allowing an explicit ``exclude_self=False`` override for
        # intentionally distinct query/reference semantics.
        same_rows = bool(
            query is reference
            or (
                query_array.shape == reference_array.shape
                and np.array_equal(query_array, reference_array)
            )
        )
    if exclude_self is None:
        exclude_self = same_rows
    if exclude_self and len(query_array) != len(reference_array):
        raise ValueError("Self-neighbor removal requires aligned query and reference rows.")
    _validate_k(k, len(reference_array), self_excluded=bool(exclude_self))

    requested = k + int(bool(exclude_self))
    model = NearestNeighbors(
        n_neighbors=requested,
        metric=metric,
        algorithm="brute",
        n_jobs=1,
    ).fit(reference_array)
    distances, indices = model.kneighbors(query_array, return_distance=True)
    output = np.empty((len(query_array), k), dtype=np.int64)
    for row, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        order = np.lexsort((row_indices, row_distances))
        ordered = row_indices[order]
        if exclude_self:
            ordered = ordered[ordered != row]
        if len(ordered) < k:
            raise RuntimeError("Exact kNN search could not remove the self neighbor safely.")
        output[row] = ordered[:k]
    return output


def _neighbor_graph(values: Any, *, name: str = "neighbor graph", allow_missing: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional non-empty integer array.")
    if allow_missing:
        valid = array[array >= 0]
    else:
        valid = array.ravel()
    if len(valid) and valid.max() >= len(array):
        raise ValueError(f"{name} contains an out-of-range row index.")
    if len(valid) and valid.min() < 0:
        raise ValueError(f"{name} contains negative neighbor indices.")
    for row, row_neighbors in enumerate(array):
        present = row_neighbors[row_neighbors >= 0] if allow_missing else row_neighbors
        if len(present) != len(set(present.tolist())):
            raise ValueError(f"{name} contains duplicate neighbor indices.")
        if row in present:
            raise ValueError(f"{name} contains a self neighbor at row {row}.")
    return array


def _agreement_inputs(reference_neighbors: Any, candidate_neighbors: Any) -> tuple[np.ndarray, np.ndarray]:
    reference = _neighbor_graph(reference_neighbors, name="reference_neighbors")
    candidate = _neighbor_graph(candidate_neighbors, name="candidate_neighbors")
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate neighbor graphs must have the same shape.")
    return reference, candidate


def neighbor_recall(reference_neighbors: Any, candidate_neighbors: Any) -> dict[str, Any]:
    """Compute per-row and aggregate reference-neighbor recall."""

    reference, candidate = _agreement_inputs(reference_neighbors, candidate_neighbors)
    k = reference.shape[1]
    per_row = np.array(
        [len(set(left) & set(right)) / k for left, right in zip(reference, candidate, strict=True)],
        dtype=float,
    )
    return {"k": k, "per_row": per_row, "mean": float(per_row.mean())}


def neighbor_jaccard(reference_neighbors: Any, candidate_neighbors: Any) -> dict[str, Any]:
    """Compute per-row and aggregate Jaccard agreement for ordered kNN sets."""

    reference, candidate = _agreement_inputs(reference_neighbors, candidate_neighbors)
    per_row = []
    for left, right in zip(reference, candidate, strict=True):
        union = set(left) | set(right)
        per_row.append(len(set(left) & set(right)) / len(union))
    values = np.asarray(per_row, dtype=float)
    return {"k": reference.shape[1], "per_row": values, "mean": float(values.mean())}


def local_rank_agreement(
    reference_neighbors: Any,
    candidate_neighbors: Any,
    *,
    minimum_shared_neighbors: int = 3,
) -> dict[str, Any]:
    """Compare ranks only among shared neighbors without hiding missing neighbors.

    The statistic is Spearman correlation between the rank positions (one-based)
    of neighbors shared by each row's two graphs.  Recall is intentionally not
    folded into this score.  A row with fewer than ``minimum_shared_neighbors``
    receives ``NaN`` and is reported as ineligible rather than a misleading zero.
    """

    if minimum_shared_neighbors < 2:
        raise ValueError("minimum_shared_neighbors must be at least 2.")
    reference, candidate = _agreement_inputs(reference_neighbors, candidate_neighbors)
    per_row = np.full(len(reference), np.nan, dtype=float)
    shared_counts = np.zeros(len(reference), dtype=np.int64)
    for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
        left_ranks = {value: rank for rank, value in enumerate(left, start=1)}
        right_ranks = {value: rank for rank, value in enumerate(right, start=1)}
        shared = sorted(set(left_ranks) & set(right_ranks))
        shared_counts[index] = len(shared)
        if len(shared) >= minimum_shared_neighbors:
            per_row[index] = float(
                spearmanr(
                    [left_ranks[value] for value in shared],
                    [right_ranks[value] for value in shared],
                ).statistic
            )
    eligible = np.isfinite(per_row)
    return {
        "k": reference.shape[1],
        "minimum_shared_neighbors": minimum_shared_neighbors,
        "per_row": per_row,
        "shared_neighbor_counts": shared_counts,
        "eligible_rows": int(eligible.sum()),
        "insufficient_shared_rows": int((~eligible).sum()),
        "mean": float(np.nanmean(per_row)) if eligible.any() else None,
    }


def trustworthiness_scores(
    reference: Any,
    candidate: Any,
    k_values: Iterable[int] = DEFAULT_NEIGHBORHOOD_K_VALUES,
    *,
    metric: str = "euclidean",
) -> dict[int, float]:
    """Return scikit-learn trustworthiness at each requested scale."""

    reference_array = _matrix(reference, name="reference")
    candidate_array = _matrix(candidate, name="candidate")
    if len(reference_array) != len(candidate_array):
        raise ValueError("reference and candidate must contain the same rows.")
    scores = {}
    for k in k_values:
        _validate_k(k, len(reference_array), self_excluded=True)
        if k >= len(reference_array) / 2:
            raise ValueError("trustworthiness requires k to be smaller than half the row count.")
        scores[k] = float(
            sklearn_trustworthiness(
                reference_array, candidate_array, n_neighbors=k, metric=metric
            )
        )
    return scores


def _validated_row_ids(row_ids: Iterable[Any], *, rows: int, name: str) -> list[str]:
    identifiers = [str(value) for value in row_ids]
    if len(identifiers) != rows:
        raise ValueError(f"{name} row IDs must be aligned one-for-one with its rows.")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} row IDs must be unique.")
    return identifiers


def _aligned_embedding_items(
    items: Sequence[tuple[Any, Any]],
    row_ids: Mapping[Any, Iterable[Any]] | None,
) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    """Reorder every representation to the first method's semantic row order."""

    if row_ids is None:
        raise ValueError("row_ids are required for cross-representation comparisons.")
    if not items:
        raise ValueError("At least one representation is required.")
    reference_name, reference_values = items[0]
    reference = _matrix(reference_values, name=f"embedding {reference_name}")
    try:
        reference_ids = _validated_row_ids(
            row_ids[reference_name], rows=len(reference), name=f"embedding {reference_name}"
        )
    except KeyError as exc:
        raise ValueError(f"Missing row IDs for embedding {reference_name}.") from exc
    reference_set = set(reference_ids)
    aligned = [(str(reference_name), reference)]
    for name, values in items[1:]:
        matrix = _matrix(values, name=f"embedding {name}")
        try:
            identifiers = _validated_row_ids(
                row_ids[name], rows=len(matrix), name=f"embedding {name}"
            )
        except KeyError as exc:
            raise ValueError(f"Missing row IDs for embedding {name}.") from exc
        if set(identifiers) != reference_set:
            raise ValueError("Cross-representation row-ID sets must match exactly.")
        positions = {identifier: index for index, identifier in enumerate(identifiers)}
        aligned.append((str(name), matrix[[positions[identifier] for identifier in reference_ids]]))
    return aligned, reference_ids


def cross_seed_neighborhood_stability(
    embeddings: Mapping[str | int, Any] | Sequence[Any],
    *,
    row_ids: Mapping[str | int, Iterable[Any]] | None = None,
    k_values: Iterable[int] = DEFAULT_NEIGHBORHOOD_K_VALUES,
    metric: str = "euclidean",
) -> dict[str, Any]:
    """Compare runs by neighbor graph agreement, independent of axis orientation."""

    requested_k = tuple(k_values)
    items = list(embeddings.items()) if isinstance(embeddings, Mapping) else list(enumerate(embeddings))
    items, aligned_row_ids = _aligned_embedding_items(items, row_ids)
    for k in requested_k:
        _validate_k(k, len(items[0][1]), self_excluded=True)
    if len(items) < 2:
        return {
            "runs": [str(name) for name, _ in items],
            "row_ids": aligned_row_ids,
            "pairs": [],
            "mean_jaccard": None,
        }
    row_count = len(items[0][1])
    graphs: dict[str, dict[int, np.ndarray]] = {}
    for name, embedding in items:
        matrix = _matrix(embedding, name=f"embedding {name}")
        if len(matrix) != row_count:
            raise ValueError("All embeddings must contain the same aligned rows.")
        graphs[str(name)] = {
            k: exact_knn_indices(matrix, k, metric=metric) for k in requested_k
        }
    pairs = []
    for left, right in combinations(graphs, 2):
        for k in graphs[left]:
            recall = neighbor_recall(graphs[left][k], graphs[right][k])
            jaccard = neighbor_jaccard(graphs[left][k], graphs[right][k])
            pairs.append(
                {
                    "left_run": left,
                    "right_run": right,
                    "k": k,
                    "neighbor_recall": recall["mean"],
                    "neighbor_jaccard": jaccard["mean"],
                }
            )
    return {
        "runs": list(graphs),
        "row_ids": aligned_row_ids,
        "pairs": pairs,
        "mean_jaccard": float(np.mean([row["neighbor_jaccard"] for row in pairs])),
    }


def consensus_neighbors(
    neighbor_graphs: Mapping[str, Any],
    *,
    row_ids: Mapping[str, Iterable[Any]] | None = None,
    include_tsne: bool = False,
    minimum_methods: int = 2,
) -> dict[str, Any]:
    """Summarize recurring neighbor identities across quantitative methods.

    t-SNE-labelled methods are excluded by default because Stage 5 treats t-SNE
    as visualization-only.  ``per_row`` retains occurrence counts so later work
    can ask which neighbors recur for a given row/configuration.
    """

    selected_raw = [
        (name, _neighbor_graph(graph, name=f"neighbor graph {name}"))
        for name, graph in neighbor_graphs.items()
        if include_tsne or "tsne" not in str(name).lower().replace("-", "")
    ]
    if row_ids is None:
        raise ValueError("row_ids are required for consensus-neighbor comparisons.")
    if not selected_raw:
        raise ValueError("No quantitative neighbor graphs were supplied.")
    reference_name, reference_graph = selected_raw[0]
    try:
        reference_ids = _validated_row_ids(
            row_ids[reference_name], rows=len(reference_graph), name=f"neighbor graph {reference_name}"
        )
    except KeyError as exc:
        raise ValueError(f"Missing row IDs for neighbor graph {reference_name}.") from exc
    reference_positions = {identifier: index for index, identifier in enumerate(reference_ids)}
    selected: dict[str, np.ndarray] = {str(reference_name): reference_graph}
    for name, graph in selected_raw[1:]:
        try:
            identifiers = _validated_row_ids(
                row_ids[name], rows=len(graph), name=f"neighbor graph {name}"
            )
        except KeyError as exc:
            raise ValueError(f"Missing row IDs for neighbor graph {name}.") from exc
        if set(identifiers) != set(reference_ids):
            raise ValueError("Consensus neighbor row-ID sets must match exactly.")
        positions = {identifier: index for index, identifier in enumerate(identifiers)}
        reordered = graph[[positions[identifier] for identifier in reference_ids]]
        selected[str(name)] = np.asarray(
            [
                [reference_positions[identifiers[neighbor]] for neighbor in row]
                for row in reordered
            ],
            dtype=np.int64,
        )
    if not selected:
        raise ValueError("No quantitative neighbor graphs were supplied.")
    shapes = {graph.shape for graph in selected.values()}
    if len(shapes) != 1:
        raise ValueError("Consensus neighbor graphs must have matching row and k dimensions.")
    if not isinstance(minimum_methods, (int, np.integer)) or isinstance(minimum_methods, bool):
        raise ValueError("minimum_methods must be a positive integer.")
    if minimum_methods < 1 or minimum_methods > len(selected):
        raise ValueError("minimum_methods must be between one and the number of methods.")
    rows: list[dict[str, int]] = []
    recurring: list[dict[str, int]] = []
    for row in range(next(iter(selected.values())).shape[0]):
        counts: dict[str, int] = {}
        for graph in selected.values():
            for neighbor in graph[row]:
                neighbor_id = reference_ids[int(neighbor)]
                counts[neighbor_id] = counts.get(neighbor_id, 0) + 1
        ordered = dict(sorted(counts.items()))
        rows.append(ordered)
        recurring.append(
            {neighbor: count for neighbor, count in ordered.items() if count >= minimum_methods}
        )
    return {
        "methods": list(selected),
        "row_ids": reference_ids,
        "k": next(iter(selected.values())).shape[1],
        "minimum_methods": minimum_methods,
        "per_row": rows,
        "recurring_neighbors": recurring,
    }


def cross_method_neighbor_agreement(
    representations: Mapping[str, Any],
    *,
    row_ids: Mapping[str, Iterable[Any]] | None = None,
    k: int,
    metric: str = "euclidean",
    include_tsne: bool = False,
) -> dict[str, Any]:
    """Build quantitative graphs, pairwise agreement, and recurrence summaries."""

    selected_items = [
        (name, value)
        for name, value in representations.items()
        if include_tsne or "tsne" not in name.lower().replace("-", "")
    ]
    if len(selected_items) < 2:
        raise ValueError("At least two quantitative representations are required.")
    aligned_items, aligned_row_ids = _aligned_embedding_items(selected_items, row_ids)
    graphs = {
        name: exact_knn_indices(value, k, metric=metric) for name, value in aligned_items
    }
    pairs = []
    for left, right in combinations(graphs, 2):
        pairs.append(
            {
                "left_method": left,
                "right_method": right,
                "neighbor_recall": neighbor_recall(graphs[left], graphs[right])["mean"],
                "neighbor_jaccard": neighbor_jaccard(graphs[left], graphs[right])["mean"],
            }
        )
    return {
        "row_ids": aligned_row_ids,
        "neighbor_graphs": graphs,
        "pairs": pairs,
        "consensus": consensus_neighbors(
            graphs,
            row_ids={name: aligned_row_ids for name in graphs},
        ),
    }


def grouped_bootstrap(
    per_row_values: Any,
    groups: Sequence[Any],
    *,
    statistic: Callable[[np.ndarray], float] | None = None,
    replicates: int = DEFAULT_GROUPED_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_GROUPED_BOOTSTRAP_SEED,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> dict[str, Any]:
    """Bootstrap complete groups, not individual rows, from precomputed values."""

    values = np.asarray(per_row_values)
    if values.ndim < 1 or not len(values):
        raise ValueError("per_row_values must have at least one row.")
    group_values = np.asarray(groups, dtype=object)
    if len(group_values) != len(values):
        raise ValueError("groups must be aligned one-for-one with per_row_values.")
    if pd.isna(group_values).any():
        raise ValueError("groups must not contain missing configuration IDs.")
    if replicates < 1:
        raise ValueError("replicates must be positive.")
    if not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("percentiles must be ordered values between 0 and 100.")
    unique = list(pd.unique(group_values))
    if not unique:
        raise ValueError("At least one group is required.")
    index_by_group = [np.flatnonzero(group_values == group) for group in unique]
    reducer = statistic or (lambda sample: float(np.nanmean(sample)))
    generator = np.random.default_rng(seed)
    estimates = []
    sampled_groups = []
    sampled_row_counts = []
    for _ in range(replicates):
        selected_positions = generator.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([index_by_group[position] for position in selected_positions])
        estimates.append(float(reducer(values[indices])))
        sampled_groups.append([unique[position] for position in selected_positions])
        sampled_row_counts.append(int(len(indices)))
    estimate_array = np.asarray(estimates, dtype=float)
    lower, upper = np.nanpercentile(estimate_array, percentiles)
    return {
        "groups": len(unique),
        "replicates": replicates,
        "seed": seed,
        "replicate_values": estimate_array,
        "percentile_interval": {
            "percentiles": list(percentiles),
            "lower": float(lower),
            "upper": float(upper),
        },
        "sampled_groups": sampled_groups,
        "sampled_row_counts": sampled_row_counts,
    }


def _workload_keys(workloads: Any) -> np.ndarray:
    if isinstance(workloads, pd.DataFrame):
        missing = [column for column in ("isl", "osl", "conc") if column not in workloads]
        if missing:
            raise ValueError("workloads is missing: " + ", ".join(missing))
        values = workloads[["isl", "osl", "conc"]].itertuples(index=False, name=None)
    else:
        values = workloads
    keys = []
    for value in values:
        if isinstance(value, np.ndarray):
            value = tuple(value.tolist())
        if not isinstance(value, tuple):
            try:
                value = tuple(value)
            except TypeError as exc:
                raise ValueError("Each workload identity must be an (isl, osl, conc) tuple.") from exc
        if len(value) != 3:
            raise ValueError("Each workload identity must contain isl, osl, and conc.")
        if any(pd.isna(part) for part in value):
            raise ValueError(
                "Exact workload identity requires non-missing isl, osl, and conc values."
            )
        keys.append(tuple(str(part) for part in value))
    result = np.empty(len(keys), dtype=object)
    result[:] = keys
    return result


def workload_neighborhood_purity(
    neighbors: Any,
    workloads: Any,
    *,
    k: int | None = None,
) -> dict[str, Any]:
    """Fraction of each row's neighbors in the exact same workload cell."""

    graph = _neighbor_graph(neighbors)
    keys = _workload_keys(workloads)
    if len(keys) != len(graph):
        raise ValueError("workloads must be aligned with the neighbor graph rows.")
    use_k = graph.shape[1] if k is None else int(k)
    if use_k < 1 or use_k > graph.shape[1]:
        raise ValueError("k must be between one and the graph width.")
    compared = graph[:, :use_k]
    per_row = np.array(
        [
            np.mean(
                [tuple(keys[row]) == tuple(keys[neighbor]) for neighbor in row_neighbors]
            )
            for row, row_neighbors in enumerate(compared)
        ],
        dtype=float,
    )
    return {
        "k": use_k,
        "per_row": per_row,
        "mean": float(per_row.mean()),
        "rows": len(graph),
        "neighbor_pairs": int(len(graph) * use_k),
        "matching_neighbor_pairs": int(per_row.sum() * use_k),
    }


def workload_cell_eligibility(workloads: Any, *, k: int) -> dict[str, Any]:
    """Report cells able to supply k within-workload neighbors (at least k + 1 rows)."""

    if k < 1:
        raise ValueError("k must be positive.")
    keys = _workload_keys(workloads)
    positions: dict[tuple[str, str, str], list[int]] = {}
    for index, key in enumerate(keys):
        positions.setdefault(tuple(key), []).append(index)
    eligible_cells = []
    excluded_cells = []
    eligible_mask = np.zeros(len(keys), dtype=bool)
    for key, rows in positions.items():
        descriptor = {"workload": list(key), "rows": len(rows)}
        if len(rows) >= k + 1:
            eligible_cells.append(descriptor)
            eligible_mask[rows] = True
        else:
            excluded_cells.append(
                {
                    **descriptor,
                    "reason": f"requires at least {k + 1} rows to supply {k} within-workload neighbors",
                }
            )
    return {
        "k": k,
        "eligible_mask": eligible_mask,
        "eligible_workload_cells": eligible_cells,
        "excluded_workload_cells": excluded_cells,
        "eligible_cells": len(eligible_cells),
        "excluded_cells": len(excluded_cells),
        "eligible_rows": int(eligible_mask.sum()),
        "excluded_rows": int((~eligible_mask).sum()),
    }


def within_workload_knn_indices(
    representation: Any,
    workloads: Any,
    *,
    k: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build local graphs only inside workload cells eligible at the requested k."""

    matrix = _matrix(representation, name="representation")
    keys = _workload_keys(workloads)
    if len(matrix) != len(keys):
        raise ValueError("representation and workloads must contain the same rows.")
    report = workload_cell_eligibility(keys, k=k)
    graph = np.full((len(matrix), k), -1, dtype=np.int64)
    for key in {tuple(key) for key in keys}:
        positions = np.flatnonzero(np.array([tuple(item) == key for item in keys]))
        if len(positions) < k + 1:
            continue
        local = exact_knn_indices(matrix[positions], k, metric=metric)
        graph[positions] = positions[local]
    return graph, report


def _pair_summary(
    neighbors: Any,
    values: Sequence[Any],
    *,
    kind: str,
) -> dict[str, Any]:
    graph = _neighbor_graph(neighbors, allow_missing=True)
    series = np.asarray(values, dtype=object)
    if len(series) != len(graph):
        raise ValueError("values must be aligned with the neighbor graph rows.")
    per_row = np.full(len(graph), np.nan, dtype=float)
    pair_values = []
    for row, row_neighbors in enumerate(graph):
        valid = [neighbor for neighbor in row_neighbors if neighbor >= 0]
        comparisons = []
        for neighbor in valid:
            left, right = series[row], series[neighbor]
            if pd.isna(left) or pd.isna(right):
                continue
            if kind == "categorical":
                comparisons.append(float(left == right))
            else:
                try:
                    comparisons.append(abs(float(left) - float(right)))
                except (TypeError, ValueError):
                    continue
        if comparisons:
            per_row[row] = float(np.mean(comparisons))
            pair_values.extend(comparisons)
    eligible = np.isfinite(per_row)
    return {
        "kind": kind,
        "per_row": per_row,
        "eligible_rows": int(eligible.sum()),
        "excluded_rows": int((~eligible).sum()),
        "eligible_neighbor_pairs": len(pair_values),
        "mean": float(np.mean(pair_values)) if pair_values else None,
    }


def categorical_neighbor_homophily(neighbors: Any, values: Sequence[Any]) -> dict[str, Any]:
    """Categorical same-value fraction over valid neighbor pairs."""

    return _pair_summary(neighbors, values, kind="categorical")


def numeric_neighbor_absolute_difference(neighbors: Any, values: Sequence[Any]) -> dict[str, Any]:
    """Mean absolute difference over valid numeric/discrete neighbor pairs."""

    return _pair_summary(neighbors, values, kind="numeric")


def within_workload_permutation_null(
    neighbors: Any,
    workloads: Any,
    values: Sequence[Any],
    *,
    value_kind: str,
    permutations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Test local configuration similarity against a workload-conditioned null."""

    if value_kind not in {"categorical", "numeric"}:
        raise ValueError("value_kind must be 'categorical' or 'numeric'.")
    if permutations < 1:
        raise ValueError("permutations must be positive.")
    graph = _neighbor_graph(neighbors, allow_missing=True)
    keys = _workload_keys(workloads)
    if len(keys) != len(graph):
        raise ValueError("workloads must be aligned with the neighbor graph rows.")
    for row, row_neighbors in enumerate(graph):
        for neighbor in row_neighbors[row_neighbors >= 0]:
            if tuple(keys[row]) != tuple(keys[neighbor]):
                raise ValueError("Within-workload permutation requires a within-workload neighbor graph.")
    summary = categorical_neighbor_homophily if value_kind == "categorical" else numeric_neighbor_absolute_difference
    observed = summary(graph, values)
    observed_value = observed["mean"]
    if observed_value is None:
        return {**observed, "null_values": [], "p_value": None, "permutations": permutations, "seed": seed}
    generator = np.random.default_rng(seed)
    original = np.asarray(values, dtype=object)
    cells: dict[tuple[str, str, str], np.ndarray] = {}
    for key in {tuple(key) for key in keys}:
        cells[key] = np.flatnonzero(np.array([tuple(item) == key for item in keys]))
    null_values = []
    for _ in range(permutations):
        shuffled = original.copy()
        for positions in cells.values():
            shuffled[positions] = original[positions][generator.permutation(len(positions))]
        value = summary(graph, shuffled)["mean"]
        if value is not None:
            null_values.append(value)
    null = np.asarray(null_values, dtype=float)
    more_extreme = null >= observed_value if value_kind == "categorical" else null <= observed_value
    return {
        **observed,
        "null_values": null,
        "null_mean": float(null.mean()) if len(null) else None,
        "null_standard_deviation": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "p_value": float((more_extreme.sum() + 1) / (len(null) + 1)) if len(null) else None,
        "alternative": "greater homophily" if value_kind == "categorical" else "smaller absolute difference",
        "permutations": permutations,
        "seed": seed,
    }


def within_workload_configuration_analysis(
    representation: Any,
    workloads: Any,
    values: Sequence[Any],
    *,
    k: int,
    value_kind: str,
    metric: str = "euclidean",
    permutations: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run an eligible-cell local configuration summary and conditioned null test.

    This convenience layer joins the reusable graph construction, explicit
    eligible/excluded-cell accounting, observed categorical/numeric statistic,
    and deterministic workload-conditioned permutation evidence.  It still makes
    no scientific interpretation of the returned values.
    """

    if value_kind not in {"categorical", "numeric"}:
        raise ValueError("value_kind must be 'categorical' or 'numeric'.")
    graph, eligibility = within_workload_knn_indices(
        representation, workloads, k=k, metric=metric
    )
    observed = (
        categorical_neighbor_homophily(graph, values)
        if value_kind == "categorical"
        else numeric_neighbor_absolute_difference(graph, values)
    )
    null = within_workload_permutation_null(
        graph,
        workloads,
        values,
        value_kind=value_kind,
        permutations=permutations,
        seed=seed,
    )
    return {
        "k": k,
        "value_kind": value_kind,
        "eligibility": eligibility,
        "observed": observed,
        "workload_conditioned_null": null,
    }


@dataclass(frozen=True)
class SourceMixedDistanceSpec:
    """Fitted source-level scales for explicit mixed-distance sensitivity work.

    Numeric differences are divided by the finite reference range and capped at
    one, preserving an equal [0, 1] contribution bound for every source field.
    A zero reference range contributes zero when both values are present; the
    explicit missing policy assigns zero to two missing values and one to a
    one-sided missing value.
    """

    feature_order: tuple[str, ...]
    numeric_ranges: Mapping[str, float]
    missing_policy: str = "both_missing_zero_one_missing_one"


def _validate_source_features(feature_order: Iterable[str]) -> tuple[str, ...]:
    ordered = tuple(feature_order)
    validate_pca_feature_schema(ordered)
    return ordered


def fit_source_mixed_distance_spec(
    frame: pd.DataFrame,
    *,
    feature_order: Iterable[str] = PCA_FEATURES,
) -> SourceMixedDistanceSpec:
    """Fit source-level numeric ranges while rejecting non-structural inputs."""

    features = _validate_source_features(feature_order)
    missing = [feature for feature in features if feature not in frame]
    if missing:
        raise ValueError("Missing frozen source features: " + ", ".join(missing))
    ranges = {}
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        span = float(finite.max() - finite.min()) if len(finite) else 0.0
        ranges[feature] = span
    return SourceMixedDistanceSpec(feature_order=features, numeric_ranges=ranges)


def source_mixed_distance_block(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    spec: SourceMixedDistanceSpec | None = None,
) -> np.ndarray:
    """Explicitly materialize one float32 query-by-reference mixed-distance block.

    This bounded block API is the normal path for 8,063 rows.  It never creates a
    full pairwise matrix unless the caller explicitly asks for one through
    ``materialize_source_mixed_distance_matrix``.
    """

    active_spec = spec or fit_source_mixed_distance_spec(reference)
    features = _validate_source_features(active_spec.feature_order)
    for name, frame in (("query", query), ("reference", reference)):
        missing = [feature for feature in features if feature not in frame]
        if missing:
            raise ValueError(f"{name} is missing frozen source features: " + ", ".join(missing))
    distances = np.zeros((len(query), len(reference)), dtype=np.float32)
    for feature in features:
        if feature in NUMERIC_FEATURES:
            left = pd.to_numeric(query[feature], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(reference[feature], errors="coerce").to_numpy(dtype=float)
            left_present = np.isfinite(left)
            right_present = np.isfinite(right)
            both_present = left_present[:, None] & right_present[None, :]
            both_missing = ~left_present[:, None] & ~right_present[None, :]
            contribution = np.ones((len(query), len(reference)), dtype=np.float32)
            contribution[both_missing] = 0.0
            if feature in BOOLEAN_FEATURES:
                left_grid = np.broadcast_to(left[:, None], contribution.shape)
                right_grid = np.broadcast_to(right[None, :], contribution.shape)
                contribution[both_present] = (
                    left_grid[both_present] != right_grid[both_present]
                )
            else:
                span = float(active_spec.numeric_ranges.get(feature, 0.0))
                if span == 0.0:
                    contribution[both_present] = 0.0
                else:
                    difference = np.abs(left[:, None] - right[None, :]) / span
                    contribution[both_present] = np.minimum(1.0, difference[both_present])
        else:
            left = query[feature].astype(object).to_numpy()
            right = reference[feature].astype(object).to_numpy()
            left_missing = pd.isna(left)
            right_missing = pd.isna(right)
            both_present = ~left_missing[:, None] & ~right_missing[None, :]
            both_missing = left_missing[:, None] & right_missing[None, :]
            contribution = np.ones((len(query), len(reference)), dtype=np.float32)
            contribution[both_missing] = 0.0
            equality = np.zeros((len(query), len(reference)), dtype=bool)
            left_grid = np.broadcast_to(left[:, None], equality.shape)
            right_grid = np.broadcast_to(right[None, :], equality.shape)
            equality[both_present] = (
                left_grid[both_present] == right_grid[both_present]
            )
            contribution[both_present] = (~equality[both_present]).astype(np.float32)
        distances += contribution
    return distances / len(features)


def source_mixed_knn_indices(
    frame: pd.DataFrame,
    k: int,
    *,
    spec: SourceMixedDistanceSpec | None = None,
    chunk_size: int = 256,
) -> np.ndarray:
    """Build source-level mixed kNN without an NxN distance allocation."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    active_spec = spec or fit_source_mixed_distance_spec(frame)
    _validate_k(k, len(frame), self_excluded=True)
    output = np.empty((len(frame), k), dtype=np.int64)
    reference_indices = np.arange(len(frame), dtype=np.int64)
    for start in range(0, len(frame), chunk_size):
        stop = min(len(frame), start + chunk_size)
        block = source_mixed_distance_block(frame.iloc[start:stop], frame, spec=active_spec)
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        # Reference-index tie breaking makes all-zero/duplicate rows reproducible.
        tie_break = block + reference_indices[None, :] * 1e-12
        candidates = np.argpartition(tie_break, kth=k - 1, axis=1)[:, :k]
        for local_row, candidate in enumerate(candidates):
            order = np.lexsort((candidate, block[local_row, candidate]))
            output[start + local_row] = candidate[order]
    return output


def mixed_distance_memory_bytes(rows: int, *, dtype: np.dtype[Any] = np.dtype("float32")) -> int:
    """Return the byte cost of an explicit dense square mixed-distance matrix."""

    if rows < 0:
        raise ValueError("rows must be non-negative.")
    return int(rows * rows * np.dtype(dtype).itemsize)


def materialize_source_mixed_distance_matrix(
    frame: pd.DataFrame,
    *,
    spec: SourceMixedDistanceSpec | None = None,
    chunk_size: int = 256,
    allow_large: bool = False,
    maximum_rows_without_opt_in: int = 2_000,
) -> np.ndarray:
    """Explicitly create a dense float32 matrix for tools requiring precomputed distance.

    At 8,063 rows this needs about 248 MiB before downstream library overhead.
    Callers must set ``allow_large=True`` above the protective row limit; normal
    validation and kNN construction never call this function implicitly.
    """

    if len(frame) > maximum_rows_without_opt_in and not allow_large:
        needed = mixed_distance_memory_bytes(len(frame)) / 1024**2
        raise ValueError(
            f"Dense mixed distance for {len(frame):,} rows needs about {needed:.1f} MiB; "
            "set allow_large=True to opt in explicitly."
        )
    active_spec = spec or fit_source_mixed_distance_spec(frame)
    result = np.empty((len(frame), len(frame)), dtype=np.float32)
    for start in range(0, len(frame), chunk_size):
        stop = min(len(frame), start + chunk_size)
        result[start:stop] = source_mixed_distance_block(frame.iloc[start:stop], frame, spec=active_spec)
    np.fill_diagonal(result, 0.0)
    return result
