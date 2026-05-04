"""Prompt variant comparison and sensitivity analysis."""

COGNITIVE_METRICS = [
    "branching_factor", "unique_perspective_count", "domain_spread", "first_idea_diversity",
    "max_elaboration_chain", "mean_branch_depth", "specificity_gradient", "reasoning_density",
    "exploration_exploitation_ratio", "backtracking_rate", "cross_branch_connectivity",
    "convergence_index", "graph_density", "revision_depth",
    "self_reflection_rate", "critique_to_hypothesis_ratio", "hedging_density", "perspective_taking",
    "token_per_idea", "redundancy_ratio",
]

METRIC_CATEGORIES: dict[str, list[str]] = {
    "Breadth":       ["branching_factor", "unique_perspective_count", "domain_spread", "first_idea_diversity"],
    "Depth":         ["max_elaboration_chain", "mean_branch_depth", "specificity_gradient", "reasoning_density"],
    "Structure":     ["exploration_exploitation_ratio", "backtracking_rate", "cross_branch_connectivity",
                      "convergence_index", "graph_density", "revision_depth"],
    "Metacognitive": ["self_reflection_rate", "critique_to_hypothesis_ratio", "hedging_density", "perspective_taking"],
    "Efficiency":    ["token_per_idea", "redundancy_ratio"],
}

CATEGORY_OF: dict[str, str] = {
    m: cat for cat, metrics in METRIC_CATEGORIES.items() for m in metrics
}


def compute_deltas(profiles_agg: list[dict], baseline: str = "pure") -> list[dict]:
    """For each (model, variant) pair compute per-metric delta vs baseline."""
    by_key = {(p["model"], p["prompt_variant"]): p for p in profiles_agg}
    results = []
    for (model, variant), profile in sorted(by_key.items()):
        if variant == baseline:
            continue
        base = by_key.get((model, baseline))
        if base is None:
            continue
        row = {"model": model, "variant": variant, "baseline": baseline}
        for m in COGNITIVE_METRICS:
            row[m] = profile.get(m, 0.0) - base.get(m, 0.0)
        results.append(row)
    return results


def prompt_sensitivity(profiles_agg: list[dict], baseline: str = "pure") -> list[dict]:
    """Rank metrics by magnitude of change from baseline to eliciting, averaged across models.

    Returns list sorted by abs_sensitivity descending.
    """
    by_key = {(p["model"], p["prompt_variant"]): p for p in profiles_agg}
    models = sorted({p["model"] for p in profiles_agg})

    rows = []
    for metric in COGNITIVE_METRICS:
        deltas_e, deltas_n = [], []
        for model in models:
            base = by_key.get((model, baseline), {})
            eliciting = by_key.get((model, "eliciting"), {})
            normal = by_key.get((model, "normal"), {})
            if base and eliciting:
                deltas_e.append(eliciting.get(metric, 0.0) - base.get(metric, 0.0))
            if base and normal:
                deltas_n.append(normal.get(metric, 0.0) - base.get(metric, 0.0))

        delta_e = sum(deltas_e) / len(deltas_e) if deltas_e else 0.0
        delta_n = sum(deltas_n) / len(deltas_n) if deltas_n else 0.0
        rows.append({
            "metric": metric,
            "category": CATEGORY_OF.get(metric, "Other"),
            "delta_eliciting": delta_e,
            "delta_normal": delta_n,
            "abs_sensitivity": abs(delta_e),
        })

    return sorted(rows, key=lambda x: x["abs_sensitivity"], reverse=True)


def classify_metrics(sensitivity: list[dict], top_n: int = 6) -> dict[str, list[str]]:
    """Split metrics into prompt-sensitive (top N by delta) and prompt-invariant (rest)."""
    return {
        "prompt_sensitive": [s["metric"] for s in sensitivity[:top_n]],
        "prompt_invariant": [s["metric"] for s in sensitivity[top_n:]],
    }
