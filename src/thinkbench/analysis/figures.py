"""Comprehensive figure generation for ThinkBench prompt variant studies.

Figures are organized in four tiers:
  1. Univariate   — per-metric distributions (violin, histogram)
  2. Bivariate    — correlation heatmap, pairwise scatters, delta bar
  3. Multivariate — PCA, parallel coordinates, hierarchical clustering
  4. Graph        — NetworkX thought-graph drawings, node/edge type breakdowns
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np

from .compare import COGNITIVE_METRICS, METRIC_CATEGORIES, CATEGORY_OF

# ── shared aesthetics ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

VARIANT_COLOR = {"empty": "#A0AEC0", "pure": "#6B7280", "normal": "#2E86AB", "eliciting": "#E84855"}
VARIANT_LABEL = {
    "empty":     "Empty (no system prompt)",
    "pure":      "Pure (no guidance)",
    "normal":    "Normal (minimal)",
    "eliciting": "Eliciting (full framing)",
}
VARIANTS = ["empty", "pure", "normal", "eliciting"]

CATEGORY_COLOR = {
    "Breadth":       "#4C9BE8",
    "Depth":         "#2ECC71",
    "Structure":     "#F39C12",
    "Metacognitive": "#9B59B6",
    "Efficiency":    "#E74C3C",
}

NODE_FAMILY_COLOR = {
    "EXPLORATION": "#4C9BE8",
    "ELABORATION": "#2ECC71",
    "EVALUATION":  "#E74C3C",
    "CONVERGENCE": "#F39C12",
}
EDGE_COLOR = {
    "BRCH": "#4C9BE8",
    "ELAB": "#2ECC71",
    "BACK": "#E74C3C",
    "SYNT": "#F39C12",
    "CRIT": "#9B2335",
    "SUPP": "#82C27A",
    "CONT": "#E67E22",
    "SEQ":  "#CCCCCC",
}

_NORM_BOUNDS: dict[str, tuple[float, float]] = {
    "branching_factor":               (0, 0.5),
    "unique_perspective_count":       (0, 10),
    "domain_spread":                  (0, 10),
    "first_idea_diversity":           (0, 1),
    "max_elaboration_chain":          (0, 15),
    "mean_branch_depth":              (0, 10),
    "specificity_gradient":           (-0.05, 0.05),
    "reasoning_density":              (0, 1),
    "exploration_exploitation_ratio": (0, 5),
    "backtracking_rate":              (0, 1),
    "cross_branch_connectivity":      (0, 1),
    "convergence_index":              (0, 1),
    "graph_density":                  (0, 0.5),
    "revision_depth":                 (0, 30),
    "self_reflection_rate":           (0, 0.3),
    "critique_to_hypothesis_ratio":   (0, 3),
    "hedging_density":                (0, 1),
    "perspective_taking":             (0, 0.5),
    "token_per_idea":                 (0, 3000),
    "redundancy_ratio":               (0, 0.2),
}

METRIC_SHORT = {
    "branching_factor":               "Branch.Factor",
    "unique_perspective_count":       "Perspectives",
    "domain_spread":                  "Domain.Spread",
    "first_idea_diversity":           "1st.Diversity",
    "max_elaboration_chain":          "Max.Elab.Chain",
    "mean_branch_depth":              "Mean.Branch.D",
    "specificity_gradient":           "Spec.Gradient",
    "reasoning_density":              "Reas.Density",
    "exploration_exploitation_ratio": "Expl/Exploit",
    "backtracking_rate":              "Backtrack.Rate",
    "cross_branch_connectivity":      "Cross.Branch",
    "convergence_index":              "Convergence",
    "graph_density":                  "Graph.Density",
    "revision_depth":                 "Rev.Depth",
    "self_reflection_rate":           "Self.Reflect",
    "critique_to_hypothesis_ratio":   "CRT/HYP",
    "hedging_density":                "Hedging",
    "perspective_taking":             "Persp.Taking",
    "token_per_idea":                 "Tok/Idea",
    "redundancy_ratio":               "Redundancy",
}


def _normalize(value: float, metric: str) -> float:
    lo, hi = _NORM_BOUNDS.get(metric, (0, 1))
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _variant_patches():
    return [mpatches.Patch(color=VARIANT_COLOR[v], label=VARIANT_LABEL[v]) for v in VARIANTS]


# ── 1. SENSITIVITY BAR (fixed) ────────────────────────────────────────────────

def sensitivity_bar(sensitivity: list[dict], output_path: Path) -> None:
    """Dual horizontal bar chart — Δ(eliciting−pure) left, Δ(normal−pure) right.
    Sorted by absolute eliciting delta, category colour-coded."""
    metrics  = [s["metric"] for s in sensitivity]
    short    = [METRIC_SHORT.get(m, m) for m in metrics]
    deltas_e = [s["delta_eliciting"] for s in sensitivity]
    deltas_n = [s["delta_normal"]    for s in sensitivity]
    colors   = [CATEGORY_COLOR.get(CATEGORY_OF.get(m, ""), "#888") for m in metrics]

    n = len(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(16, max(10, n * 0.55)),
                             sharey=True, constrained_layout=True)

    for ax, deltas, title, xlab in [
        (axes[0], deltas_e, "Eliciting − Pure", "Δ value"),
        (axes[1], deltas_n, "Normal − Pure",    "Δ value"),
    ]:
        bars = ax.barh(range(n), deltas, color=colors,
                       edgecolor="white", linewidth=0.6, height=0.72)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xlabel(xlab, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="x", alpha=0.3)
        # value labels
        for bar, val in zip(bars, deltas):
            xpos = bar.get_width()
            ha = "left" if xpos >= 0 else "right"
            offset = 0.005 * (max(abs(d) for d in deltas) or 1)
            ax.text(xpos + (offset if xpos >= 0 else -offset), bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(short, fontsize=10)
    axes[0].invert_yaxis()

    # category legend
    cat_patches = [mpatches.Patch(facecolor=c, label=cat)
                   for cat, c in CATEGORY_COLOR.items()]
    fig.legend(handles=cat_patches, loc="lower center", ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9, title="Category", title_fontsize=10)

    fig.suptitle("Prompt Sensitivity — Δ vs Pure Baseline",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 2. RADAR (improved) ───────────────────────────────────────────────────────

def radar_comparison(profiles_agg: list[dict], output_path: Path) -> None:
    """Overlaid radar chart for all prompt variants — 20 cognitive metrics."""
    n = len(COGNITIVE_METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(13, 13), subplot_kw=dict(polar=True))

    for profile in sorted(profiles_agg, key=lambda p: p.get("prompt_variant", "")):
        v = profile.get("prompt_variant", "normal")
        col = VARIANT_COLOR.get(v, "#888")
        lbl = f"{VARIANT_LABEL.get(v, v)} (N={profile.get('num_traces', '?')})"
        vals = [_normalize(profile.get(m, 0.0), m) for m in COGNITIVE_METRICS]
        vals_plot = vals + [vals[0]]
        ax.plot(angles_plot, vals_plot, "o-", lw=2.2, color=col, label=lbl)
        ax.fill(angles_plot, vals_plot, alpha=0.10, color=col)

    short_labels = [METRIC_SHORT.get(m, m) for m in COGNITIVE_METRICS]
    ax.set_xticks(angles)
    ax.set_xticklabels(short_labels, size=9, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=7, color="gray")
    ax.set_rlabel_position(15)

    model = profiles_agg[0].get("model", "") if profiles_agg else ""
    ax.set_title(f"Cognitive Profile — Prompt Variant Comparison\n{model}",
                 size=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=11,
              framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 3. METRIC HEATMAP (improved) ──────────────────────────────────────────────

def metric_heatmap(profiles_agg: list[dict], output_path: Path) -> None:
    """Heatmap: metrics × prompt variants, normalized [0–1], coloured by category."""
    variants = [v for v in VARIANTS
                if any(p.get("prompt_variant") == v for p in profiles_agg)]
    by_v = {p["prompt_variant"]: p for p in profiles_agg}

    data = np.array([
        [_normalize(by_v.get(v, {}).get(m, 0.0), m) for v in variants]
        for m in COGNITIVE_METRICS
    ])

    fig, ax = plt.subplots(figsize=(max(7, len(variants) * 3), 13))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels([VARIANT_LABEL.get(v, v) for v in variants], fontsize=11)
    ax.set_yticks(range(len(COGNITIVE_METRICS)))
    ax.set_yticklabels(COGNITIVE_METRICS, fontsize=9)

    for i, m in enumerate(COGNITIVE_METRICS):
        cat = CATEGORY_OF.get(m, "")
        ax.get_yticklabels()[i].set_color(CATEGORY_COLOR.get(cat, "black"))

    for i in range(len(COGNITIVE_METRICS)):
        for j in range(len(variants)):
            val = data[i, j]
            tc = "black" if 0.2 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=tc)

    plt.colorbar(im, ax=ax, label="Normalized value [0–1]", shrink=0.5, pad=0.02)
    ax.set_title("Cognitive Metrics × Prompt Variant (normalized)", fontsize=13, fontweight="bold")

    cat_patches = [mpatches.Patch(facecolor=c, label=cat)
                   for cat, c in CATEGORY_COLOR.items()]
    fig.legend(handles=cat_patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.45, -0.04))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 4. CATEGORY COMPARISON (improved) ────────────────────────────────────────

def category_comparison(profiles_agg: list[dict], output_path: Path) -> None:
    """Grouped bar chart: mean normalised score per cognitive category."""
    variants = [v for v in VARIANTS
                if any(p.get("prompt_variant") == v for p in profiles_agg)]
    by_v = {p["prompt_variant"]: p for p in profiles_agg}
    cats = list(METRIC_CATEGORIES.keys())

    x = np.arange(len(cats))
    width = 0.8 / len(variants)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, v in enumerate(variants):
        profile = by_v.get(v, {})
        means = [
            float(np.mean([_normalize(profile.get(m, 0.0), m)
                           for m in METRIC_CATEGORIES[cat]]))
            for cat in cats
        ]
        offset = (i - len(variants) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width * 0.9,
                      label=VARIANT_LABEL.get(v, v),
                      color=VARIANT_COLOR.get(v, "#888"), edgecolor="white")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_ylabel("Mean normalised score [0–1]", fontsize=11)
    ax.set_title("Cognitive Category Scores by Prompt Variant", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 5. METRIC VIOLIN (univariate distributions) ───────────────────────────────

def metric_violin(profiles_per_trace: list[dict], output_path: Path) -> None:
    """5×4 grid: violin + strip plot for each cognitive metric × prompt variant."""
    n_cols, n_rows = 4, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 22), constrained_layout=True)

    by_v: dict[str, list[float]]
    for idx, metric in enumerate(COGNITIVE_METRICS):
        ax = axes[idx // n_cols][idx % n_cols]
        data_by_v = {v: [p[metric] for p in profiles_per_trace
                         if p.get("prompt_variant") == v]
                     for v in VARIANTS}
        vals_list = [data_by_v.get(v, [0]) for v in VARIANTS]

        n_v = len(VARIANTS)
        parts = ax.violinplot(vals_list, positions=range(n_v),
                              showmedians=True, showextrema=False, widths=0.7)
        for i, (pc, v) in enumerate(zip(parts["bodies"], VARIANTS)):
            pc.set_facecolor(VARIANT_COLOR[v])
            pc.set_alpha(0.55)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # strip (jitter)
        rng = np.random.default_rng(42)
        for i, (vals, v) in enumerate(zip(vals_list, VARIANTS)):
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=VARIANT_COLOR[v], s=12, alpha=0.55, zorder=3)

        short_labels = [v[:3].upper() for v in VARIANTS]
        ax.set_xticks(range(n_v))
        ax.set_xticklabels(short_labels, fontsize=9)
        short = METRIC_SHORT.get(metric, metric)
        cat = CATEGORY_OF.get(metric, "")
        ax.set_title(f"{short}", fontsize=10, fontweight="bold",
                     color=CATEGORY_COLOR.get(cat, "black"))
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Per-trace Metric Distributions by Prompt Variant  (P=pure  N=normal  E=eliciting)",
                 fontsize=14, fontweight="bold")
    fig.legend(handles=_variant_patches(), loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.015))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 6. CORRELATION HEATMAP (bivariate) ────────────────────────────────────────

def correlation_heatmap(profiles_per_trace: list[dict], output_path: Path) -> None:
    """Spearman ρ matrix of all 20 cognitive metrics, hierarchically clustered."""
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list

    n = len(COGNITIVE_METRICS)
    mat = np.array([[p[m] for m in COGNITIVE_METRICS] for p in profiles_per_trace])

    rho = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                rho[i, j] = 1.0
            else:
                r, _ = scipy_stats.spearmanr(mat[:, i], mat[:, j])
                rho[i, j] = r if not np.isnan(r) else 0.0

    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    order = leaves_list(Z)

    rho_ord = rho[np.ix_(order, order)]
    labels_ord = [METRIC_SHORT.get(COGNITIVE_METRICS[i], COGNITIVE_METRICS[i]) for i in order]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(rho_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.7, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_ord, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_ord, fontsize=9)

    # annotate cells with ρ value
    for i in range(n):
        for j in range(n):
            val = rho_ord[i, j]
            tc = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=tc)

    ax.set_title("Spearman Correlation Matrix — Cognitive Metrics\n(Ward-clustered order)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 7. PCA SCATTER (multivariate) ─────────────────────────────────────────────

def pca_traces(profiles_per_trace: list[dict], output_path: Path) -> None:
    """PCA of 90 traces in 20-dim metric space — PC1 vs PC2 and PC1 vs PC3."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    mat = np.array([[p[m] for m in COGNITIVE_METRICS] for p in profiles_per_trace])
    variants_list = [p.get("prompt_variant", "unknown") for p in profiles_per_trace]

    scaler = StandardScaler()
    mat_z = scaler.fit_transform(mat)

    pca = PCA(n_components=min(6, mat_z.shape[1]))
    coords = pca.fit_transform(mat_z)
    ev = pca.explained_variance_ratio_ * 100

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for pi, (xi, yi) in enumerate(pairs):
        ax = fig.add_subplot(1, 3, pi + 1)
        for v in VARIANTS:
            mask = [i for i, vv in enumerate(variants_list) if vv == v]
            x = coords[mask, xi]
            y = coords[mask, yi]
            ax.scatter(x, y, color=VARIANT_COLOR[v], label=VARIANT_LABEL.get(v, v),
                       s=55, alpha=0.72, edgecolors="white", linewidths=0.5)
            # centroid
            ax.scatter(x.mean(), y.mean(), color=VARIANT_COLOR[v],
                       s=220, marker="*", edgecolors="black", linewidths=0.8, zorder=5)

        ax.set_xlabel(f"PC{xi+1} ({ev[xi]:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC{yi+1} ({ev[yi]:.1f}% var)", fontsize=11)
        ax.set_title(f"PC{xi+1} vs PC{yi+1}", fontsize=12, fontweight="bold")
        if pi == 0:
            ax.legend(fontsize=8, framealpha=0.9, markerscale=1.2)

    fig.suptitle("PCA of Cognitive Profiles — 90 Traces in 20-Metric Space\n"
                 "Stars = variant centroids",
                 fontsize=13, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 8. PARALLEL COORDINATES (multivariate) ───────────────────────────────────

def parallel_coordinates(profiles_per_trace: list[dict], output_path: Path) -> None:
    """Parallel coordinates plot — all 20 metrics, lines coloured by variant."""
    from sklearn.preprocessing import MinMaxScaler

    mat = np.array([[p[m] for m in COGNITIVE_METRICS] for p in profiles_per_trace])
    variants_list = [p.get("prompt_variant", "unknown") for p in profiles_per_trace]
    short = [METRIC_SHORT.get(m, m) for m in COGNITIVE_METRICS]

    scaler = MinMaxScaler()
    mat_n = scaler.fit_transform(mat)

    fig, ax = plt.subplots(figsize=(22, 7), constrained_layout=True)
    x = np.arange(len(COGNITIVE_METRICS))

    for i, (row, v) in enumerate(zip(mat_n, variants_list)):
        ax.plot(x, row, color=VARIANT_COLOR.get(v, "#888"), alpha=0.22, linewidth=0.9)

    # variant medians (thicker)
    for v in VARIANTS:
        mask = [i for i, vv in enumerate(variants_list) if vv == v]
        median = np.median(mat_n[mask], axis=0)
        ax.plot(x, median, color=VARIANT_COLOR[v], linewidth=3.0, alpha=0.95,
                label=VARIANT_LABEL.get(v, v), zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("Min-max normalised value", fontsize=11)
    ax.set_title("Parallel Coordinates — Cognitive Metrics per Trace\n"
                 "Thick lines = variant medians",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)

    # category zone backgrounds
    cat_boundaries = {}
    current_cat = None
    start = 0
    for i, m in enumerate(COGNITIVE_METRICS):
        cat = CATEGORY_OF.get(m, "")
        if cat != current_cat:
            if current_cat is not None:
                cat_boundaries[current_cat] = (start, i - 1)
            current_cat = cat
            start = i
    if current_cat:
        cat_boundaries[current_cat] = (start, len(COGNITIVE_METRICS) - 1)

    for cat, (lo, hi) in cat_boundaries.items():
        ax.axvspan(lo - 0.4, hi + 0.4, alpha=0.06,
                   color=CATEGORY_COLOR.get(cat, "#eee"), zorder=0)
        ax.text((lo + hi) / 2, 1.055, cat, ha="center", fontsize=8,
                color=CATEGORY_COLOR.get(cat, "#555"), fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 9. HIERARCHICAL CLUSTERING DENDROGRAM ─────────────────────────────────────

def cluster_dendrogram(profiles_per_trace: list[dict], output_path: Path) -> None:
    """Ward dendrogram of 90 traces in metric space, coloured by variant."""
    from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
    from sklearn.preprocessing import StandardScaler

    mat = np.array([[p[m] for m in COGNITIVE_METRICS] for p in profiles_per_trace])
    variants_list = [p.get("prompt_variant", "unknown") for p in profiles_per_trace]
    scaler = StandardScaler()
    mat_z = scaler.fit_transform(mat)

    Z = linkage(mat_z, method="ward", metric="euclidean")

    fig, ax = plt.subplots(figsize=(22, 6), constrained_layout=True)
    ddata = scipy_dendrogram(Z, ax=ax, color_threshold=0, above_threshold_color="#AAAAAA",
                              no_labels=True, link_color_func=lambda _: "#AAAAAA")

    # colour leaf tick marks by variant
    for i, leaf_idx in enumerate(ddata["leaves"]):
        v = variants_list[leaf_idx]
        ax.get_xticklabels()  # ensure labels exist
        line = ax.axvline(x=i * 10 + 5, ymin=0, ymax=0.02,
                          color=VARIANT_COLOR.get(v, "#888"), lw=3, alpha=0.85)

    ax.set_xlabel("Trace index (sorted by Ward linkage)", fontsize=11)
    ax.set_ylabel("Ward distance", fontsize=11)
    ax.set_title("Hierarchical Clustering of 90 Traces — 20-Metric Space\n"
                 "Bottom tick colour = prompt variant",
                 fontsize=13, fontweight="bold")
    ax.legend(handles=_variant_patches(), fontsize=10, framealpha=0.9)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 10. SCATTER PAIRS (key bivariate) ─────────────────────────────────────────

def scatter_key_pairs(profiles_per_trace: list[dict], output_path: Path) -> None:
    """3×3 grid of scatter plots for the most informative metric pairs."""
    PAIRS = [
        ("branching_factor",           "revision_depth"),
        ("branching_factor",           "unique_perspective_count"),
        ("unique_perspective_count",   "convergence_index"),
        ("backtracking_rate",          "critique_to_hypothesis_ratio"),
        ("mean_branch_depth",          "max_elaboration_chain"),
        ("hedging_density",            "reasoning_density"),
        ("graph_density",              "exploration_exploitation_ratio"),
        ("perspective_taking",         "first_idea_diversity"),
        ("token_per_idea",             "avg_tus"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    for idx, (mx, my) in enumerate(PAIRS):
        ax = axes[idx // 3][idx % 3]
        for v in VARIANTS:
            xs = [p[mx] for p in profiles_per_trace if p.get("prompt_variant") == v]
            ys = [p[my] for p in profiles_per_trace if p.get("prompt_variant") == v]
            ax.scatter(xs, ys, color=VARIANT_COLOR[v], s=30, alpha=0.65,
                       edgecolors="white", linewidths=0.4,
                       label=v if idx == 0 else "")
        ax.set_xlabel(METRIC_SHORT.get(mx, mx), fontsize=9)
        ax.set_ylabel(METRIC_SHORT.get(my, my), fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle("Key Metric Pair Scatters — coloured by prompt variant",
                 fontsize=13, fontweight="bold")
    fig.legend(handles=_variant_patches(), loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 11. NODE / EDGE TYPE DISTRIBUTIONS ───────────────────────────────────────

def node_edge_distributions(graphs_dir: Path, tid_variant: dict[str, str],
                             output_path: Path) -> None:
    """Stacked bar charts: node-type and edge-type counts per prompt variant."""
    from thinkbench.extract.schemas import ThoughtGraph

    node_counts: dict[str, dict[str, int]] = {v: defaultdict(int) for v in VARIANTS}
    edge_counts: dict[str, dict[str, int]] = {v: defaultdict(int) for v in VARIANTS}
    trace_counts: dict[str, int]           = defaultdict(int)

    for gf in sorted(graphs_dir.glob("*.json")):
        with open(gf) as f:
            g = ThoughtGraph(**json.load(f))
        v = tid_variant.get(g.trace_id, "unknown")
        if v not in VARIANTS:
            continue
        trace_counts[v] += 1
        for node in g.nodes:
            node_counts[v][node.node_type.value] += 1
        for edge in g.edges:
            edge_counts[v][edge.edge_type.value] += 1

    NODE_TYPES = ["HYP", "RFR", "BRS", "ANA", "SPC", "JUS", "IMP", "CON",
                  "CRT", "CMP", "MET", "SYN"]
    EDGE_TYPES = ["SEQ", "ELAB", "BRCH", "BACK", "SYNT", "SUPP", "CONT", "CRIT"]

    NODE_COLORS = plt.cm.tab20(np.linspace(0, 1, len(NODE_TYPES)))
    EDGE_COLORS_LIST = [EDGE_COLOR.get(et, "#888") for et in EDGE_TYPES]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    for ax, type_list, counts_dict, colors_list, title in [
        (axes[0], NODE_TYPES, node_counts, NODE_COLORS, "Node Type Distribution"),
        (axes[1], EDGE_TYPES, edge_counts, EDGE_COLORS_LIST, "Edge Type Distribution"),
    ]:
        bottoms = np.zeros(len(VARIANTS))
        for ti, t in enumerate(type_list):
            vals = []
            for v in VARIANTS:
                n_traces = max(trace_counts[v], 1)
                vals.append(counts_dict[v].get(t, 0) / n_traces)
            bars = ax.bar(VARIANTS, vals, bottom=bottoms,
                          color=colors_list[ti] if hasattr(colors_list[ti], "__len__") else colors_list[ti],
                          label=t, edgecolor="white", linewidth=0.4)
            bottoms += np.array(vals)

        ax.set_xlabel("Prompt variant", fontsize=11)
        ax.set_ylabel("Mean count per trace", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels([VARIANT_LABEL.get(v, v) for v in VARIANTS], fontsize=9)

    fig.suptitle("Thought-Graph Composition by Prompt Variant",
                 fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 12. PER-TRACE HEATMAP (supplement) ───────────────────────────────────────

def per_trace_profile_heatmap(profiles_per_trace: list[dict], output_path: Path) -> None:
    """Each of the 90 traces as a row in a heatmap — grouped by variant."""
    groups = {v: [p for p in profiles_per_trace if p.get("prompt_variant") == v]
              for v in VARIANTS}
    ordered = []
    boundaries = []
    pos = 0
    for v in VARIANTS:
        ordered.extend(groups[v])
        boundaries.append((pos, pos + len(groups[v]) - 1, v))
        pos += len(groups[v])

    mat = np.array([[_normalize(p.get(m, 0.0), m) for m in COGNITIVE_METRICS]
                    for p in ordered])
    short = [METRIC_SHORT.get(m, m) for m in COGNITIVE_METRICS]

    fig, ax = plt.subplots(figsize=(22, max(12, len(ordered) * 0.22)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(COGNITIVE_METRICS)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel("Traces (grouped by variant)", fontsize=11)

    # variant labels and dividers
    for lo, hi, v in boundaries:
        mid = (lo + hi) / 2
        ax.axhline(lo - 0.5, color="black", lw=1.5)
        ax.text(-1.2, mid, VARIANT_LABEL.get(v, v),
                va="center", ha="right", fontsize=9,
                color=VARIANT_COLOR.get(v, "black"), fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalised value [0–1]", shrink=0.4, pad=0.01)
    ax.set_title("Per-trace Cognitive Profile Heatmap — All 90 Traces",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 13. GRAPH EXAMPLES (3 × 3 grid) ──────────────────────────────────────────

_NODE_FAMILY_COLOR = {
    "EXPLORATION": "#4C9BE8",
    "ELABORATION": "#2ECC71",
    "EVALUATION":  "#E74C3C",
    "CONVERGENCE": "#F39C12",
    None:          "#AAAAAA",
}
_EDGE_DRAW_COLOR = {
    "BRCH": "#2166AC",
    "ELAB": "#1A9641",
    "BACK": "#D7191C",
    "SYNT": "#F5A623",
    "CRIT": "#9B2335",
    "SUPP": "#74C476",
    "CONT": "#E67E22",
    "SEQ":  "#DDDDDD",
}


def _draw_single_graph(ax, graph, title: str) -> None:
    """Draw one ThoughtGraph on an axes using NetworkX spring layout."""
    import networkx as nx

    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.tu_id, family=getattr(node.node_family, "value", None),
                   ntype=getattr(node.node_type, "value", "?"))
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target, etype=edge.edge_type.value)

    # separate semantic edges for layout (spring on semantic only)
    G_layout = nx.DiGraph()
    G_layout.add_nodes_from(G.nodes())
    G_layout.add_edges_from(
        (u, v) for u, v, d in G.edges(data=True) if d.get("etype") != "SEQ"
    )
    if G_layout.number_of_edges() == 0:
        G_layout = G
    try:
        pos = nx.spring_layout(G_layout, k=2.0 / max(np.sqrt(len(graph.nodes)), 1),
                               seed=42, iterations=80)
    except Exception:
        pos = nx.circular_layout(G)

    node_colors = [_NODE_FAMILY_COLOR.get(G.nodes[n].get("family"), "#AAA")
                   for n in G.nodes()]
    node_sizes = [max(80, 400 // max(len(graph.nodes), 1)) for _ in G.nodes()]

    # draw edges by type (skip SEQ to reduce clutter)
    for etype, ecolor in _EDGE_DRAW_COLOR.items():
        if etype == "SEQ":
            elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == etype]
            if elist:
                nx.draw_networkx_edges(G, pos, edgelist=elist, ax=ax,
                                       edge_color=ecolor, width=0.5, alpha=0.3,
                                       arrows=True, arrowsize=6,
                                       connectionstyle="arc3,rad=0.05")
        else:
            elist = [(u, v) for u, v, d in G.edges(data=True) if d.get("etype") == etype]
            if elist:
                nx.draw_networkx_edges(G, pos, edgelist=elist, ax=ax,
                                       edge_color=ecolor, width=1.5, alpha=0.80,
                                       arrows=True, arrowsize=10,
                                       connectionstyle="arc3,rad=0.12")

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           edgecolors="white", linewidths=0.5)

    # small type labels on nodes
    labels = {n: G.nodes[n].get("ntype", "?") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=5, font_color="white", font_weight="bold")

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")


def graph_examples(graphs_dir: Path, tid_variant: dict[str, str],
                   output_path: Path, n_per_variant: int = 3) -> None:
    """3 × n_per_variant grid of representative thought graphs."""
    import networkx as nx
    from thinkbench.extract.schemas import ThoughtGraph

    # Score graphs by structural richness and pick n_per_variant per variant
    best: dict[str, list] = {v: [] for v in VARIANTS}
    for gf in sorted(graphs_dir.glob("*.json")):
        with open(gf) as f:
            g = ThoughtGraph(**json.load(f))
        v = tid_variant.get(g.trace_id, "unknown")
        if v not in VARIANTS:
            continue
        brch = sum(1 for e in g.edges if e.edge_type.value == "BRCH")
        back = sum(1 for e in g.edges if e.edge_type.value == "BACK")
        synt = sum(1 for e in g.edges if e.edge_type.value == "SYNT")
        # prefer medium-complexity graphs (not too large to draw)
        size_penalty = max(0, len(g.nodes) - 35) * 2
        score = brch * 3 + back * 2 + synt - size_penalty
        best[v].append((score, gf.name, g))

    for v in VARIANTS:
        best[v].sort(key=lambda x: x[0], reverse=True)
        best[v] = best[v][:n_per_variant]

    rows = n_per_variant
    cols = len(VARIANTS)
    fig = plt.figure(figsize=(cols * 6, rows * 5.5), constrained_layout=True)

    for ci, v in enumerate(VARIANTS):
        for ri, (score, fname, g) in enumerate(best[v]):
            ax = fig.add_subplot(rows, cols, ri * cols + ci + 1)
            n_nodes = len(g.nodes)
            n_edges = len(g.edges)
            brch = sum(1 for e in g.edges if e.edge_type.value == "BRCH")
            back = sum(1 for e in g.edges if e.edge_type.value == "BACK")
            synt = sum(1 for e in g.edges if e.edge_type.value == "SYNT")
            title = (f"[{VARIANT_LABEL.get(v, v)}]\n"
                     f"{n_nodes} nodes · {n_edges} edges "
                     f"(↗{brch} ↩{back} ⊕{synt})")
            _draw_single_graph(ax, g, title)

    # shared node-family legend
    family_patches = [
        mpatches.Patch(facecolor=c, label=fam)
        for fam, c in _NODE_FAMILY_COLOR.items() if fam
    ]
    # edge-type legend
    edge_lines = [
        Line2D([0], [0], color=c, linewidth=2, label=et)
        for et, c in _EDGE_DRAW_COLOR.items() if et != "SEQ"
    ]
    all_handles = family_patches + edge_lines
    fig.legend(handles=all_handles, loc="lower center",
               ncol=len(all_handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.95,
               title="Node family (fill) · Edge type (colour)",
               title_fontsize=9)

    fig.suptitle("Thought Graph Examples by Prompt Variant\n"
                 "↗=BRCH  ↩=BACK  ⊕=SYNT",
                 fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


# ── 14. GRAPH GRID — ALL TRACES (supplement) ─────────────────────────────────

def graph_grid_all(graphs_dir: Path, tid_variant: dict[str, str],
                   output_dir: Path, per_page: int = 9) -> None:
    """Generate multi-PNG supplement: all traces grouped by variant, per_page per file."""
    from thinkbench.extract.schemas import ThoughtGraph

    by_variant: dict[str, list] = {v: [] for v in VARIANTS}
    for gf in sorted(graphs_dir.glob("*.json")):
        with open(gf) as f:
            g = ThoughtGraph(**json.load(f))
        v = tid_variant.get(g.trace_id, "unknown")
        if v in VARIANTS:
            by_variant[v].append(g)

    cols = 3
    rows = per_page // cols
    for v in VARIANTS:
        graphs = by_variant[v]
        pages = (len(graphs) + per_page - 1) // per_page
        for pg in range(pages):
            batch = graphs[pg * per_page:(pg + 1) * per_page]
            fig, axes = plt.subplots(rows, cols,
                                     figsize=(cols * 5, rows * 4.5),
                                     constrained_layout=True)
            axes = np.array(axes).flatten()
            for ai, g in enumerate(batch):
                n_nodes = len(g.nodes)
                n_edges = len(g.edges)
                brch = sum(1 for e in g.edges if e.edge_type.value == "BRCH")
                back = sum(1 for e in g.edges if e.edge_type.value == "BACK")
                synt = sum(1 for e in g.edges if e.edge_type.value == "SYNT")
                title = (f"{g.trace_id[:8]}\n"
                         f"n={n_nodes} e={n_edges} "
                         f"↗{brch} ↩{back} ⊕{synt}")
                _draw_single_graph(axes[ai], g, title)
            for ai in range(len(batch), len(axes)):
                axes[ai].axis("off")

            fig.suptitle(f"All Traces — {VARIANT_LABEL.get(v, v)} "
                         f"(page {pg+1}/{pages})",
                         fontsize=13, fontweight="bold")
            out_path = output_dir / f"graphs_all_{v}_p{pg+1:02d}.png"
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  {out_path.name}")


# ── 15. METRIC DELTA DISTRIBUTION (violin of Δ per trace) ────────────────────

def delta_violin(profiles_per_trace: list[dict], output_path: Path) -> None:
    """Show per-trace deltas (eliciting−pure) as violin for each metric.

    Since we don't have paired traces per question, we approximate:
    show the raw value distributions for eliciting and pure side by side,
    and overlay the difference of means.
    """
    def _mat(variant):
        rows = [[p[m] for m in COGNITIVE_METRICS]
                for p in profiles_per_trace if p.get("prompt_variant") == variant]
        return np.array(rows) if rows else np.zeros((0, len(COGNITIVE_METRICS)))

    pure_mat   = _mat("pure")
    elicit_mat = _mat("eliciting")
    normal_mat = _mat("normal")

    # compute effect size (Cohen's d) per metric
    def cohens_d(a, b):
        pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
        return (b.mean() - a.mean()) / pooled if pooled > 0 else 0.0

    ds = [cohens_d(pure_mat[:, i] if pure_mat.ndim == 2 and pure_mat.shape[0] > 0 else np.zeros(1),
                  elicit_mat[:, i] if elicit_mat.ndim == 2 and elicit_mat.shape[0] > 0 else np.zeros(1))
          for i in range(len(COGNITIVE_METRICS))]
    order = np.argsort(ds)[::-1]

    short_ord = [METRIC_SHORT.get(COGNITIVE_METRICS[i], COGNITIVE_METRICS[i]) for i in order]
    ds_ord = [ds[i] for i in order]
    colors_ord = [CATEGORY_COLOR.get(CATEGORY_OF.get(COGNITIVE_METRICS[i], ""), "#888")
                  for i in order]

    fig, ax = plt.subplots(figsize=(14, max(10, len(COGNITIVE_METRICS) * 0.55)),
                           constrained_layout=True)
    bars = ax.barh(range(len(ds_ord)), ds_ord, color=colors_ord,
                   edgecolor="white", height=0.72)
    ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0.2, color="gray", lw=0.7, ls=":", alpha=0.5)
    ax.axvline(-0.2, color="gray", lw=0.7, ls=":", alpha=0.5)
    ax.axvline(0.8, color="gray", lw=0.7, ls=":", alpha=0.5)
    ax.axvline(-0.8, color="gray", lw=0.7, ls=":", alpha=0.5)
    ax.text(0.22, len(ds_ord) + 0.2, "small", fontsize=7, color="gray")
    ax.text(0.82, len(ds_ord) + 0.2, "large", fontsize=7, color="gray")

    for bar, val in zip(bars, ds_ord):
        ha = "left" if val >= 0 else "right"
        off = 0.03
        ax.text(val + (off if val >= 0 else -off), bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=8)

    ax.set_yticks(range(len(short_ord)))
    ax.set_yticklabels(short_ord, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d  (eliciting − pure)", fontsize=11)
    ax.set_title("Effect Size of Eliciting Prompt on Each Metric\n"
                 "(|d| < 0.2 small · 0.2–0.8 medium · > 0.8 large)",
                 fontsize=13, fontweight="bold")

    cat_patches = [mpatches.Patch(facecolor=c, label=cat)
                   for cat, c in CATEGORY_COLOR.items()]
    ax.legend(handles=cat_patches, fontsize=9, framealpha=0.9,
              title="Category", title_fontsize=9, loc="lower right")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── generate_all — convenience wrapper ────────────────────────────────────────

def generate_all(
    profiles_agg: list[dict],
    profiles_per_trace: list[dict],
    sensitivity: list[dict],
    graphs_dir: Path,
    tid_variant: dict[str, str],
    figures_dir: Path,
    supplement_dir: Path | None = None,
) -> None:
    """Generate all paper and supplementary figures into figures_dir."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    def _save(fn, *args, **kwargs):
        path = figures_dir / fn
        print(f"  {fn}", flush=True)
        return path

    # ── main figures ──────────────────────────────────────────────────────────
    p = figures_dir / "fig1_sensitivity_bar.png"
    print(f"  {p.name}"); sensitivity_bar(sensitivity, p)

    p = figures_dir / "fig2_radar.png"
    print(f"  {p.name}"); radar_comparison(profiles_agg, p)

    p = figures_dir / "fig3_category_comparison.png"
    print(f"  {p.name}"); category_comparison(profiles_agg, p)

    p = figures_dir / "fig4_metric_violin.png"
    print(f"  {p.name}"); metric_violin(profiles_per_trace, p)

    p = figures_dir / "fig5_effect_size.png"
    print(f"  {p.name}"); delta_violin(profiles_per_trace, p)

    p = figures_dir / "fig6_correlation_heatmap.png"
    print(f"  {p.name}"); correlation_heatmap(profiles_per_trace, p)

    p = figures_dir / "fig7_pca.png"
    print(f"  {p.name}"); pca_traces(profiles_per_trace, p)

    p = figures_dir / "fig8_graph_examples.png"
    print(f"  {p.name}"); graph_examples(graphs_dir, tid_variant, p)

    p = figures_dir / "fig9_node_edge_dist.png"
    print(f"  {p.name}"); node_edge_distributions(graphs_dir, tid_variant, p)

    p = figures_dir / "fig10_parallel_coords.png"
    print(f"  {p.name}"); parallel_coordinates(profiles_per_trace, p)

    # ── supplement ────────────────────────────────────────────────────────────
    supp = supplement_dir or (figures_dir / "supplement")
    supp.mkdir(parents=True, exist_ok=True)

    p = supp / "S1_metric_heatmap.png"
    print(f"  supplement/{p.name}"); metric_heatmap(profiles_agg, p)

    p = supp / "S2_scatter_pairs.png"
    print(f"  supplement/{p.name}"); scatter_key_pairs(profiles_per_trace, p)

    p = supp / "S3_dendrogram.png"
    print(f"  supplement/{p.name}"); cluster_dendrogram(profiles_per_trace, p)

    p = supp / "S4_per_trace_heatmap.png"
    print(f"  supplement/{p.name}"); per_trace_profile_heatmap(profiles_per_trace, p)

    print("  supplement/graphs_all_*.png")
    graph_grid_all(graphs_dir, tid_variant, supp, per_page=9)
