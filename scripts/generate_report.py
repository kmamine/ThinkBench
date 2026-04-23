#!/usr/bin/env python3
"""Generate benchmark report with visualizations (v2 metrics)."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

PROFILE_FILE = Path("data/profiles/effort_comparison.json")
OUTPUT_DIR = Path("docs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# Placeholder bounds — replaced by empirical min/max after full benchmark run
_BOUNDS: dict[str, tuple[float, float]] = {
    "branching_factor": (0, 0.5),
    "unique_perspective_count": (0, 10),
    "domain_spread": (0, 10),
    "first_idea_diversity": (0, 1),
    "max_elaboration_chain": (0, 15),
    "mean_branch_depth": (0, 10),
    "specificity_gradient": (-0.1, 0.1),
    "reasoning_density": (0, 1),
    "exploration_exploitation_ratio": (0, 5),
    "backtracking_rate": (0, 1),
    "cross_branch_connectivity": (0, 1),
    "convergence_index": (0, 1),
    "graph_density": (0, 0.5),
    "revision_depth": (0, 20),
    "self_reflection_rate": (0, 0.3),
    "critique_to_hypothesis_ratio": (0, 1),
    "hedging_density": (0, 1),
    "perspective_taking": (0, 0.3),
    "token_per_idea": (0, 2000),
    "redundancy_ratio": (0, 0.5),
}

_RADAR_METRICS = list(_BOUNDS.keys())


def normalize_value(value: float, metric_name: str) -> float:
    lo, hi = _BOUNDS.get(metric_name, (0, 1))
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def load_profile():
    with open(PROFILE_FILE) as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def generate_radar_chart(profile: dict, output_path: Path) -> None:
    values = [normalize_value(profile.get(m, 0.0), m) for m in _RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, len(_RADAR_METRICS), endpoint=False).tolist()
    values_plot = np.concatenate([values, [values[0]]])
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#2E86AB")
    ax.fill(angles_plot, values_plot, alpha=0.25, color="#2E86AB")
    ax.set_xticks(angles)
    ax.set_xticklabels(_RADAR_METRICS, size=7)
    model_name = profile.get("model", profile.get("effort", "unknown"))
    ax.set_title(
        f"Cognitive Profile: {model_name}\n(N={profile.get('num_traces', '?')} traces)",
        size=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved radar chart: {output_path}")


def generate_report(profile: dict) -> str:
    model_name = profile.get("model", profile.get("effort", "unknown"))
    return f"""# ThinkBench Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Model:** {model_name}
**N:** {profile.get('num_traces', '?')} traces

---

## Breadth
| Metric | Value |
|--------|-------|
| branching_factor | {profile.get('branching_factor', 0):.4f} |
| unique_perspective_count | {profile.get('unique_perspective_count', 0):.2f} |
| domain_spread | {profile.get('domain_spread', 0):.2f} |
| first_idea_diversity | {profile.get('first_idea_diversity', 0):.4f} |

## Depth
| Metric | Value |
|--------|-------|
| max_elaboration_chain | {profile.get('max_elaboration_chain', 0):.2f} |
| mean_branch_depth | {profile.get('mean_branch_depth', 0):.2f} |
| specificity_gradient | {profile.get('specificity_gradient', 0):.4f} |
| reasoning_density | {profile.get('reasoning_density', 0):.4f} |

## Structure
| Metric | Value |
|--------|-------|
| exploration_exploitation_ratio | {profile.get('exploration_exploitation_ratio', 0):.4f} |
| backtracking_rate | {profile.get('backtracking_rate', 0):.4f} |
| cross_branch_connectivity | {profile.get('cross_branch_connectivity', 0):.4f} |
| convergence_index | {profile.get('convergence_index', 0):.4f} |
| graph_density | {profile.get('graph_density', 0):.4f} |
| revision_depth | {profile.get('revision_depth', 0):.4f} |

## Metacognitive
| Metric | Value |
|--------|-------|
| self_reflection_rate | {profile.get('self_reflection_rate', 0):.4f} |
| critique_to_hypothesis_ratio | {profile.get('critique_to_hypothesis_ratio', 0):.4f} |
| hedging_density | {profile.get('hedging_density', 0):.4f} |
| perspective_taking | {profile.get('perspective_taking', 0):.4f} |

## Efficiency
| Metric | Value |
|--------|-------|
| token_per_idea | {profile.get('token_per_idea', 0):.2f} |
| redundancy_ratio | {profile.get('redundancy_ratio', 0):.4f} |

## Summary
| Metric | Value |
|--------|-------|
| avg_tokens | {profile.get('avg_tokens', 0):.1f} |
| avg_tus | {profile.get('avg_tus', 0):.1f} |

---

## Key Observations

1. **Backtracking**: {profile.get('backtracking_rate', 0) * 100:.1f}% of semantic edges are BACK edges
2. **Exploration/Exploitation**: ratio = {profile.get('exploration_exploitation_ratio', 0):.2f}
3. **Hedging density**: {profile.get('hedging_density', 0) * 100:.1f}% of TUs contain uncertainty markers
4. **Revision depth**: mean = {profile.get('revision_depth', 0):.1f} TUs between BACK edge endpoints

---

![Cognitive Profile Radar Chart](benchmark_radar_{TIMESTAMP}.png)

*Generated by ThinkBench v2*
"""


def main():
    print("Loading profile data...")
    profile = load_profile()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating radar chart...")
    generate_radar_chart(profile, OUTPUT_DIR / f"benchmark_radar_{TIMESTAMP}.png")

    print("Generating markdown report...")
    report = generate_report(profile)
    report_path = OUTPUT_DIR / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
