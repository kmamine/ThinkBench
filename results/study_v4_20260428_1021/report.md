# ThinkBench Prompt Variant Study Report

**Generated**: 2026-04-28 10:58
**Model**: Qwen/Qwen3.5-35B-A3B
**Questions**: data/questions/ethical_dilemmas.jsonl
**Runs per variant**: 3
**Extraction**: 120 graphs extracted (0 failed, 0 skipped)

---

## Experimental Design

Four prompt variants were tested to separate **intrinsic model capability** from
**prompted reasoning behavior**:

| Variant | Description |
|---|---|
| `empty` | No system prompt (completely absent from the request) |
| `pure` | Bare system prompt: "You are a helpful assistant." |
| `normal` | Minimal guidance: "Think carefully before answering." |
| `eliciting` | Full framing: explicit instructions to spread wide, reframe, go deep, challenge, connect, reflect, converge |

---

## Variant Summary

| Variant | N traces | Branching Factor | Perspectives | Reasoning Density | Self-Reflection |
|---|---|---|---|---|---|
| empty | 30 | 0.237 | 1.83 | 1.000 | 0.026 |
| pure | 30 | 0.236 | 1.67 | 1.000 | 0.003 |
| normal | 30 | 0.240 | 1.80 | 1.000 | 0.031 |
| eliciting | 30 | 0.233 | 3.10 | 1.000 | 0.002 |


---

## Prompt Sensitivity Analysis

Metrics ranked by |Δ(eliciting − pure)|. Positive = increases with more guidance.

| Metric | Category | Δ Eliciting | Δ Normal | Abs Sensitivity |
|---|---|---|---|---|
| token_per_idea | Efficiency | +358.2833 | +1.6768 | 358.2833 |
| revision_depth | Structure | +7.7038 | -0.9933 | 7.7038 |
| max_elaboration_chain | Depth | +4.9000 | -0.2667 | 4.9000 |
| domain_spread | Breadth | +3.2333 | -0.1000 | 3.2333 |
| mean_branch_depth | Depth | +2.9490 | +0.1751 | 2.9490 |
| unique_perspective_count | Breadth | +1.4333 | +0.1333 | 1.4333 |
| critique_to_hypothesis_ratio | Metacognitive | +1.1842 | -0.2111 | 1.1842 |
| first_idea_diversity | Breadth | +0.6001 | +0.0004 | 0.6001 |
| exploration_exploitation_ratio | Structure | +0.2232 | +0.1606 | 0.2232 |
| graph_density | Structure | -0.1818 | +0.0131 | 0.1818 |

---

## Metric Classification

**Prompt-sensitive** (top 6 by Δ — likely reflect prompted behavior):
`token_per_idea`, `revision_depth`, `max_elaboration_chain`, `domain_spread`, `mean_branch_depth`, `unique_perspective_count`

**Prompt-invariant** (lower Δ — likely reflect intrinsic capability):
`critique_to_hypothesis_ratio`, `first_idea_diversity`, `exploration_exploitation_ratio`, `graph_density`, `cross_branch_connectivity`, `perspective_taking`, `convergence_index`, `hedging_density`, `backtracking_rate`, `redundancy_ratio`, `specificity_gradient`, `branching_factor`, `self_reflection_rate`, `reasoning_density`

---

## Figures

| Figure | Description |
|---|---|
| `figures/radar_comparison.png` | Overlaid radar chart — all 20 metrics, all 3 variants |
| `figures/sensitivity_ranking.png` | Horizontal bar chart ranking metrics by Δ (dual panel) |
| `figures/metric_heatmap.png` | Heatmap: metrics × variants, normalized [0–1] |
| `figures/category_comparison.png` | Grouped bar: mean category score per variant |

---

## Interpretation Notes

- Metrics with **large positive Δ** under `eliciting` but not `normal` are
  specifically responsive to explicit framing instructions — these measure
  *prompted* rather than *intrinsic* behavior.
- Metrics that remain **stable across all three variants** reflect the model's
  baseline cognitive style regardless of how it is prompted.
- The `pure → normal` gap reveals the effect of a simple "think carefully"
  nudge; the `normal → eliciting` gap reveals the effect of structured
  reasoning framing.
