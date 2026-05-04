# ThinkBench Prompt Variant Study Report

**Generated**: 2026-04-23 13:55
**Model**: Qwen/Qwen3.5-35B-A3B
**Questions**: data/questions/ethical_dilemmas.jsonl
**Runs per variant**: 3
**Extraction**: 90 graphs extracted (0 failed, 0 skipped)

---

## Experimental Design

Three prompt variants were tested to separate **intrinsic model capability** from
**prompted reasoning behavior**:

| Variant | Description |
|---|---|
| `pure` | Bare system prompt: "You are a helpful assistant." |
| `normal` | Minimal guidance: "Think carefully before answering." |
| `eliciting` | Full framing: explicit instructions to spread wide, reframe, go deep, challenge, connect, reflect, converge |

---

## Variant Summary

| Variant | N traces | Branching Factor | Perspectives | Reasoning Density | Self-Reflection |
|---|---|---|---|---|---|
| pure | 30 | 0.000 | 0.00 | 1.000 | 0.000 |
| normal | 30 | 0.000 | 0.00 | 1.000 | 0.000 |
| eliciting | 30 | 0.000 | 0.00 | 0.997 | 0.000 |


---

## Prompt Sensitivity Analysis

Metrics ranked by |Δ(eliciting − pure)|. Positive = increases with more guidance.

| Metric | Category | Δ Eliciting | Δ Normal | Abs Sensitivity |
|---|---|---|---|---|
| token_per_idea | Efficiency | +1651.2000 | -0.6667 | 1651.2000 |
| mean_branch_depth | Depth | +6.1567 | +0.0267 | 6.1567 |
| max_elaboration_chain | Depth | +1.0667 | -0.1333 | 1.0667 |
| revision_depth | Structure | +0.1500 | +0.0000 | 0.1500 |
| exploration_exploitation_ratio | Structure | -0.0886 | +0.0022 | 0.0886 |
| cross_branch_connectivity | Structure | -0.0863 | +0.0178 | 0.0863 |
| graph_density | Structure | -0.0753 | +0.0013 | 0.0753 |
| convergence_index | Structure | +0.0619 | +0.0177 | 0.0619 |
| critique_to_hypothesis_ratio | Metacognitive | +0.0333 | +0.0000 | 0.0333 |
| hedging_density | Metacognitive | -0.0205 | -0.0530 | 0.0205 |

---

## Metric Classification

**Prompt-sensitive** (top 6 by Δ — likely reflect prompted behavior):
`token_per_idea`, `mean_branch_depth`, `max_elaboration_chain`, `revision_depth`, `exploration_exploitation_ratio`, `cross_branch_connectivity`

**Prompt-invariant** (lower Δ — likely reflect intrinsic capability):
`graph_density`, `convergence_index`, `critique_to_hypothesis_ratio`, `hedging_density`, `backtracking_rate`, `reasoning_density`, `redundancy_ratio`, `branching_factor`, `unique_perspective_count`, `domain_spread`, `first_idea_diversity`, `specificity_gradient`, `self_reflection_rate`, `perspective_taking`

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
