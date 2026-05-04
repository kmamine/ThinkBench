# ThinkBench Prompt Variant Study Report

**Generated**: 2026-04-30 10:30
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
| empty | 30 | 0.422 | 5.53 | 1.000 | 0.001 |
| pure | 30 | 0.429 | 5.17 | 1.000 | 0.002 |
| normal | 30 | 0.429 | 4.83 | 1.000 | 0.003 |
| eliciting | 30 | 0.371 | 10.47 | 1.000 | 0.000 |


---

## Prompt Sensitivity Analysis

Metrics ranked by |Δ(eliciting − pure)|. Positive = increases with more guidance.

| Metric | Category | Δ Eliciting | Δ Normal | Abs Sensitivity |
|---|---|---|---|---|
| token_per_idea | Efficiency | +127.2263 | +7.3395 | 127.2263 |
| max_elaboration_chain | Depth | +11.9000 | -0.5667 | 11.9000 |
| domain_spread | Breadth | +7.9333 | +0.0667 | 7.9333 |
| unique_perspective_count | Breadth | +5.3000 | -0.3333 | 5.3000 |
| mean_branch_depth | Depth | +4.1660 | -0.3418 | 4.1660 |
| revision_depth | Structure | +3.9026 | +0.2001 | 3.9026 |
| first_idea_diversity | Breadth | +0.4521 | +0.1292 | 0.4521 |
| critique_to_hypothesis_ratio | Metacognitive | +0.3725 | +0.5583 | 0.3725 |
| exploration_exploitation_ratio | Structure | +0.2891 | +0.2496 | 0.2891 |
| cross_branch_connectivity | Structure | -0.2105 | -0.0232 | 0.2105 |

---

## Metric Classification

**Prompt-sensitive** (top 6 by Δ — likely reflect prompted behavior):
`token_per_idea`, `max_elaboration_chain`, `domain_spread`, `unique_perspective_count`, `mean_branch_depth`, `revision_depth`

**Prompt-invariant** (lower Δ — likely reflect intrinsic capability):
`first_idea_diversity`, `critique_to_hypothesis_ratio`, `exploration_exploitation_ratio`, `cross_branch_connectivity`, `perspective_taking`, `graph_density`, `convergence_index`, `branching_factor`, `hedging_density`, `redundancy_ratio`, `specificity_gradient`, `self_reflection_rate`, `backtracking_rate`, `reasoning_density`

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
