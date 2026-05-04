# ThinkBench Prompt Variant Study Report

**Generated**: 2026-04-30 16:06
**Model**: Qwen/Qwen3.5-35B-A3B
**Questions**: data/questions/test_2q.jsonl
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
| empty | 30 | 0.420 | 5.63 | 1.000 | 0.001 |
| pure | 30 | 0.433 | 5.23 | 1.000 | 0.002 |
| normal | 30 | 0.430 | 4.73 | 1.000 | 0.003 |
| eliciting | 30 | 0.346 | 11.40 | 1.000 | 0.000 |


---

## Prompt Sensitivity Analysis

Metrics ranked by |Δ(eliciting − pure)|. Positive = increases with more guidance.

| Metric | Category | Δ Eliciting | Δ Normal | Abs Sensitivity |
|---|---|---|---|---|
| token_per_idea | Efficiency | +118.7186 | +9.8362 | 118.7186 |
| domain_spread | Breadth | +11.0333 | +0.1333 | 11.0333 |
| unique_perspective_count | Breadth | +6.1667 | -0.5000 | 6.1667 |
| max_elaboration_chain | Depth | +5.7000 | -0.6000 | 5.7000 |
| mean_branch_depth | Depth | +4.8745 | -0.3235 | 4.8745 |
| revision_depth | Structure | +3.2835 | +0.1745 | 3.2835 |
| first_idea_diversity | Breadth | +0.4744 | +0.2242 | 0.4744 |
| critique_to_hypothesis_ratio | Metacognitive | +0.2671 | +0.5750 | 0.2671 |
| cross_branch_connectivity | Structure | -0.2339 | -0.0284 | 0.2339 |
| exploration_exploitation_ratio | Structure | -0.1441 | +0.2313 | 0.1441 |

---

## Metric Classification

**Prompt-sensitive** (top 6 by Δ — likely reflect prompted behavior):
`token_per_idea`, `domain_spread`, `unique_perspective_count`, `max_elaboration_chain`, `mean_branch_depth`, `revision_depth`

**Prompt-invariant** (lower Δ — likely reflect intrinsic capability):
`first_idea_diversity`, `critique_to_hypothesis_ratio`, `cross_branch_connectivity`, `exploration_exploitation_ratio`, `perspective_taking`, `graph_density`, `branching_factor`, `hedging_density`, `convergence_index`, `backtracking_rate`, `redundancy_ratio`, `specificity_gradient`, `self_reflection_rate`, `reasoning_density`

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
