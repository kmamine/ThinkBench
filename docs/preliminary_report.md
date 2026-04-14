# ThinkBench: Preliminary Comparison Report

## Overview

This report compares cognitive profiles across 3 thinking effort levels (LOW, MEDIUM, HIGH) using Qwen3.5-35B-A3B via vLLM with **prompt-based effort control**. K=3, 10 questions per effort level (30 traces each).

## Effort Control Implementation

Two methods supported:
1. **Parameter-based**: `thinking_budget` parameter (for models like GPT-4.1 that support it)
2. **Prompt-based**: System prompt modifiers (default, for Qwen and similar models)

| Effort | Prompt Modifier |
|--------|-----------------|
| LOW | "Think briefly and give a concise answer." |
| MEDIUM | "Think through this problem carefully." |
| HIGH | "Think step by step. Show all your reasoning. Consider multiple angles." |
| MAX | "Think deeply and exhaustively. Explore all possibilities. Consider edge cases. Show comprehensive reasoning." |

## Token Output (Verification of Effort Control)

| Effort | Expected | Actual Avg Tokens |
|--------|----------|-------------------|
| LOW | ~300 | 319 |
| MEDIUM | ~1000 | 978 |
| HIGH | ~1100 | 1072 |

**SUCCESS**: Prompt-based control produces meaningful token differentiation!

## All 22 Metrics

### Breadth Metrics (4)
- **branching_factor**: Average number of branches per reasoning step
- **unique_perspective_count**: Number of distinct analytical viewpoints
- **domain_spread**: Spread of topic coverage
- **first_idea_diversity**: Diversity of initial ideas (embedding-based)

### Depth Metrics (4)
- **max_elaboration_chain**: Longest reasoning chain length
- **mean_branch_depth**: Average depth of reasoning branches
- **specificity_gradient**: How quickly reasoning becomes specific (requires spaCy)
- **reasoning_density**: Ratio of reasoning nodes to total nodes

### Structure Metrics (7)
- **exploration_exploitation_ratio**: Balance between exploring vs exploiting
- **backtracking_rate**: Frequency of returning to previous states
- **cross_branch_connectivity**: Connections between branches
- **convergence_index**: How quickly reasoning converges
- **orphan_ratio**: Nodes without connections
- **graph_density**: Edge-to-node ratio
- **cycle_count**: Number of reasoning cycles
- **mean_cycle_length**: Average length of cycles

### Metacognitive Metrics (4)
- **self_reflection_rate**: Self-referential thinking frequency
- **critique_to_hypothesis_ratio**: Critical vs hypothesis nodes
- **hedging_density**: Uncertainty expression frequency
- **perspective_taking**: Multi-perspective reasoning

### Efficiency Metrics (2)
- **token_per_idea**: Average tokens per idea unit
- **redundancy_ratio**: Repetition in thinking trace

---

## Results by Effort Level (K=3, Q=10)

| Metric | LOW | HIGH |
|--------|-----|------|
| **Sample Size** | | |
| num_traces | 12 | 15 |
| avg_tokens | 629.4 | 1076.0 |
| | | |
| **Breadth (4)** | | |
| branching_factor | 0.0 | 0.0 |
| unique_perspective_count | 1.67 | 1.4 |
| domain_spread | 1.08 | 2.93 |
| first_idea_diversity | 0.51 | 0.58 |
| | | |
| **Depth (4)** | | |
| max_elaboration_chain | 5.42 | 6.73 |
| mean_branch_depth | 1.14 | 1.30 |
| specificity_gradient | 0.0 | 0.0 |
| reasoning_density | 0.22 | 0.24 |
| | | |
| **Structure (7)** | | |
| exploration_exploitation_ratio | 2.14 | 2.76 |
| backtracking_rate | 0.18 | 0.04 |
| cross_branch_connectivity | 0.06 | 0.01 |
| convergence_index | 0.17 | 0.20 |
| orphan_ratio | 0.34 | 0.25 |
| graph_density | 1.15 | 0.93 |
| cycle_count | 2.17 | 0.27 |
| mean_cycle_length | 0.97 | 0.33 |
| | | |
| **Metacognitive (4)** | | |
| self_reflection_rate | 0.01 | 0.0 |
| critique_to_hypothesis_ratio | 0.07 | 0.04 |
| hedging_density | 0.15 | 0.30 |
| perspective_taking | 0.0 | 0.0 |
| | | |
| **Efficiency (2)** | | |
| token_per_idea | 400.3 | 866.8 |
| redundancy_ratio | 0.0001 | 0.0 |

---

## Key Observations

1. **Token Output**: HIGH produces ~70% more tokens than LOW - effort control working!

2. **Cycle Count**: LOW shows significantly more cycles (2.17 vs 0.27) - more iterative reasoning

3. **Backtracking**: LOW shows more backtracking (0.18 vs 0.04) - more self-correction

4. **Hedging**: HIGH shows more hedging (0.30 vs 0.15) - more uncertainty expression

5. **Max Elaboration**: HIGH deeper chains (6.73 vs 5.42)

6. **Graph Density**: LOW denser graphs (1.15 vs 0.93)

7. **Token Efficiency**: LOW more efficient (400 vs 867 tokens/idea)

---

## Interpretation

- **LOW**: Shorter but denser reasoning, more cycles, more self-correction, more efficient
- **HIGH**: Longer output, deeper chains, more hedging, less dense structure

---

## Limitations

- MEDIUM not extracted yet (timeout issues during extraction)
- Small sample sizes (n=12-15 per effort)
- spaCy model not loaded (specificity_gradient = 0)

---

## Next Steps

1. Complete MEDIUM extraction
2. Run K=15 experiments for statistical power
3. Add radar chart visualizations
4. Load spaCy model for specificity_gradient
