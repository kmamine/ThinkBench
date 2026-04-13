# ThinkBench: Preliminary Comparison Report

## Overview

This report compares cognitive profiles across 4 thinking effort levels (LOW, MEDIUM, HIGH, MAX) using the Qwen3.5-35B-A3B model via vLLM. Sample size: 7 traces (K=5 per effort, partial extraction completed).

## All 22 Metrics

### Breadth Metrics (4)
- **branching_factor**: Average number of branches per reasoning step
- **unique_perspective_count**: Number of distinct analytical viewpoints
- **domain_spread**: Spread of topic coverage
- **first_idea_diversity**: Diversity of initial ideas (embedding-based)

### Depth Metrics (4)
- **max_elaboration_chain**: Longest reasoning chain length
- **mean_branch_depth**: Average depth of reasoning branches
- **specificity_gradient**: How quickly reasoning becomes specific
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

## Results by Effort Level

| Metric | LOW | MEDIUM | HIGH | MAX |
|--------|-----|--------|------|-----|
| **Token Count** | | | | |
| avg_tokens | 1010.5 | 762.0 | 1025.5 | 1042.0 |
| | | | | |
| **Breadth (4)** | | | | |
| branching_factor | 0.0 | 0.0 | 0.0 | 0.0 |
| unique_perspective_count | 1.5 | 0.5 | 2.0 | 1.5 |
| domain_spread | - | - | - | - |
| first_idea_diversity | - | - | - | - |
| | | | | |
| **Depth (4)** | | | | |
| max_elaboration_chain | 4.0 | 3.0 | 5.0 | 4.5 |
| mean_branch_depth | - | - | - | - |
| specificity_gradient | - | - | - | - |
| reasoning_density | 0.17 | 0.56 | 0.17 | 0.18 |
| | | | | |
| **Structure (7)** | | | | |
| exploration_exploitation_ratio | - | - | - | - |
| backtracking_rate | - | - | - | - |
| cross_branch_connectivity | - | - | - | - |
| convergence_index | - | - | - | - |
| orphan_ratio | - | - | - | - |
| graph_density | 1.18 | 0.48 | 1.27 | 1.33 |
| cycle_count | - | - | - | - |
| mean_cycle_length | - | - | - | - |
| | | | | |
| **Metacognitive (4)** | | | | |
| self_reflection_rate | 0.0 | 0.0 | 0.0 | 0.0 |
| critique_to_hypothesis_ratio | - | - | - | - |
| hedging_density | 0.18 | 0.75 | 0.17 | 0.18 |
| perspective_taking | - | - | - | - |
| | | | | |
| **Efficiency (2)** | | | | |
| token_per_idea | 752.25 | 762.0 | 491.5 | 793.5 |
| redundancy_ratio | - | - | - | - |

---

## Key Observations

1. **Token Output**: MEDIUM produces ~25% fewer tokens (762 vs ~1000 for others)

2. **Reasoning Density**: MEDIUM shows significantly higher RD (0.56 vs ~0.17)

3. **Hedging**: MEDIUM has much higher hedging density (0.75), indicating more uncertainty

4. **Graph Density**: LOW/MEDIUM lower (0.48-1.18) vs HIGH/MAX higher (1.27-1.33)

5. **Unique Perspectives**: HIGH achieves highest UPC (2.0), suggesting more diverse thinking

6. **Max Elaboration**: HIGH also leads in MEC (5.0), deepest reasoning chains

7. **Token Efficiency**: HIGH most efficient (491.5 tokens/idea), MAX least (793.5)

---

## Interpretation

- **LOW**: Baseline reasoning, moderate token usage
- **MEDIUM**: More focused but uncertain, shows highest self-doubt (hedging)
- **HIGH**: Best balance - diverse perspectives, deep chains, efficient
- **MAX**: Similar to LOW but with slightly more depth, less efficient

---

## Limitations

- Small sample (n=7, HIGH only n=1)
- Missing metric values indicate extraction issues
- vLLM `thinking_budget` shows limited effect on output length

---

## Next Steps

1. Complete full extraction for all K=5 traces
2. Run K=15 experiments for statistical power
3. Add radar chart visualizations
4. Test additional models (DeepSeek, Claude)