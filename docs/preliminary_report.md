# ThinkBench: Preliminary Comparison Report

## Overview

This report compares cognitive profiles across 4 thinking effort levels (LOW, MEDIUM, HIGH, MAX) using the Qwen3.5-35B-A3B model via vLLM. Sample size: K=3 traces per effort level (9 total traces extracted).

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
- **backtracking_rate**: Frequency of returning to previous states (BACK + CRIT edges)
- **cross_branch_connectivity**: Connections between branches (SYNT edges)
- **convergence_index**: How quickly reasoning converges
- **orphan_ratio**: Nodes without connections
- **graph_density**: Edge-to-node ratio
- **cycle_count**: Number of reasoning cycles
- **mean_cycle_length**: Average length of cycles

### Metacognitive Metrics (4)
- **self_reflection_rate**: Self-referential thinking frequency (MET nodes)
- **critique_to_hypothesis_ratio**: Critical vs hypothesis nodes
- **hedging_density**: Uncertainty expression frequency
- **perspective_taking**: Multi-perspective reasoning (RFR nodes)

### Efficiency Metrics (2)
- **token_per_idea**: Average tokens per idea unit
- **redundancy_ratio**: Repetition in thinking trace

---

## Results by Effort Level (K=3)

| Metric | LOW | MEDIUM | HIGH |
|--------|-----|--------|------|
| **Sample Size** | | | |
| num_traces | 5 | 3 | 1 |
| avg_tokens | 793.0 | 772.3 | 887.0 |
| | | | |
| **Breadth (4)** | | | |
| branching_factor | 0.0 | 0.0 | 0.0 |
| unique_perspective_count | 1.4 | 1.33 | 1.0 |
| domain_spread | 3.6 | 2.67 | 6.0 |
| first_idea_diversity | 0.36 | 0.35 | 0.49 |
| | | | |
| **Depth (4)** | | | |
| max_elaboration_chain | 3.2 | 3.0 | 4.0 |
| mean_branch_depth | 0.91 | 0.78 | 1.0 |
| specificity_gradient | 0.0 | 0.0 | 0.0 |
| reasoning_density | 0.19 | 0.23 | 0.40 |
| | | | |
| **Structure (7)** | | | |
| exploration_exploitation_ratio | 2.47 | 1.78 | 1.5 |
| backtracking_rate | 0.11 | 0.0 | 0.0 |
| cross_branch_connectivity | 0.0 | 0.0 | 0.0 |
| convergence_index | 0.0 | 0.0 | 0.0 |
| orphan_ratio | 0.35 | 0.44 | 0.33 |
| graph_density | 0.83 | 0.75 | 1.3 |
| cycle_count | 0.6 | 0.0 | 0.0 |
| mean_cycle_length | 0.9 | 0.0 | 0.0 |
| | | | |
| **Metacognitive (4)** | | | |
| self_reflection_rate | 0.0 | 0.0 | 0.0 |
| critique_to_hypothesis_ratio | 0.07 | 0.0 | 0.0 |
| hedging_density | 0.19 | 0.13 | 0.20 |
| perspective_taking | 0.0 | 0.33 | 0.0 |
| | | | |
| **Efficiency (2)** | | | |
| token_per_idea | 604.7 | 602.5 | 887.0 |
| redundancy_ratio | 0.0 | 0.0 | 0.0 |

---

## Key Observations

1. **Cycle Detection**: LOW shows 0.6 cycles per trace with mean length 0.9 - unique behavior!

2. **Backtracking**: LOW has highest backtracking rate (0.11), MEDIUM/HIGH show 0

3. **Critique**: Only LOW shows critique activity (0.07 ratio)

4. **Perspective Taking**: Only MEDIUM shows perspective-taking (0.33)

5. **Exploration**: LOW most exploratory (2.47), HIGH least (1.5)

6. **Reasoning Density**: HIGH highest (0.40), LOW lowest (0.19)

7. **Graph Density**: HIGH densest (1.3), MEDIUM sparse (0.75)

8. **Token Efficiency**: MEDIUM most efficient (602.5), HIGH least (887)

---

## Interpretation

- **LOW**: Most dynamic structure - shows cycles, backtracking, critique, high exploration
- **MEDIUM**: Includes perspective-taking, moderate structure, most efficient
- **HIGH**: Most reasoning-dense, sparse structure, deepest elaboration chains

---

## Fixes Applied

1. Expanded classifier patterns for all node types (RFR, MET, SYN, etc.)
2. Added rule-based edge detection with node type transitions
3. Improved backtracking detection with fallback heuristics
4. Added spaCy logging for specificity_gradient
5. Added edge types to depth graph (BRCH, SUPP)

---

## Limitations

- Small sample sizes (n=1-5 per effort)
- spaCy model may not be loaded (specificity_gradient = 0)
- Cycle detection limited to simple patterns

---

## Next Steps

1. Run K=15 experiments for statistical power
2. Add radar chart visualizations
3. Test spaCy model loading
4. Test additional models (DeepSeek, Claude)
