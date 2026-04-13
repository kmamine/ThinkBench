# ThinkBench: Thinking Effort Comparison Report

## Overview

This report compares cognitive profiles across 4 thinking effort levels (LOW, MEDIUM, HIGH, MAX) using the Qwen3.5-35B-A3B model. Sample size: 7 traces total (2 per effort, 1 for HIGH).

## Key Findings

### Token Output
| Effort | Avg Tokens |
|--------|------------|
| LOW    | 1010       |
| MEDIUM | 762        |
| HIGH   | 1026       |
| MAX    | 1042       |

**Observation**: MEDIUM produces notably fewer tokens (~25% less). Other levels show similar output length (~1000 tokens).

### Reasoning Metrics
| Effort | Unique Perspectives (UPC) | Max Elaboration (MEC) | Reasoning Density (RD) |
|--------|---------------------------|----------------------|----------------------|
| LOW    | 1.5                       | 4.0                  | 0.17                 |
| MEDIUM | 0.5                       | 3.0                  | 0.56                 |
| HIGH   | 2.0                       | 5.0                  | 0.17                 |
| MAX    | 1.5                       | 4.5                  | 0.18                 |

### Observations

1. **HIGH effort** achieves highest UPC (2.0) and MEC (5.0), indicating more diverse perspectives and deeper reasoning chains.

2. **MEDIUM** shows highest RD (0.56) but lowest UPC/MEC - more focused but less comprehensive.

3. **Hedging density** is highest in MEDIUM (0.75) vs ~0.17 for others, suggesting more uncertainty.

4. **Token per idea** varies: HIGH is most efficient (491.5), MAX least efficient (793.5).

## Limitations

- Small sample size (n=7)
- HIGH only has 1 trace
- vLLM thinking_budget parameter shows limited effect on actual output length

## Next Steps

- Increase K to 15+ for statistical significance
- Test other models (DeepSeek, Claude)
- Add visualization (radar charts)