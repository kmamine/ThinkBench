"""
ThinkBench metric-space analysis: degenerate detection, Spearman correlation,
hierarchical clustering, variance / discriminability ranking.
"""

import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/m.a.kerkouri/SideProjects/ThoughtBench/src')

import numpy as np
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler

from thinkbench.extract.schemas import ThoughtGraph
from thinkbench.metrics import compute_profile

# ── 1. Load per-trace profiles ────────────────────────────────────────────────
graphs_dir = Path('/home/m.a.kerkouri/SideProjects/ThoughtBench/results/study_20260423_1326/graphs')
traces_dir = Path('/home/m.a.kerkouri/SideProjects/ThoughtBench/results/study_20260423_1326/traces')

tid_variant: dict[str, str] = {}
for tf in traces_dir.glob('traces_*.jsonl'):
    with open(tf) as f:
        for line in f:
            t = json.loads(line)
            tid_variant[t['trace_id']] = t.get('prompt_variant', 'normal')

profiles = []
for gf in sorted(graphs_dir.glob('*.json')):
    with open(gf) as f:
        g = ThoughtGraph(**json.load(f))
    p = compute_profile(g, model=g.model, domain=g.domain).model_dump()
    p['prompt_variant'] = tid_variant.get(g.trace_id, 'unknown')
    profiles.append(p)

print(f"Loaded: {len(profiles)} traces")
print(f"Variants: { {v: sum(1 for p in profiles if p['prompt_variant']==v) for v in ['pure','normal','eliciting']} }\n")

# ── 2. Metric matrix & basic statistics ──────────────────────────────────────
NUMERIC_FIELDS = [
    'branching_factor', 'unique_perspective_count', 'domain_spread',
    'first_idea_diversity', 'max_elaboration_chain', 'mean_branch_depth',
    'specificity_gradient', 'reasoning_density',
    'exploration_exploitation_ratio', 'backtracking_rate',
    'cross_branch_connectivity', 'convergence_index', 'graph_density',
    'revision_depth', 'self_reflection_rate', 'critique_to_hypothesis_ratio',
    'hedging_density', 'perspective_taking',
    'token_per_idea', 'redundancy_ratio',
    'avg_tokens', 'avg_tus',
]

variants = [p['prompt_variant'] for p in profiles]
mat = np.array([[p[m] for m in NUMERIC_FIELDS] for p in profiles])   # (90, 22)

print("=" * 80)
print("SECTION 2 — PER-METRIC STATISTICS (n=90 traces)")
print("=" * 80)
header = f"{'Metric':<38} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'Zero%':>7}"
print(header)
print("-" * 80)
for i, m in enumerate(NUMERIC_FIELDS):
    col = mat[:, i]
    zero_frac = (col == 0.0).mean() * 100
    print(f"{m:<38} {col.mean():>9.4f} {col.std():>9.4f} {col.min():>9.4f} {col.max():>9.4f} {zero_frac:>6.1f}%")

# ── 3. Spearman correlation matrix (full) ────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 3 — SPEARMAN CORRELATION (|ρ| > 0.70 pairs flagged)")
print("=" * 80)

n_m = len(NUMERIC_FIELDS)
rho_mat = np.zeros((n_m, n_m))
pval_mat = np.ones((n_m, n_m))
for i in range(n_m):
    for j in range(n_m):
        if i == j:
            rho_mat[i, j] = 1.0
        else:
            r, p = stats.spearmanr(mat[:, i], mat[:, j])
            rho_mat[i, j] = r
            pval_mat[i, j] = p

# Print compact correlation matrix (abbreviate names)
ABBR = {m: m[:12] for m in NUMERIC_FIELDS}
print("\nFull Spearman ρ matrix (rows/cols = metric index):")
idx_labels = [f"{i:02d}" for i in range(n_m)]
print("     " + " ".join(f"{l:>6}" for l in idx_labels))
for i, m in enumerate(NUMERIC_FIELDS):
    row = " ".join(f"{rho_mat[i,j]:>6.2f}" for j in range(n_m))
    print(f"{i:02d}   {row}  <- {m}")

print("\nHigh-correlation pairs (|ρ| > 0.70, i < j):")
found = False
for i in range(n_m):
    for j in range(i+1, n_m):
        if abs(rho_mat[i, j]) > 0.70:
            sig = "**" if pval_mat[i,j] < 0.01 else "*"
            print(f"  [{i:02d}] {NUMERIC_FIELDS[i]:<38} <-> [{j:02d}] {NUMERIC_FIELDS[j]:<38}  ρ={rho_mat[i,j]:+.3f} p={pval_mat[i,j]:.2e} {sig}")
            found = True
if not found:
    print("  (none)")

# ── 4. Hierarchical clustering ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 4 — HIERARCHICAL CLUSTERING (Ward, distance = 1 - |ρ|)")
print("=" * 80)

dist_mat = 1.0 - np.abs(rho_mat)
np.fill_diagonal(dist_mat, 0.0)
condensed = squareform(dist_mat, checks=False)
Z = linkage(condensed, method='ward')

# Cut at threshold 0.4 → clusters of |ρ| > 0.6 effectively
THRESHOLD = 0.4
labels = fcluster(Z, t=THRESHOLD, criterion='distance')

clusters: dict[int, list[str]] = {}
for i, c in enumerate(labels):
    clusters.setdefault(c, []).append(NUMERIC_FIELDS[i])

print(f"\nClusters at Ward distance threshold {THRESHOLD}  (metrics with |ρ|≥0.60 approximately):")
for c_id in sorted(clusters):
    members = clusters[c_id]
    print(f"  Cluster {c_id} ({len(members)} metrics): {', '.join(members)}")

# ── 5. Variance & discriminability ───────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 5 — VARIANCE & DISCRIMINABILITY ANALYSIS")
print("=" * 80)

pure_idx    = [i for i,v in enumerate(variants) if v == 'pure']
elicit_idx  = [i for i,v in enumerate(variants) if v == 'eliciting']
normal_idx  = [i for i,v in enumerate(variants) if v == 'normal']

pure_mat   = mat[pure_idx]
elicit_mat = mat[elicit_idx]
normal_mat = mat[normal_idx]

def cohens_d(a, b):
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (b.mean() - a.mean()) / pooled_std if pooled_std > 0 else 0.0

def anova_f(groups):
    grand_mean = np.concatenate(groups).mean()
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_within  = sum(((g - g.mean())**2).sum() for g in groups)
    df_b = k - 1
    df_w = n_total - k
    if df_w == 0 or ss_within == 0:
        return 0.0
    return (ss_between / df_b) / (ss_within / df_w)

header2 = f"{'Metric':<38} {'Var':>10} {'CohenD':>9} {'ANOVA-F':>9}"
print(header2)
print("-" * 70)

variance_scores = []
discriminability_scores = []

for i, m in enumerate(NUMERIC_FIELDS):
    var = mat[:, i].var()
    d = cohens_d(pure_mat[:, i], elicit_mat[:, i])
    f = anova_f([pure_mat[:, i], normal_mat[:, i], elicit_mat[:, i]])
    variance_scores.append(var)
    discriminability_scores.append(abs(d))
    print(f"{m:<38} {var:>10.4f} {d:>+9.3f} {f:>9.3f}")

# ── 6. Ranked recommendation table ───────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6 — FINAL METRIC RANKING AND VERDICT")
print("=" * 80)

# Rank by variance (desc) and discriminability (desc)
var_arr = np.array(variance_scores)
disc_arr = np.array(discriminability_scores)

var_ranks = stats.rankdata(-var_arr)          # rank 1 = highest variance
disc_ranks = stats.rankdata(-disc_arr)        # rank 1 = highest discriminability
composite = (var_ranks + disc_ranks) / 2.0

# Max |ρ| with any other metric
max_rho = []
for i in range(n_m):
    others = [abs(rho_mat[i, j]) for j in range(n_m) if j != i]
    max_rho.append(max(others))

# Verdict logic
CATEGORIES = {
    'branching_factor': 'Breadth', 'unique_perspective_count': 'Breadth',
    'domain_spread': 'Breadth', 'first_idea_diversity': 'Breadth',
    'max_elaboration_chain': 'Depth', 'mean_branch_depth': 'Depth',
    'specificity_gradient': 'Depth', 'reasoning_density': 'Depth',
    'exploration_exploitation_ratio': 'Structure', 'backtracking_rate': 'Structure',
    'cross_branch_connectivity': 'Structure', 'convergence_index': 'Structure',
    'graph_density': 'Structure', 'revision_depth': 'Structure',
    'self_reflection_rate': 'Metacog', 'critique_to_hypothesis_ratio': 'Metacog',
    'hedging_density': 'Metacog', 'perspective_taking': 'Metacog',
    'token_per_idea': 'Efficiency', 'redundancy_ratio': 'Efficiency',
    'avg_tokens': 'Summary', 'avg_tus': 'Summary',
}

verdicts = []
for i, m in enumerate(NUMERIC_FIELDS):
    zero_pct = (mat[:, i] == 0.0).mean() * 100
    std_val = mat[:, i].std()
    mr = max_rho[i]

    if zero_pct >= 95 and std_val < 0.001:
        verdict = "DROP"
        reason = f"constant 0 in {zero_pct:.0f}% traces — pipeline never generates this node/edge type"
    elif zero_pct >= 80:
        verdict = "DROP"
        reason = f"zero in {zero_pct:.0f}% traces — insufficient signal"
    elif m == 'token_per_idea':
        verdict = "DROP"
        reason = "bug: equals avg_tokens when unique_perspective_count=0; not a rate"
    elif m == 'domain_spread':
        verdict = "DROP"
        reason = "degenerate: constant 1.0 when <3 HYP/BRS nodes; measures nothing currently"
    elif m == 'avg_tokens' and 'token_per_idea' in NUMERIC_FIELDS:
        verdict = "KEEP"
        reason = "direct length measure; rename to response_length for clarity"
    elif mr >= 0.90:
        # Find the partner
        partner_idx = max((j for j in range(n_m) if j != i), key=lambda j: abs(rho_mat[i, j]))
        partner = NUMERIC_FIELDS[partner_idx]
        # Keep the one with higher composite rank (lower number)
        if composite[i] <= composite[partner_idx]:
            verdict = "KEEP"
            reason = f"|ρ|={mr:.2f} with {partner}; preferred (lower composite rank)"
        else:
            verdict = "MERGE"
            reason = f"|ρ|={mr:.2f} with {partner}; subsume into that metric"
    elif mr >= 0.70:
        partner_idx = max((j for j in range(n_m) if j != i), key=lambda j: abs(rho_mat[i, j]))
        partner = NUMERIC_FIELDS[partner_idx]
        verdict = "REVIEW"
        reason = f"|ρ|={mr:.2f} with {partner}; collinear but may capture distinct variance"
    else:
        verdict = "KEEP"
        reason = f"std={std_val:.4f}, zero%={zero_pct:.0f}%, max|ρ|={mr:.2f}"
    verdicts.append((composite[i], m, verdict, reason))

verdicts.sort(key=lambda x: x[0])

print(f"\n{'#':>3} {'Metric':<38} {'Cat':<10} {'Zero%':>6} {'Std':>8} {'Max|ρ|':>7} {'Verdict':<8} Reason")
print("-" * 130)
for rank_i, (score, m, verdict, reason) in enumerate(verdicts, 1):
    i = NUMERIC_FIELDS.index(m)
    zero_pct = (mat[:, i] == 0.0).mean() * 100
    std_val = mat[:, i].std()
    mr = max_rho[i]
    cat = CATEGORIES.get(m, '?')
    print(f"{rank_i:>3} {m:<38} {cat:<10} {zero_pct:>5.1f}% {std_val:>8.4f} {mr:>7.3f} {verdict:<8} {reason}")

print("\n--- SUMMARY ---")
keep  = [m for _, m, v, _ in verdicts if v == 'KEEP']
merge = [m for _, m, v, _ in verdicts if v == 'MERGE']
review= [m for _, m, v, _ in verdicts if v == 'REVIEW']
drop  = [m for _, m, v, _ in verdicts if v == 'DROP']
print(f"KEEP   ({len(keep)}):   {keep}")
print(f"REVIEW ({len(review)}): {review}")
print(f"MERGE  ({len(merge)}):  {merge}")
print(f"DROP   ({len(drop)}):   {drop}")
print(f"\nRecommended reduced set: {len(keep)} metrics")
