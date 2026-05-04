"""Publication-quality pipeline figure — clean blocks and arrows only."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = Path("results/pipeline_v3.png")

# ── Palette ─────────────────────────────────────────────────────────────────
GRAY_BG  = "#F7F8FA"
P1_COL   = "#5E35B1"   # purple   – cue-phrase
P2_COL   = "#1565C0"   # blue     – TextTiling
ASM_COL  = "#37474F"   # slate    – assembly/embed
L3_COL   = "#BF360C"   # burnt    – semantic trajectory
L4_COL   = "#880E4F"   # crimson  – cross-segment
P5_COL   = "#546E7A"   # mid-gray – NLI (optional)
CLS_COL  = "#1B5E20"   # forest   – classifier
GA_COL   = "#0D47A1"   # navy     – graph assembly
MC_COL   = "#4A148C"   # deep purple – metrics
IO_COL   = "#212121"   # near-black – I/O

# region fill colours (very light)
SEG_FILL = "#EDE7F6"   # light purple
EDGE_FILL= "#FBE9E7"   # light orange
ANA_FILL = "#E8F5E9"   # light green

ARROW_COL = "#37474F"
TEXT_LIGHT = "#FFFFFF"

# ── Figure setup ─────────────────────────────────────────────────────────────
FIG_W, FIG_H = 9, 16
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=GRAY_BG)
ax.set_facecolor(GRAY_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── Layout constants ──────────────────────────────────────────────────────────
BX  = FIG_W / 2   # block centre-x
BW  = 6.8         # block width
BH  = 0.72        # block height (normal)
BH_IO = 0.54      # block height (I/O)
GAP = 0.42        # gap between block bottom and next block top
ARH = GAP * 0.55  # arrow covers this fraction of the gap

# ── Helpers ───────────────────────────────────────────────────────────────────
def block(y_top, title, subtitle, color, height=BH, dashed=False):
    """Draw a rounded rectangle block; y_top is the top edge in data coords."""
    x0 = BX - BW / 2
    lw = 1.2 if not dashed else 1.5
    ls = "--" if dashed else "-"
    fc = color if not dashed else "#ECF0F1"
    ec = color

    rect = FancyBboxPatch(
        (x0, y_top - height), BW, height,
        boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls,
        zorder=3,
    )
    ax.add_patch(rect)

    tc = TEXT_LIGHT if not dashed else color
    # Title
    ax.text(BX, y_top - height * 0.36, title,
            ha="center", va="center",
            fontsize=9.8, fontweight="bold", color=tc,
            fontfamily="DejaVu Sans", zorder=4)
    # Subtitle
    if subtitle:
        ax.text(BX, y_top - height * 0.72, subtitle,
                ha="center", va="center",
                fontsize=7.8,
                color=TEXT_LIGHT if not dashed else "#555555",
                fontfamily="DejaVu Sans", zorder=4,
                alpha=0.88)
    return y_top - height  # returns bottom y of this block


def arrow_with_label(y_from, y_to, label_text=""):
    """Draw a downward arrow and an optional label beside it."""
    mid_y = (y_from + y_to) / 2
    ax.annotate(
        "", xy=(BX, y_to + 0.03), xytext=(BX, y_from - 0.03),
        arrowprops=dict(
            arrowstyle="->,head_width=0.28,head_length=0.16",
            color=ARROW_COL, lw=1.8,
        ),
        zorder=2,
    )
    if label_text:
        ax.text(BX + BW / 2 + 0.12, mid_y, label_text,
                ha="left", va="center",
                fontsize=7.0, color="#555555",
                fontfamily="DejaVu Sans",
                style="italic", zorder=4)


def region(y_top, y_bot, fill, label_text, label_color):
    """Shaded background region grouping several blocks."""
    pad = 0.18
    rect = mpatches.FancyBboxPatch(
        (BX - BW / 2 - pad, y_bot - pad),
        BW + 2 * pad, y_top - y_bot + 2 * pad,
        boxstyle="round,pad=0.0",
        facecolor=fill, edgecolor="none",
        alpha=0.55, zorder=1,
    )
    ax.add_patch(rect)
    # region label on the left
    ax.text(BX - BW / 2 - pad - 0.08, (y_top + y_bot) / 2, label_text,
            ha="right", va="center",
            fontsize=8, fontweight="bold", color=label_color,
            fontfamily="DejaVu Sans", rotation=90, zorder=2)


# ═════════════════════════════════════════════════════════════════════════════
# TITLE
# ═════════════════════════════════════════════════════════════════════════════
ax.text(BX, FIG_H - 0.25,
        "ThinkBench v3 — Pipeline",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1A1A2E",
        fontfamily="DejaVu Sans")
ax.text(BX, FIG_H - 0.58,
        "Raw chain-of-thought  →  22-dimensional Cognitive Profile",
        ha="center", va="center",
        fontsize=8.5, color="#555",
        fontfamily="DejaVu Sans")
ax.plot([0.6, FIG_W - 0.6], [FIG_H - 0.76, FIG_H - 0.76],
        color="#CCCCCC", lw=0.8)

# ═════════════════════════════════════════════════════════════════════════════
# BLOCKS  (placed top → bottom)
# ═════════════════════════════════════════════════════════════════════════════
y = FIG_H - 1.10

# ── INPUT ────────────────────────────────────────────────────────────────────
y_in_top = y
y = block(y, "Raw CoT Trace", "free-form reasoning text", IO_COL, height=BH_IO)
y_in_bot = y

arrow_with_label(y, y - GAP + ARH)
y -= GAP

# ── SEGMENTATION REGION ───────────────────────────────────────────────────────
seg_top = y

y_p1_top = y
y = block(y, "Pass 1 — Cue-Phrase Segmentation",
          "META · CONVERGENCE · BACKTRACK  via regex patterns", P1_COL)
y_p1_bot = y
arrow_with_label(y, y - GAP + ARH, "paragraph spans")
y -= GAP

y_p2_top = y
y = block(y, "Pass 2 — TextTiling",
          "MiniLM-L6-v2 · sliding window w=3 · τ = 30th pct", P2_COL)
y_p2_bot = y
arrow_with_label(y, y - GAP + ARH, "segmented spans")
y -= GAP

y_asm_top = y
y = block(y, "ThoughtUnit Assembly  +  Batch Embedding",
          "span → ThoughtUnit · encode all TUs → ℝ³⁸⁴  (all-MiniLM-L6-v2)", ASM_COL)
y_asm_bot = y

seg_bot = y
region(seg_top, seg_bot, SEG_FILL, "Segmentation", P1_COL)

arrow_with_label(y, y - GAP + ARH, "TUs  +  embeddings ∈ ℝⁿˣ³⁸⁴")
y -= GAP

# ── EDGE ASSIGNMENT REGION ────────────────────────────────────────────────────
edge_top = y

y_l3_top = y
y = block(y, "Layer 3 — Semantic Trajectory",
          "s_i = cos(eᵢ₋₁, eᵢ)   τ_branch=25th pct → BRCH   τ_elab=65th pct → ELAB", L3_COL)
y_l3_bot = y
arrow_with_label(y, y - GAP + ARH, "SEQ + BRCH + ELAB edges")
y -= GAP

y_l4_top = y
y = block(y, "Layer 4 — Cross-Segment Analysis",
          "centroid sim > 0.50, ≥2 segs → SYNT   gap-drop criterion → BACK", L4_COL)
y_l4_bot = y
arrow_with_label(y, y - GAP + ARH, "+  SYNT  +  BACK  edges")
y -= GAP

y_p5_top = y
y = block(y, "Pass 5 — NLI Refinement  [optional]",
          "DeBERTa cross-encoder · untyped SEQ only · threshold = 0.78",
          P5_COL, dashed=True)
y_p5_bot = y

edge_bot = y
region(edge_top, edge_bot, EDGE_FILL, "Edge Assignment", L3_COL)

arrow_with_label(y, y - GAP + ARH, "+  SUPP  +  CONT  +  ELAB  edges")
y -= GAP

# ── ANALYSIS REGION ───────────────────────────────────────────────────────────
ana_top = y

y_cls_top = y
y = block(y, "Node Classifier  (8-priority rules)",
          "SYNT hub→SYN · BACK→CRT · BRCH+sim→RFR/HYP · emb-MET · default SPC", CLS_COL)
y_cls_bot = y
arrow_with_label(y, y - GAP + ARH, "node_type + node_family per TU")
y -= GAP

y_ga_top = y
y = block(y, "ThoughtGraph  G = (V, E)",
          "directed graph with cycles · ELAB subgraph verified acyclic", GA_COL)
y_ga_bot = y
arrow_with_label(y, y - GAP + ARH)
y -= GAP

y_mc_top = y
y = block(y, "Metric Computation  (22 dimensions)",
          "Breadth · Depth · Structure · Metacognitive · Efficiency", MC_COL)
y_mc_bot = y

ana_bot = y
region(ana_top, ana_bot, ANA_FILL, "Graph & Metrics", CLS_COL)

arrow_with_label(y, y - GAP + ARH)
y -= GAP

# ── OUTPUT ───────────────────────────────────────────────────────────────────
y = block(y, "CognitiveProfile  ∈  ℝ²²",
          "model signature · prompt-invariant subset · archetype clustering",
          IO_COL, height=BH_IO)

# ═════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═════════════════════════════════════════════════════════════════════════════
LEG_Y = y - 0.50
ax.plot([0.6, FIG_W - 0.6], [LEG_Y + 0.05, LEG_Y + 0.05],
        color="#CCCCCC", lw=0.8)

legend_items = [
    (P1_COL,  "Cue-phrase pass",   False),
    (L3_COL,  "Embedding-based",   False),
    (P5_COL,  "NLI (optional)",    True),
    (IO_COL,  "Input / Output",    False),
]
lx = [1.4, 3.4, 5.5, 7.4]
for (col, lbl, dash), x in zip(legend_items, lx):
    rect = FancyBboxPatch(
        (x - 0.22, LEG_Y - 0.50), 0.44, 0.28,
        boxstyle="round,pad=0.06",
        facecolor=col if not dash else "#ECF0F1",
        edgecolor=col, linewidth=1.2,
        linestyle="--" if dash else "-",
    )
    ax.add_patch(rect)
    ax.text(x + 0.32, LEG_Y - 0.36, lbl,
            ha="left", va="center",
            fontsize=7.2, color="#333",
            fontfamily="DejaVu Sans")

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor=GRAY_BG)
print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
plt.close(fig)
