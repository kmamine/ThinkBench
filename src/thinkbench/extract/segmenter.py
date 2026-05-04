"""
Semantic thought-graph segmentation pipeline.

Architecture (5 passes):
  Pass 0+1 — Structural pre-segmentation + cue-phrase classification
             Markdown headers, bold section headers, list items, and paragraph
             breaks establish the primary TU boundaries.
             META / CONVERGENCE / BACKTRACK cue phrases refine boundary classes.
  Pass 2  — TextTiling with MiniLM embeddings (soft boundaries within long NONE spans)
  Layer 3 — Semantic trajectory (BRCH / ELAB from per-trace adaptive cosine thresholds)
  Layer 4 — Cross-segment analysis (SYNT synthesis, BACK revision/loop detection)
  Pass 5  — NLI edge refinement (SUPP / CONT / ELAB on remaining SEQ transitions)

Design: Structural boundaries (markdown, lists, paragraphs) provide the first-pass
segmentation. Embedding-based analysis then assigns semantic edge types. Surface cue
phrases supply supplementary BoundaryClass signals for META/CONVERGENCE/BACKTRACK.
"""

import re
from dataclasses import dataclass

import numpy as np

from .schemas import BoundaryClass, Edge, EdgeType, ThoughtUnit

# ---------------------------------------------------------------------------
# Structural split patterns (Pass 0)
# Used before any embedding analysis to identify hard formatting boundaries.
# ---------------------------------------------------------------------------

# Markdown ATX headers: # to ######
_HEADER_RE    = re.compile(r'^(#{1,6})[ \t]+(.+)$')

# Standalone bold line used as a section header:
#   **Heading text** or **Heading text:**
# Conservative: only matches if the content does NOT end with sentence-final
# punctuation (.!?) — those are bold statements, not headers.
_BOLD_HDR_RE  = re.compile(r'^\*\*([^*\n]{2,80})\*\*\s*:?\s*$')

# Horizontal rules (--- / *** / ___)
_HRULE_RE     = re.compile(r'^[-*_]{3,}\s*$')

# Bullet list items: -, *, + (top-level or indented)
_BULLET_RE    = re.compile(r'^[ \t]*[-*+][ \t]+\S')

# Numbered list items: 1. or 1)
_NUMBERED_RE  = re.compile(r'^[ \t]*\d+[.)][ \t]+\S')

# Strip leading list marker from an item line
_LIST_STRIP   = re.compile(r'^[ \t]*(?:[-*+]|\d+[.)])[ \t]+')

# Strip markdown emphasis/code characters from header text
_MD_STRIP_RE  = re.compile(r'[\*\_\`\~]+')

# Sentence-ending punctuation (used for single-newline paragraph detection)
_SENT_END_RE  = re.compile(r'[.?!]\s*$')

# Inline bold label on the same line as its content: **Label:** prose...
# The colon may be inside OR outside the closing **: both **Label:** text
# and **Label**: text are common. Keep `:?` optional after `**`.
# Complements _BOLD_HDR_RE which only fires when bold text is the entire line.
_INLINE_BOLD_RE = re.compile(r'^\*\*([^*\n]{2,40})\*\*\s*:?\s+\S')


# ---------------------------------------------------------------------------
# Pass 1 — supplementary cue-phrase detection
# Detects META, CONVERGENCE, BACKTRACK only.
# BRANCH / ELABORATION / CONTRAST / SUPPORT are derived from structure
# (Pass 0) or the semantic trajectory (Layer 3).
# ---------------------------------------------------------------------------

_META_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"^i need to be more careful[,.]?",
    r"^this is getting circular[,.]?",
    r"^i'?m overcomplicating this[,.]?",
    r"^let me re-?read the question[,.]?",
    r"^let me step back[,.]?",
    r"^stepping back[,.]?",
    r"^at a higher level[,.]?",
    r"^thinking about this more carefully[,.]?",
    r"^i'?m going in circles[,.]?",
    r"^let me reconsider the (problem|question|approach)[,.]?",
    r"^i should reconsider[,.]?",
    r"^reflecting on (this|my reasoning)[,.]?",
]]

_CONVERGENCE_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"^so the key insight is[,.]?",
    r"^putting this (all )?together[,.]?",
    r"^to summarize[,:]?",
    r"^in summary[,:]?",
    r"^the answer is[,:]?",
    r"^therefore[,.]?",
    r"^this means that[,.]?",
    r"^in conclusion[,:]?",
    r"^balancing these[,.]?",
    r"^ultimately[,.]?",
    r"^bringing (this|these) together[,.]?",
    r"^my (final |overall )?recommendation[,:]?",
    r"^the (core|main|key) takeaway[,:]?",
    r"^synthesis[,:]?",
    r"^to conclude[,:]?",
    r"^final (answer|recommendation|verdict)[,:]?",
]]

_BACKTRACK_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"^no wait[,.]?",
    r"^on second thought[,.]?",
    r"^going back to",
    r"^i was wrong[,.]?",
    r"^that'?s not right[,.]?",
    r"^wait[,.]?",
    r"^hmm[,.]?",
    r"^but actually[,.]?",
    r"^on reflection[,.]?",
    r"^reconsidering[,.]?",
    r"^actually[,.]?",
    r"^hold on[,.]?",
]]


def _detect_supplementary_class(sentence: str) -> BoundaryClass:
    for pat in _META_PATTERNS:
        if pat.match(sentence):
            return BoundaryClass.META
    for pat in _CONVERGENCE_PATTERNS:
        if pat.match(sentence):
            return BoundaryClass.CONVERGENCE
    for pat in _BACKTRACK_PATTERNS:
        if pat.match(sentence):
            return BoundaryClass.BACKTRACK
    return BoundaryClass.NONE


_MIN_TOKENS = 30
_MIN_SENTENCES = 2


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p for p in parts if p]


@dataclass
class _RawSpan:
    text: str
    start_char: int
    end_char: int
    boundary_class: BoundaryClass
    is_list_item: bool = False


# ---------------------------------------------------------------------------
# Pass 0 — Structural pre-segmentation
#
# Splits text into raw spans by detecting hard structural markers:
#   Markdown headers (# to ######)    → BoundaryClass.BRANCH
#   Bold standalone headers (**X**)   → BoundaryClass.BRANCH
#   Horizontal rules (---, ***)       → separator only (no span emitted)
#   List items (- x, * x, 1. x)      → BoundaryClass.NONE, one span per item
#   Blank lines                       → paragraph separator
#   Single-newline paragraph breaks   → separator when preceded by a sentence end
#   Plain prose paragraphs            → BoundaryClass.NONE
#
# Cue-phrase classification (META/CONVERGENCE/BACKTRACK) is applied separately
# in Pass 1 after this function returns.
# ---------------------------------------------------------------------------

def _structural_split(text: str) -> list[_RawSpan]:
    """Split text into structural spans along markdown, list, and paragraph boundaries."""
    lines = text.split('\n')
    spans: list[_RawSpan] = []

    block_lines: list[str] = []
    block_start: int = 0
    block_bc: BoundaryClass = BoundaryClass.NONE
    block_is_list: bool = False  # tracks whether current block came from a list item

    char_pos: int = 0
    prev_stripped: str = ''

    def _flush() -> None:
        nonlocal block_lines, block_start, block_bc, block_is_list
        if not block_lines:
            return
        joined = ' '.join(bl for bl in block_lines if bl).strip()
        if joined:
            spans.append(_RawSpan(joined, block_start, block_start + len(joined),
                                  block_bc, block_is_list))
        block_lines = []
        block_bc = BoundaryClass.NONE
        block_is_list = False

    def _emit(text_: str, start_: int, end_: int, bc_: BoundaryClass,
              is_list: bool = False) -> None:
        t = text_.strip()
        if t:
            spans.append(_RawSpan(t, start_, end_, bc_, is_list))

    for line in lines:
        stripped = line.strip()
        line_start = char_pos
        line_end = char_pos + len(line)
        char_pos += len(line) + 1  # +1 for \n

        # ── Blank line → paragraph separator ──────────────────────────────
        if not stripped:
            _flush()
            prev_stripped = ''
            continue

        # ── Markdown header (# to ######) → BRANCH ────────────────────────
        m = _HEADER_RE.match(stripped)
        if m:
            _flush()
            header_text = _MD_STRIP_RE.sub('', m.group(2)).strip()
            # Strip leading numbering from headers like "1. Heading" or "1.2 Heading"
            header_text = re.sub(r'^\d+[\d.]*\.?\s+', '', header_text).strip()
            if header_text:
                _emit(header_text, line_start, line_end, BoundaryClass.BRANCH)
            prev_stripped = stripped
            continue

        # ── Bold standalone header (**Text** or **Text:**) → BRANCH ────────
        # Only match if the content does NOT end in sentence-final punctuation
        # (those are bold statements, not section headers).
        m = _BOLD_HDR_RE.match(stripped)
        if m:
            content = m.group(1).strip()
            # Treat as a header unless it looks like a bold prose sentence.
            # Exception: "Label: Subtitle." contains a colon → always a header,
            # even when it ends with a period (e.g. "**Thread A: Collateral Risk.**")
            ends_sentence = content.rstrip().endswith(('.', '!', '?'))
            has_label_structure = ':' in content
            if content and (not ends_sentence or has_label_structure):
                _flush()
                if content:
                    _emit(content, line_start, line_end, BoundaryClass.BRANCH)
                prev_stripped = stripped
                continue

        # ── Horizontal rule → flush only (no span emitted) ────────────────
        if _HRULE_RE.match(stripped):
            _flush()
            prev_stripped = stripped
            continue

        # ── List item (bullet or numbered) → one span per item ────────────
        is_bullet   = bool(_BULLET_RE.match(line))   # match original line (preserves indent)
        is_numbered = bool(_NUMBERED_RE.match(line))

        if is_bullet or is_numbered:
            _flush()
            indent = len(line) - len(line.lstrip())
            item_text = _LIST_STRIP.sub('', stripped).strip()
            if item_text:
                # A list item whose entire content is a bold header should be
                # treated as a BRANCH, not a list item.
                # e.g. "*   **Thread A: The 20% Collateral Risk.**" → BRANCH
                m_bold = _BOLD_HDR_RE.match(item_text)
                if m_bold:
                    bold_content = m_bold.group(1).strip()
                    ends_sentence = bold_content.rstrip().endswith(('.', '!', '?'))
                    has_label_structure = ':' in bold_content
                    if bold_content and (not ends_sentence or has_label_structure):
                        block_lines = [bold_content]
                        block_start = line_start
                        block_bc = BoundaryClass.BRANCH
                        block_is_list = False
                        prev_stripped = stripped
                        continue
                # Regular list item: detect indentation for nested sub-bullets.
                item_bc = BoundaryClass.ELABORATION if indent >= 4 else BoundaryClass.NONE
                block_lines = [item_text]
                block_start = line_start
                block_bc = item_bc
                block_is_list = True
            prev_stripped = stripped
            continue

        # ── Inline bold label (**Label:** prose...) → ELABORATION ─────────
        # Fires when bold text is followed by content on the same line.
        # Standalone bold headers (**Heading** alone) are caught above (→ BRANCH).
        if _INLINE_BOLD_RE.match(stripped) and not _BOLD_HDR_RE.match(stripped):
            _flush()
            _emit(stripped, line_start, line_end, BoundaryClass.ELABORATION)
            prev_stripped = stripped
            continue

        # ── Single-newline paragraph break detection ───────────────────────
        # If the previous non-blank line ended a sentence AND this line begins
        # with an uppercase letter (or with a list/header marker already
        # handled above), treat the newline as a paragraph separator.
        # This handles models that use \n rather than \n\n between paragraphs.
        if (block_lines
                and prev_stripped
                and _SENT_END_RE.search(prev_stripped)
                and stripped
                and stripped[0].isupper()
                and not stripped.startswith(('*', '-', '+'))):
            _flush()

        # ── Regular prose line → accumulate in current block ──────────────
        if not block_lines:
            block_start = line_start
        block_lines.append(stripped)
        prev_stripped = stripped

    _flush()
    return spans


# ---------------------------------------------------------------------------
# Pass 1 — Structural pre-segmentation + cue-phrase classification
#
# Calls _structural_split, then applies META/CONVERGENCE/BACKTRACK cue-phrase
# detection to the first sentence of each NONE-class span, then merges
# spans that are too short to be meaningful TUs.
# ---------------------------------------------------------------------------

def _pass1_supplementary(text: str) -> list[_RawSpan]:
    """Structural split + cue-phrase classification + short-span merging."""
    # Step 1: structural boundaries (markdown, lists, paragraphs)
    raw = _structural_split(text)

    # Step 2: apply cue-phrase classification to NONE-class spans only
    refined: list[_RawSpan] = []
    for span in raw:
        if span.boundary_class == BoundaryClass.NONE:
            sents = _split_sentences(span.text)
            first = sents[0] if sents else span.text
            bc = _detect_supplementary_class(first)
            if bc != BoundaryClass.NONE:
                # Preserve is_list_item so merge logic downstream stays correct
                span = _RawSpan(span.text, span.start_char, span.end_char,
                                bc, span.is_list_item)
        refined.append(span)

    # Step 3: merge spans that are too short.
    #
    # Rules:
    # a) A short BRANCH header (< 12 tokens) absorbs the NEXT span,
    #    keeping BRANCH class — the header becomes the TU's opening phrase.
    # b) A short NONE span merges FORWARD into the next span, BUT only when
    #    the next span is also NONE.  Never merge NONE into BRANCH (that would
    #    prepend unrelated prose to a section header, corrupting its class).
    # c) A short NONE span with no valid forward target is kept as-is.
    merged: list[_RawSpan] = []
    i = 0
    while i < len(refined):
        span = refined[i]

        # Rule (a): short BRANCH header → absorb next span
        if (span.boundary_class == BoundaryClass.BRANCH
                and _approx_tokens(span.text) < 12
                and i + 1 < len(refined)):
            nxt = refined[i + 1]
            refined[i + 1] = _RawSpan(
                text=span.text + ": " + nxt.text,
                start_char=span.start_char,
                end_char=nxt.end_char,
                boundary_class=BoundaryClass.BRANCH,
            )
            i += 1
            continue

        # Rule (b): short NONE span → merge forward into the next NONE span.
        # Guard: a short list item must NOT merge into a non-list (prose/header)
        # span — that would fuse the tail of one section into the start of another.
        too_short = (
            span.boundary_class == BoundaryClass.NONE
            and len(_split_sentences(span.text)) < _MIN_SENTENCES
            and _approx_tokens(span.text) < _MIN_TOKENS
        )
        next_is_none = (
            i + 1 < len(refined)
            and refined[i + 1].boundary_class == BoundaryClass.NONE
        )
        next_compatible = next_is_none and (
            not span.is_list_item or refined[i + 1].is_list_item
        )
        if too_short and next_compatible:
            nxt = refined[i + 1]
            refined[i + 1] = _RawSpan(
                text=span.text + " " + nxt.text,
                start_char=span.start_char,
                end_char=nxt.end_char,
                boundary_class=BoundaryClass.NONE,
                is_list_item=span.is_list_item and nxt.is_list_item,
            )
            i += 1
            continue

        merged.append(span)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Pass 2 — TextTiling with MiniLM embeddings (soft boundary detection)
# Runs only on NONE-class spans that weren't already split by Pass 0+1.
# ---------------------------------------------------------------------------

_WINDOW = 3
_PROMINENCE = 0.15
_CALIBRATION_PERCENTILE = 30


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _soft_boundaries(span_text: str, embed_model, tau: float) -> list[int]:
    from scipy.signal import find_peaks
    sentences = _split_sentences(span_text)
    if len(sentences) < 2 * _WINDOW + 1:
        return []
    embs = embed_model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
    sims = []
    for i in range(_WINDOW, len(sentences) - _WINDOW):
        left = embs[i - _WINDOW:i].mean(axis=0)
        right = embs[i:i + _WINDOW].mean(axis=0)
        sims.append(_cosine(left, right))
    if not sims:
        return []
    sims_arr = np.array(sims)
    peaks, _ = find_peaks(-sims_arr, prominence=_PROMINENCE)
    return [int(p) + _WINDOW for p in peaks if sims_arr[p] < tau]


def _pass2_soft_boundaries(hard_spans: list[_RawSpan], embed_model) -> list[_RawSpan]:
    all_sims: list[float] = []
    for span in hard_spans:
        sents = _split_sentences(span.text)
        if len(sents) >= 2:
            embs = embed_model.encode(sents, show_progress_bar=False, normalize_embeddings=True)
            for i in range(len(embs) - 1):
                all_sims.append(_cosine(embs[i], embs[i + 1]))
    tau = float(np.percentile(all_sims, _CALIBRATION_PERCENTILE)) if all_sims else 0.5

    result: list[_RawSpan] = []
    for span in hard_spans:
        if span.boundary_class != BoundaryClass.NONE:
            result.append(span)
            continue
        boundaries = _soft_boundaries(span.text, embed_model, tau)
        if not boundaries:
            result.append(span)
            continue
        sents = _split_sentences(span.text)
        prev = 0
        char_offset = span.start_char
        for b in sorted(set(boundaries)):
            chunk = " ".join(sents[prev:b])
            if chunk.strip():
                result.append(_RawSpan(chunk, char_offset,
                                       char_offset + len(chunk), BoundaryClass.NONE))
            char_offset += len(chunk) + 1
            prev = b
        chunk = " ".join(sents[prev:])
        if chunk.strip():
            result.append(_RawSpan(chunk, char_offset,
                                   char_offset + len(chunk), BoundaryClass.NONE))
    return result


# ---------------------------------------------------------------------------
# Layer 3 — Semantic trajectory: BRCH / ELAB from cosine similarity
#
# For consecutive TU pair (i, i+1):
#   sim < τ_branch  →  BRCH edge  (topic shift = new branch)
#   sim ≥ τ_elab    →  ELAB edge  (staying deep on same topic)
#   else            →  SEQ        (neutral transition)
#
# τ_branch = 25th percentile of local similarities (adaptive per-trace)
# τ_elab   = 65th percentile of local similarities (adaptive per-trace)
#
# Structural boundaries (BRANCH class from markdown/lists) override the
# cosine classification and always emit a BRCH edge, matching the intent
# of the model's own formatting.
# Cue-phrase boundaries (META/CONVERGENCE/BACKTRACK) also override.
# ---------------------------------------------------------------------------

_BRANCH_PERCENTILE = 25
_ELAB_PERCENTILE = 65


def _semantic_trajectory(
    tus: list[ThoughtUnit],
    embeddings: np.ndarray,
) -> tuple[list[Edge], list[int]]:
    """
    Compute BRCH/ELAB/SEQ edges from the cosine similarity trajectory.

    Returns
    -------
    edges       : list of Edge (SEQ backbone + semantic typed edges)
    segment_ids : topic segment index for each TU (segments separated by BRCH points)
    """
    n = len(tus)
    if n < 2:
        return [], [0] * n

    local_sim = np.array([
        _cosine(embeddings[i], embeddings[i - 1]) for i in range(1, n)
    ])

    τ_branch = float(np.percentile(local_sim, _BRANCH_PERCENTILE))
    τ_elab = float(np.percentile(local_sim, _ELAB_PERCENTILE))

    edges: list[Edge] = []
    segment_ids = [0] * n
    seg_id = 0

    for i in range(1, n):
        sim = float(local_sim[i - 1])
        bc = tus[i].boundary_class

        # Structural and cue-phrase overrides (highest priority)
        if bc == BoundaryClass.BRANCH:
            # Markdown header or bold section header → explicit new branch
            etype = EdgeType.BRCH
            seg_id += 1
        elif bc == BoundaryClass.CONVERGENCE:
            etype = EdgeType.SYNT
        elif bc == BoundaryClass.BACKTRACK:
            etype = EdgeType.BACK
            seg_id += 1
        elif bc == BoundaryClass.META:
            etype = EdgeType.SEQ  # meta-reflection is in-place, not a branch
        # Embedding-based classification
        elif sim < τ_branch:
            etype = EdgeType.BRCH
            seg_id += 1
        elif sim >= τ_elab:
            etype = EdgeType.ELAB
        else:
            etype = EdgeType.SEQ

        segment_ids[i] = seg_id

        # Always emit the SEQ backbone edge
        edges.append(Edge(source=i - 1, target=i, edge_type=EdgeType.SEQ,
                          confidence=1.0, is_sequential=True))
        # Emit typed edge if structurally meaningful
        if etype != EdgeType.SEQ:
            edges.append(Edge(source=i - 1, target=i, edge_type=etype,
                              confidence=round(abs(sim), 4), is_sequential=False))

    return edges, segment_ids


# ---------------------------------------------------------------------------
# Layer 4 — Cross-segment analysis: synthesis (SYNT) and revision loops (BACK)
# ---------------------------------------------------------------------------

_SYN_THRESH = 0.50
_SYN_MIN_SEGS = 2
_BACK_THRESH = 0.65
_BACK_DROP = 0.45
_BACK_MIN_GAP = 4
_BACK_MAX_LOOKBACK = 25


def _cross_segment_edges(
    tus: list[ThoughtUnit],
    embeddings: np.ndarray,
    segment_ids: list[int],
) -> list[Edge]:
    n = len(tus)
    n_segs = max(segment_ids) + 1
    new_edges: list[Edge] = []

    centroids: list[np.ndarray | None] = []
    seg_last_tu: list[int] = []
    for s in range(n_segs):
        indices = [i for i, sid in enumerate(segment_ids) if sid == s]
        centroids.append(embeddings[indices].mean(axis=0) if indices else None)
        seg_last_tu.append(indices[-1] if indices else 0)

    # Synthesis detection
    for j in range(1, n):
        seg_j = segment_ids[j]
        bridged: list[int] = []
        for s in range(seg_j):
            if centroids[s] is None:
                continue
            if _cosine(embeddings[j], centroids[s]) > _SYN_THRESH:
                bridged.append(s)
        if len(bridged) >= _SYN_MIN_SEGS:
            for s in bridged:
                rep = seg_last_tu[s]
                if rep != j:
                    new_edges.append(Edge(
                        source=j, target=rep, edge_type=EdgeType.SYNT,
                        confidence=round(_cosine(embeddings[j], embeddings[rep]), 4),
                        is_sequential=False,
                    ))
            if tus[j].boundary_class == BoundaryClass.NONE:
                tus[j].boundary_class = BoundaryClass.CONVERGENCE

    # Revision / loop detection
    already_linked: set[int] = set()
    for j in range(_BACK_MIN_GAP + 1, n):
        if j in already_linked:
            continue
        seg_j = segment_ids[j]
        best_i: int | None = None
        best_sim = _BACK_THRESH

        for i in range(max(0, j - _BACK_MAX_LOOKBACK), j - _BACK_MIN_GAP):
            if segment_ids[i] == seg_j:
                continue
            sim_ji = _cosine(embeddings[j], embeddings[i])
            if sim_ji <= best_sim:
                continue
            gap_end = min(i + 4, j)
            gap_sims = [_cosine(embeddings[k], embeddings[i]) for k in range(i + 1, gap_end)]
            if gap_sims and min(gap_sims) < _BACK_DROP:
                best_i, best_sim = i, sim_ji

        if best_i is not None:
            new_edges.append(Edge(
                source=j, target=best_i, edge_type=EdgeType.BACK,
                confidence=round(best_sim, 4), is_sequential=False,
            ))
            already_linked.add(j)
            # Do not override boundary_class here: thematic similarity to an
            # earlier TU (the criterion for BACK edges) does not imply the
            # current TU is a revision — it may simply share a topic. Forcing
            # BACKTRACK on list items and inline-bold labels produces false CRT
            # nodes. The BACK edge itself carries the revision signal.

    return new_edges


# ---------------------------------------------------------------------------
# Pass 5 — NLI edge refinement
# Promotes remaining SEQ transitions to SUPP / CONT / ELAB.
# Threshold 0.78 reduces near-universal SUPP over-triggering.
# ---------------------------------------------------------------------------

_NLI_WINDOW = 3
_NLI_THRESH = 0.78

_H_SUPP = "The second passage provides evidence or justification for the first."
_H_CONT = "The second passage contradicts or opposes the first."
_H_ELAB = "The second passage is a more specific or detailed version of the first."
_NLI_HYPS = [_H_SUPP, _H_CONT, _H_ELAB]
_NLI_TYPES = [EdgeType.SUPP, EdgeType.CONT, EdgeType.ELAB]


def _entailment_probs(nli_model, pairs: list[tuple[str, str]]) -> list[float]:
    logits = nli_model.predict(pairs)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs[:, 2].tolist()


def _pass5_nli_refinement(
    tus: list[ThoughtUnit],
    edges: list[Edge],
    nli_model,
) -> list[Edge]:
    result = list(edges)
    texts = [tu.text[:500] for tu in tus]

    typed_pairs: set[tuple[int, int]] = {
        (e.source, e.target) for e in result if not e.is_sequential
    }

    pairs: list[tuple[str, str]] = []
    meta: list[tuple[int, int, int]] = []
    for j in range(1, len(tus)):
        for offset in range(1, min(j + 1, _NLI_WINDOW + 1)):
            i = j - offset
            if (i, j) in typed_pairs:
                continue
            for hyp_idx, h in enumerate(_NLI_HYPS):
                pairs.append((texts[i], h))
                meta.append((i, j, hyp_idx))

    if not pairs:
        return result

    scores = _entailment_probs(nli_model, pairs)
    best_for_pair: dict[tuple[int, int], tuple[float, EdgeType]] = {}
    for (i, j, hyp_idx), score in zip(meta, scores):
        cur_best, _ = best_for_pair.get((i, j), (0.0, _NLI_TYPES[hyp_idx]))
        if score > cur_best:
            best_for_pair[(i, j)] = (score, _NLI_TYPES[hyp_idx])

    for (i, j), (best, etype) in best_for_pair.items():
        if best > _NLI_THRESH:
            result.append(Edge(source=i, target=j, edge_type=etype,
                               confidence=round(best, 4), is_sequential=False))

    return result


# ---------------------------------------------------------------------------
# Model accessors
# ---------------------------------------------------------------------------

from ..utils.models import get_embed_model as _get_embed_model
from ..utils.models import get_nli_model as _get_nli_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment(raw_cot: str, use_nli: bool = True) -> tuple[list[ThoughtUnit], list[Edge], np.ndarray]:
    """
    Segment a raw CoT trace into ThoughtUnits, typed Edges, and TU embeddings.

    Segmentation priority:
      1. Markdown headers (### Heading)     → BRANCH boundary (hard)
      2. Bold section headers (**Heading**) → BRANCH boundary (hard)
      3. List items (- item, 1. item)       → NONE per item (hard)
      4. Horizontal rules (---)             → paragraph separator
      5. Blank lines / double newlines      → paragraph separator
      6. Single-newline paragraph breaks    → separator (sentence-end + uppercase)
      7. META/CONVERGENCE/BACKTRACK cues    → class override on first sentence
      8. TextTiling (MiniLM embeddings)     → soft breaks within long NONE spans
      9. Semantic trajectory (Layer 3)      → BRCH / ELAB edges
     10. Cross-segment analysis (Layer 4)   → SYNT / BACK edges
     11. NLI refinement (Pass 5, optional)  → SUPP / CONT / ELAB edges

    Returns
    -------
    tus        : list[ThoughtUnit] with boundary_class set
    edges      : list[Edge] — SEQ backbone + semantic typed edges
    embeddings : np.ndarray of shape (n_tus, embed_dim) — normalized TU embeddings
    """
    if _approx_tokens(raw_cot) < 80:
        return [], [], np.zeros((0, 384))

    embed_model = _get_embed_model()

    hard_spans = _pass1_supplementary(raw_cot)
    all_spans = _pass2_soft_boundaries(hard_spans, embed_model)

    if not all_spans:
        return [], [], np.zeros((0, 384))

    tus = [
        ThoughtUnit(
            tu_id=idx,
            text=span.text,
            start_char=span.start_char,
            end_char=span.end_char,
            token_count=_approx_tokens(span.text),
            boundary_class=span.boundary_class,
        )
        for idx, span in enumerate(all_spans)
    ]

    if len(tus) < 2:
        return tus, [], np.zeros((0, 384))

    embeddings: np.ndarray = embed_model.encode(
        [tu.text[:400] for tu in tus],
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # Layer 3: semantic trajectory → BRCH / ELAB edges + segment IDs
    edges, segment_ids = _semantic_trajectory(tus, embeddings)

    # Layer 4: cross-segment synthesis + revision loops
    edges.extend(_cross_segment_edges(tus, embeddings, segment_ids))

    # Pass 5: NLI refinement of remaining SEQ transitions
    if use_nli and len(tus) > 1:
        try:
            edges = _pass5_nli_refinement(tus, edges, _get_nli_model())
        except Exception:
            pass

    return tus, edges, embeddings
