"""3-pass non-generative segmentation: regex hard boundaries → TextTiling soft boundaries → NLI edge promotion."""

import re
from dataclasses import dataclass

import numpy as np

from .schemas import (
    BoundaryClass,
    BOUNDARY_EDGE_MAP,
    Edge,
    EdgeType,
    ThoughtUnit,
)

# ---------------------------------------------------------------------------
# Pass 1 — cue-phrase lexicon (longer phrases first within each class)
# ---------------------------------------------------------------------------

_BOUNDARY_PATTERNS: list[tuple[BoundaryClass, list[str]]] = [
    (BoundaryClass.BACKTRACK, [
        r"no wait[,.]?", r"on second thought[,.]?", r"going back to",
        r"i was wrong[,.]?", r"that'?s not right[,.]?", r"let me reconsider",
        r"actually[,.]?", r"wait[,.]?", r"hmm[,.]?",
    ]),
    (BoundaryClass.BRANCH, [
        r"a different way to think(?: about this)?[,.]?",
        r"let me try a different angle[,.]?",
        r"what if instead[,.]?",
        r"another approach[,.]?",
        r"alternatively[,.]?",
        r"or perhaps[,.]?",
    ]),
    (BoundaryClass.META, [
        r"i need to be more careful[,.]?",
        r"this is getting circular[,.]?",
        r"i'?m overcomplicating this[,.]?",
        r"let me re-?read the question[,.]?",
        r"let me step back[,.]?",
    ]),
    (BoundaryClass.CONVERGENCE, [
        r"so the key insight is[,.]?",
        r"putting this together[,.]?",
        r"to summarize[,.]?",
        r"the answer is[,.]?",
        r"therefore[,.]?",
        r"this means that[,.]?",
    ]),
    (BoundaryClass.ELABORATION, [
        r"to be concrete[,.]?",
        r"more precisely[,.]?",
        r"in other words[,.]?",
        r"specifically[,.]?",
        r"for example[,.]?",
        r"that is[,.]?",
    ]),
    (BoundaryClass.CONTRAST, [
        r"on the other hand[,.]?",
        r"in contrast[,.]?",
        r"despite this[,.]?",
        r"however[,.]?",
        r"but[,.]?",
    ]),
    (BoundaryClass.SUPPORT, [
        r"this is because[,.]?",
        r"the reason is[,.]?",
        r"evidence for this[,:]?",
        r"because[,.]?",
        r"since[,.]?",
    ]),
]

_COMPILED: list[tuple[BoundaryClass, list[re.Pattern]]] = [
    (cls, [re.compile(r"^" + p, re.IGNORECASE) for p in patterns])
    for cls, patterns in _BOUNDARY_PATTERNS
]

_MIN_TOKENS = 50
_MIN_SENTENCES = 3


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p for p in parts if p]


def _detect_boundary_class(sentence: str) -> BoundaryClass:
    for cls, patterns in _COMPILED:
        for pat in patterns:
            if pat.match(sentence):
                return cls
    return BoundaryClass.NONE


@dataclass
class _RawSpan:
    text: str
    start_char: int
    end_char: int
    boundary_class: BoundaryClass


def _pass1_hard_boundaries(text: str) -> list[_RawSpan]:
    """Split on cue phrases and paragraph breaks; merge spans that are too short."""
    paragraphs: list[tuple[str, int]] = []
    pos = 0
    for para in re.split(r"\n\n+", text):
        paragraphs.append((para, pos))
        pos += len(para) + 2

    raw_spans: list[_RawSpan] = []

    for para_text, para_start in paragraphs:
        sentences = _split_sentences(para_text)
        if not sentences:
            continue

        current_sentences: list[str] = []
        current_start = para_start
        current_class = BoundaryClass.NONE
        char_offset = para_start

        for sent in sentences:
            bc = _detect_boundary_class(sent)
            if bc != BoundaryClass.NONE and current_sentences:
                span_text = " ".join(current_sentences)
                raw_spans.append(_RawSpan(
                    text=span_text,
                    start_char=current_start,
                    end_char=current_start + len(span_text),
                    boundary_class=current_class,
                ))
                current_start = char_offset
                current_sentences = [sent]
                current_class = bc
            else:
                current_sentences.append(sent)
            char_offset += len(sent) + 1

        if current_sentences:
            span_text = " ".join(current_sentences)
            raw_spans.append(_RawSpan(
                text=span_text,
                start_char=current_start,
                end_char=current_start + len(span_text),
                boundary_class=current_class,
            ))

    # Merge spans that are too short into the following span
    merged: list[_RawSpan] = []
    i = 0
    while i < len(raw_spans):
        span = raw_spans[i]
        sentences = _split_sentences(span.text)
        too_short = (
            len(sentences) < _MIN_SENTENCES
            and _approx_tokens(span.text) < _MIN_TOKENS
        )
        if too_short and i + 1 < len(raw_spans):
            nxt = raw_spans[i + 1]
            raw_spans[i + 1] = _RawSpan(
                text=span.text + " " + nxt.text,
                start_char=span.start_char,
                end_char=nxt.end_char,
                boundary_class=span.boundary_class,
            )
            i += 1
            continue
        merged.append(span)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Pass 2 — TextTiling with MiniLM embeddings
# ---------------------------------------------------------------------------

_WINDOW = 3
_PROMINENCE = 0.15
_CALIBRATION_PERCENTILE = 30  # τ = 30th percentile of within-trace similarities


def _embed(sentences: list[str], model) -> np.ndarray:
    return model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _soft_boundaries(span_text: str, embed_model, tau: float) -> list[int]:
    """Return 0-based sentence indices where a soft boundary should be inserted."""
    from scipy.signal import find_peaks

    sentences = _split_sentences(span_text)
    if len(sentences) < 2 * _WINDOW + 1:
        return []

    embs = _embed(sentences, embed_model)
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


def _pass2_soft_boundaries(
    hard_spans: list[_RawSpan],
    embed_model,
) -> list[_RawSpan]:
    # Compute τ from all adjacent-sentence similarities in this trace
    all_sims: list[float] = []
    for span in hard_spans:
        sents = _split_sentences(span.text)
        if len(sents) >= 2:
            embs = _embed(sents, embed_model)
            for i in range(len(embs) - 1):
                all_sims.append(_cosine(embs[i], embs[i + 1]))

    tau = float(np.percentile(all_sims, _CALIBRATION_PERCENTILE)) if all_sims else 0.5

    result: list[_RawSpan] = []
    for span in hard_spans:
        if span.boundary_class != BoundaryClass.NONE:
            result.append(span)
            continue

        sents = _split_sentences(span.text)
        boundaries = _soft_boundaries(span.text, embed_model, tau)
        if not boundaries:
            result.append(span)
            continue

        prev = 0
        char_offset = span.start_char
        for b in sorted(set(boundaries)):
            chunk = " ".join(sents[prev:b])
            result.append(_RawSpan(
                text=chunk,
                start_char=char_offset,
                end_char=char_offset + len(chunk),
                boundary_class=BoundaryClass.NONE,
            ))
            char_offset += len(chunk) + 1
            prev = b
        chunk = " ".join(sents[prev:])
        result.append(_RawSpan(
            text=chunk,
            start_char=char_offset,
            end_char=char_offset + len(chunk),
            boundary_class=BoundaryClass.NONE,
        ))

    return result


# ---------------------------------------------------------------------------
# Pass 3 — NLI edge promotion
# ---------------------------------------------------------------------------

_NLI_WINDOW = 4
_NLI_THRESH = 0.75
_NLI_BACK_THRESH = 0.70

_H_SUPP = "The second passage provides evidence or justification for the first."
_H_CONT = "The second passage contradicts or opposes the first."
_H_ELAB = "The second passage is a more specific or detailed version of the first."
_H_BACK = "The second passage revises or contradicts an earlier position."

_NLI_HYPS = [_H_SUPP, _H_CONT, _H_ELAB]
_NLI_TYPES = [EdgeType.SUPP, EdgeType.CONT, EdgeType.ELAB]


def _entailment_probs(nli_model, pairs: list[tuple[str, str]]) -> list[float]:
    """Return entailment probabilities for a batch of (premise, hypothesis) pairs.
    cross-encoder/nli-deberta-v3-large outputs [contradiction, neutral, entailment] logits."""
    logits = nli_model.predict(pairs)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs[:, 2].tolist()


def _pass3_nli_promotion(
    tus: list[ThoughtUnit],
    edges: list[Edge],
    nli_model,
) -> list[Edge]:
    result = list(edges)
    texts = [tu.text for tu in tus]

    # Build a set of existing SEQ edges for fast lookup
    seq_set = {(e.source, e.target) for e in result if e.edge_type == EdgeType.SEQ}

    # Collect all local-window pairs to promote in a single batch
    local_pairs: list[tuple[str, str]] = []
    local_meta: list[tuple[int, int, int]] = []  # (i, j, hyp_idx)
    for j in range(1, len(tus)):
        for offset in range(1, min(j + 1, _NLI_WINDOW + 1)):
            i = j - offset
            if (i, j) not in seq_set:
                continue
            for hyp_idx, h in enumerate(_NLI_HYPS):
                local_pairs.append((texts[i], h))
                local_meta.append((i, j, hyp_idx))

    # Collect BACKTRACK non-local pairs
    back_pairs: list[tuple[str, str]] = []
    back_meta: list[tuple[int, int]] = []  # (j, i_candidate)
    for j in range(_NLI_WINDOW + 1, len(tus)):
        if tus[j].boundary_class != BoundaryClass.BACKTRACK:
            continue
        for i in range(j - _NLI_WINDOW - 1, -1, -1):
            back_pairs.append((texts[i], _H_BACK))
            back_meta.append((j, i))

    # Single batched prediction for all pairs
    all_pairs = local_pairs + back_pairs
    if not all_pairs:
        return result
    scores = _entailment_probs(nli_model, all_pairs)

    # Process local-window results
    # Group by (i, j): take the best hypothesis
    best_for_pair: dict[tuple[int, int], tuple[float, EdgeType]] = {}
    for (i, j, hyp_idx), score in zip(local_meta, scores[:len(local_pairs)]):
        cur_best, _ = best_for_pair.get((i, j), (0.0, _NLI_TYPES[hyp_idx]))
        if score > cur_best:
            best_for_pair[(i, j)] = (score, _NLI_TYPES[hyp_idx])

    for (i, j), (best, etype) in best_for_pair.items():
        if best > _NLI_THRESH:
            result.append(Edge(source=i, target=j, edge_type=etype,
                               confidence=best, is_sequential=False))

    # Process BACKTRACK results — link each j to the first i above threshold
    back_scores = scores[len(local_pairs):]
    linked: set[int] = set()
    for (j, i), score in zip(back_meta, back_scores):
        if j in linked:
            continue
        if score > _NLI_BACK_THRESH:
            result.append(Edge(source=j, target=i, edge_type=EdgeType.BACK,
                               confidence=score, is_sequential=False))
            linked.add(j)

    return result


# ---------------------------------------------------------------------------
# Model accessors (delegates to shared singleton cache)
# ---------------------------------------------------------------------------

from ..utils.models import get_embed_model as _get_embed_model
from ..utils.models import get_nli_model as _get_nli_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment(raw_cot: str, use_nli: bool = True) -> tuple[list[ThoughtUnit], list[Edge]]:
    """
    Segment a raw <think> trace into ThoughtUnits and typed Edges.
    Returns (tus, edges) — edges include the full sequential backbone plus
    any promoted semantic edges.
    """
    if _approx_tokens(raw_cot) < 100:
        return [], []

    embed_model = _get_embed_model()

    hard_spans = _pass1_hard_boundaries(raw_cot)
    all_spans = _pass2_soft_boundaries(hard_spans, embed_model)

    if not all_spans:
        return [], []

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

    # Layer 1: sequential backbone + Layer 2: hard-boundary typed edges
    edges: list[Edge] = []
    for i in range(len(tus) - 1):
        j = i + 1
        bc = tus[j].boundary_class
        default_edge = BOUNDARY_EDGE_MAP[bc]

        edges.append(Edge(
            source=i, target=j,
            edge_type=EdgeType.SEQ,
            confidence=1.0,
            is_sequential=True,
        ))
        if bc != BoundaryClass.NONE and default_edge != EdgeType.SEQ:
            edges.append(Edge(
                source=i, target=j,
                edge_type=default_edge,
                confidence=1.0,
                is_sequential=False,
            ))

    # Layer 3: NLI promotion
    if use_nli and len(tus) > 1:
        try:
            nli_model = _get_nli_model()
            edges = _pass3_nli_promotion(tus, edges, nli_model)
        except Exception:
            pass  # NLI unavailable — keep backbone + hard edges

    return tus, edges
