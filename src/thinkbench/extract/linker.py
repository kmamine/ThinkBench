"""Thought graph extraction: Linker - Edge detection and graph assembly."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..collect.models import LLMClient
from .schemas import ThoughtUnit, Edge, EdgeType, ThoughtGraph, NodeType

logger = logging.getLogger(__name__)

EDGE_CLASSIFY_PROMPT = """Given two thought units from a reasoning trace, determine their relationship.

Thought unit A (earlier): "{tu_a_text}"
Thought unit B (later): "{tu_b_text}"

Possible relationships:
- ELAB: B deepens/specifies A
- BRCH: B starts a new direction from A's setup  
- BACK: B revises or abandons A
- CRIT: B challenges or critiques A
- SUPP: B provides evidence for A
- CONT: B contrasts with A (opposing perspective)
- NONE: No meaningful relationship beyond sequence

Output valid JSON only:"""

SYNTHESIS_VERIFY_PROMPT = """Does this synthesis thought unit draw on the idea expressed in the candidate source?

Synthesis TU: "{syn_text}"
Candidate source TU: "{candidate_text}"

Output valid JSON only:"""

BACKTRACK_PROMPT = """Given a thought unit that appears to revise or revisit an earlier idea, identify what it is revising.

Current thought unit: "{current_text}"

Previous thought units:
{prev_units}

Output valid JSON only:"""


class Linker:
    def __init__(
        self,
        client: LLMClient,
        local_window: int = 5,
        confidence_threshold: float = 0.7,
    ):
        self.client = client
        self.local_window = local_window
        self.confidence_threshold = confidence_threshold

    async def link(self, thought_units: list[ThoughtUnit]) -> list[Edge]:
        """Build edges using 4-pass algorithm."""
        if not thought_units:
            return []

        edges = []

        edges.extend(self._pass1_sequential(thought_units))
        edges = await self._pass2_local_semantic(thought_units, edges)
        edges = await self._pass3_synthesis(thought_units, edges)
        edges = await self._pass4_backtracking(thought_units, edges)

        logger.info(f"Built {len(edges)} edges for {len(thought_units)} nodes")
        return edges

    def _pass1_sequential(self, nodes: list[ThoughtUnit]) -> list[Edge]:
        """Pass 1: Sequential backbone."""
        edges = []
        for i in range(len(nodes) - 1):
            edges.append(
                Edge(source=i, target=i + 1, edge_type=EdgeType.SEQ, confidence=1.0)
            )
        return edges

    async def _pass2_local_semantic(
        self, nodes: list[ThoughtUnit], edges: list[Edge]
    ) -> list[Edge]:
        """Pass 2: Local semantic linking (window=5)."""
        new_edges = []

        for i in range(1, len(nodes)):
            found_semantic = False
            for j in range(max(0, i - self.local_window), i):
                prompt = EDGE_CLASSIFY_PROMPT.format(
                    tu_a_text=nodes[j].text[:500], tu_b_text=nodes[i].text[:500]
                )
                messages = [{"role": "user", "content": prompt}]

                try:
                    response = await self.client.chat(
                        messages=messages, temperature=0.1, max_tokens=128
                    )
                    result = self._parse_edge_response(response)
                except Exception as e:
                    logger.warning(f"Edge classification failed for {j}->{i}: {e}")
                    continue

                if (
                    result
                    and result.get("relation") != "NONE"
                    and result.get("confidence", 0) >= self.confidence_threshold
                ):
                    try:
                        edge_type = EdgeType(result["relation"])
                    except ValueError:
                        continue

                    new_edges.append(
                        Edge(
                            source=j,
                            target=i,
                            edge_type=edge_type,
                            confidence=result["confidence"],
                        )
                    )
                    found_semantic = True

            # Fallback to rule-based if no semantic edge found
            if not found_semantic:
                edge_result = self._classify_edge_rules(
                    nodes[i - 1].text, nodes[i].text
                )
                if edge_result:
                    new_edges.append(
                        Edge(
                            source=i - 1,
                            target=i,
                            edge_type=edge_result["type"],
                            confidence=edge_result["confidence"],
                        )
                    )
                    # Remove SEQ between adjacent nodes
                    edges = [
                        e
                        for e in edges
                        if not (
                            e.source == i - 1
                            and e.target == i
                            and e.edge_type == EdgeType.SEQ
                        )
                    ]

        edges.extend(new_edges)
        return edges

    async def _pass3_synthesis(
        self, nodes: list[ThoughtUnit], edges: list[Edge]
    ) -> list[Edge]:
        """Pass 3: Long-range synthesis detection."""
        synth_nodes = [n for n in nodes if n.node_type == NodeType.SYN]

        for syn in synth_nodes:
            for j in range(syn.tu_id):
                if j >= syn.tu_id:
                    continue

                prompt = SYNTHESIS_VERIFY_PROMPT.format(
                    syn_text=syn.text[:300], candidate_text=nodes[j].text[:300]
                )
                messages = [{"role": "user", "content": prompt}]

                try:
                    response = await self.client.chat(
                        messages=messages, temperature=0.1, max_tokens=128
                    )
                    result = self._parse_verification_response(response)
                except Exception:
                    continue

                if (
                    result
                    and result.get("draws_from")
                    and result.get("confidence", 0) >= self.confidence_threshold
                ):
                    edges.append(
                        Edge(
                            source=j,
                            target=syn.tu_id,
                            edge_type=EdgeType.SYNT,
                            confidence=result["confidence"],
                        )
                    )

        return edges

    async def _pass4_backtracking(
        self, nodes: list[ThoughtUnit], edges: list[Edge]
    ) -> list[Edge]:
        """Pass 4: Backtracking detection."""
        backtrack_markers = [
            "Wait,",
            "Actually,",
            "Let me reconsider",
            "I'm going in circles",
            "Let me think again",
        ]

        for i, node in enumerate(nodes):
            is_critique = node.node_type == NodeType.CRT
            has_marker = any(
                marker.lower() in node.text.lower() for marker in backtrack_markers
            )

            if is_critique or has_marker:
                target = await self._find_backtrack_target(node, nodes)
                if target:
                    edge_type = EdgeType.CRIT if is_critique else EdgeType.BACK
                    edges.append(
                        Edge(
                            source=i,
                            target=target["target_id"],
                            edge_type=edge_type,
                            confidence=target["confidence"],
                        )
                    )

        return edges

    async def _find_backtrack_target(
        self, node: ThoughtUnit, nodes: list[ThoughtUnit]
    ) -> Optional[dict]:
        """Find what the backtracking/critique node is targeting."""
        prev_units = "\n".join(
            [f"[{n.tu_id}] {n.text[:150]}" for n in nodes[: node.tu_id][-5:]]
        )

        prompt = BACKTRACK_PROMPT.format(
            current_text=node.text[:300], prev_units=prev_units
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.client.chat(
                messages=messages, temperature=0.1, max_tokens=128
            )
            return self._parse_backtrack_response(response)
        except Exception as e:
            logger.warning(f"Backtrack target identification failed: {e}")
            return None

    def _parse_edge_response(self, response: str) -> Optional[dict]:
        """Parse edge classification response."""
        import re

        text = response.strip()

        # Try to find edge type directly
        type_match = re.search(r"\b(ELAB|BRCH|BACK|SYNT|CRIT|SUPP|CONT|NONE)\b", text)
        if type_match:
            relation = type_match.group(1)
            conf_match = re.search(r"0\.[0-9]+", text)
            confidence = float(conf_match.group(0)) if conf_match else 0.7
            return {"relation": relation, "confidence": confidence}

        return None

    def _parse_verification_response(self, response: str) -> Optional[dict]:
        """Parse synthesis verification response."""
        import re

        text = response.strip()

        if "true" in text.lower() or "yes" in text.lower():
            conf_match = re.search(r"0\.[0-9]+", text)
            confidence = float(conf_match.group(0)) if conf_match else 0.7
            return {"draws_from": True, "confidence": confidence}
        elif "false" in text.lower() or "no" in text.lower():
            return {"draws_from": False, "confidence": 0.5}

        return None

    def _parse_backtrack_response(self, response: str) -> Optional[dict]:
        """Parse backtrack target response."""
        import re

        text = response.strip()

        # Try to find target_id
        id_match = re.search(r"target.*?(\d+)", text, re.IGNORECASE)
        if id_match:
            conf_match = re.search(r"0\.[0-9]+", text)
            confidence = float(conf_match.group(0)) if conf_match else 0.7
            return {"target_id": int(id_match.group(1)), "confidence": confidence}

        return None


EDGE_PATTERNS = {
    EdgeType.ELAB: [
        r"\bspecifically\b",
        r"\bfor example\b",
        r"\bin practice\b",
        r"\bto elaborate\b",
        r"\bin fact\b",
        r"\bthat is\b",
        r"\bi.e\.\b",
    ],
    EdgeType.BRCH: [
        r"\banother\b",
        r"\balternative\b",
        r"\bwhat if\b",
        r"\bon the other hand\b",
        r"\bhowever\b",
        r"\bbut\b",
        r"\balso\b(?!\s+(?:be|have|is))",
    ],
    EdgeType.BACK: [
        r"\bwait\b",
        r"\bactually\b",
        r"\blet me reconsider\b",
        r"\bi was wrong\b",
        r"\bthat\'s not right\b",
        r"\bi realize\b",
        r"\bhowever\b",
    ],
    EdgeType.CRIT: [
        r"\bthe problem\b",
        r"\bfails\b",
        r"\bdoesn\'t work\b",
        r"\bweakness\b",
        r"\bflaw\b",
        r"\bissue\b",
        r"\bshortcoming\b",
        r"\bbut\b",
    ],
    EdgeType.SUPP: [
        r"\bbecause\b",
        r"\bsince\b",
        r"\bsupported by\b",
        r"\bevidence\b",
        r"\bproves\b",
        r"\bdemonstrates\b",
        r"\bshows that\b",
    ],
    EdgeType.CONT: [
        r"\bhowever\b",
        r"\bin contrast\b",
        r"\bon the contrary\b",
        r"\bwhereas\b",
        r"\bwhile\b",
        r"\bbut\b",
    ],
}


def build_graph(
    trace: dict,
    thought_units: list[ThoughtUnit],
    edges: list[Edge],
) -> ThoughtGraph:
    """Build complete ThoughtGraph from components."""
    return ThoughtGraph(
        trace_id=trace["trace_id"],
        model=trace["model"],
        question_id=trace["question_id"],
        domain=trace["domain"],
        run=trace["run"],
        nodes=thought_units,
        edges=edges,
        token_count=trace.get("token_count", len(trace.get("raw_cot", "").split())),
    )


def _classify_edge_rules(text_a: str, text_b: str) -> Optional[dict]:
    """Rule-based edge classification."""
    import re

    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()

    # Check for specific patterns in text_b (the later thought)
    for edge_type, patterns in EDGE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_b_lower):
                return {"type": edge_type, "confidence": 0.6}

    return None
