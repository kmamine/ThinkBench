"""Thought graph extraction: Classifier - Node type classification (rule-based fallback)."""

import re
import logging
from pathlib import Path
from typing import Optional

from ..collect.models import LLMClient
from .schemas import ThoughtUnit, NodeType, NodeFamily, NODE_FAMILY_MAP

logger = logging.getLogger(__name__)

CLASSIFY_PATTERNS = {
    NodeType.HYP: [
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bi think\b",
        r"\bi believe\b",
        r"\bone approach\b",
        r"\bcould be\b",
        r"\bwhat if\b",
        r"\bmight be\b",
        r"\bi suggest\b",
        r"\bpropose\b",
        r"\bconsider\b(?!\s+this\b)",
        r"\bpossible\b",
        r"\bpotentially\b",
        r"\bone possibility\b",
        r"\bwe could\b(?!\s+consider)",
        r"\bital\b",
    ],
    NodeType.RFR: [
        r"\blooking at\b",
        r"\bfrom .* perspective\b",
        r"\bdifferently\b",
        r"\breframe\b",
        r"\banother way\b",
        r"\bputting it\b",
        r"\bviewing\b",
        r"\bconsidering from\b",
        r"\blet's view\b",
        r"\bas .* see it\b",
    ],
    NodeType.ANA: [
        r"\bsimilar to\b",
        r"\blike\b(?=.*\bcase\b|\bexample\b)",
        r"\banalogous\b",
        r"\bcompare\b",
        r"\bjust like\b",
        r"\bin the same way\b",
        r"\bby comparison\b",
        r"\bparallel\b",
    ],
    NodeType.BRS: [
        r"\boptions include\b",
        r"\bwe could\b",
        r"\balternatives\b",
        r"\bpossibilities\b",
        r"\bseveral ways\b",
        r"\blist:\b",
        r"\bfirst,\s+second\b",
        r"\bfor one thing\b",
    ],
    NodeType.JUS: [
        r"\bbecause\b",
        r"\bsupported by\b",
        r"\bthe reason\b",
        r"\bevidenced\b",
        r"\bsince\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bconsequently\b",
        r"\bthis is due to\b",
        r"\bowing to\b",
        r"\bfollows that\b",
    ],
    NodeType.SPC: [
        r"\bspecifically\b",
        r"\bfor example\b",
        r"\bin practice\b",
        r"\bin particular\b",
        r"\bfor instance\b",
        r"\bnamely\b",
        r"\bthat is\b",
        r"\bi.e.\b",
        r"\be.g.\b",
    ],
    NodeType.IMP: [
        r"\bthis means\b",
        r"\bimplies\b",
        r"\bas a result\b",
        r"\bleading to\b",
        r"\bresults in\b",
        r"\bultimately\b",
        r"\bthis leads\b",
        r"\bwhich leads to\b",
        r"\bresulting in\b",
    ],
    NodeType.CON: [
        r"\bhowever\b",
        r"\bbut\b",
        r"\bonly works if\b",
        r"\blimitation\b",
        r"\bprovided that\b",
        r"\bassuming\b",
        r"\bexcept\b",
        r"\bunless\b",
        r"\brestriction\b",
        r"\bconstraint\b",
        r"\bcondition\b",
        r"\bwith the caveat\b",
    ],
    NodeType.CRT: [
        r"\bthe problem\b",
        r"\bfails because\b",
        r"\bweakness\b",
        r"\bcritique\b",
        r"\bflaw\b",
        r"\bissue with\b",
        r"\bshortcoming\b",
        r"\bdownside\b",
        r"\bdoesn't work\b",
        r"\bwon't\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bchallenge\b",
        r"\bconcern\b",
        r"\bdrawback\b",
    ],
    NodeType.CMP: [
        r"\bcompared to\b",
        r"\bwhereas\b",
        r"\bwhile\b",
        r"\bin contrast\b",
        r"\btradeoff\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\balternatively\b",
        r"\bon the other hand\b",
    ],
    NodeType.MET: [
        r"\bi'm going in circles\b",
        r"\blet me step back\b",
        r"\breconsidering\b",
        r"\bwait\b",
        r"\bactually\b",
        r"\bi realize\b",
        r"\blet me think\b",
        r"\bi need to rethink\b",
        r"\blet's pause\b",
        r"\bneed to reflect\b",
        r"\bmeta\b",
    ],
    NodeType.SYN: [
        r"\bcombining\b",
        r"\bbringing together\b",
        r"\bin conclusion\b",
        r"\bultimately\b",
        r"\btherefore\b",
        r"\bto sum up\b",
        r"\bsynthesizing\b",
        r"\boverall\b",
        r"\ball things considered\b",
        r"\bbased on\b",
        r"\bin summary\b",
        r"\bto summarize\b",
        r"\bkey takeaways\b",
    ],
}


class Classifier:
    def __init__(self, client: LLMClient, confidence_threshold: float = 0.7):
        self.client = client
        self.confidence_threshold = confidence_threshold

    async def classify(self, thought_units: list[ThoughtUnit]) -> list[ThoughtUnit]:
        """Classify all thought units with context."""
        if not thought_units:
            return []

        classified = []
        for i, tu in enumerate(thought_units):
            # Try LLM first
            result = await self._classify_llm(tu, thought_units[:i])

            if not result:
                # Fall back to rule-based
                result = self._classify_rules(tu.text)

            if result:
                try:
                    node_type = NodeType(result["type"])
                except ValueError:
                    node_type = NodeType.HYP

                tu.node_type = node_type
                tu.node_family = NODE_FAMILY_MAP.get(node_type)
                tu.classification_confidence = result.get("confidence", 0.5)

            classified.append(tu)

        logger.info(f"Classified {len(classified)} thought units")
        return classified

    async def _classify_llm(
        self, tu: ThoughtUnit, context: list[ThoughtUnit]
    ) -> Optional[dict]:
        """Try LLM classification."""
        try:
            context_text = ""
            if context:
                context_text = context[-1].text[:100]

            prompt = f'''Classify: "{tu.text[:150]}"
Types: HYP, RFR, ANA, BRS, JUS, SPC, IMP, CON, CRT, CMP, MET, SYN
Output:'''

            response = await self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
            )

            return self._parse_response(response)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None

    def _parse_response(self, response: str) -> Optional[dict]:
        """Parse classification response."""
        text = response.strip()

        type_match = re.search(
            r"\b(HYP|RFR|ANA|BRS|JUS|SPC|IMP|CON|CRT|CMP|MET|SYN)\b", text
        )
        if type_match:
            return {"type": type_match.group(1), "confidence": 0.7}

        return None

    def _classify_rules(self, text: str) -> Optional[dict]:
        """Rule-based classification as fallback."""
        text_lower = text.lower()

        scores = {}
        for node_type, patterns in CLASSIFY_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            if score > 0:
                scores[node_type] = score

        if not scores:
            return {"type": "HYP", "confidence": 0.3}

        best_type = max(scores, key=scores.get)
        confidence = min(0.9, 0.4 + scores[best_type] * 0.1)

        return {"type": best_type.value, "confidence": confidence}
