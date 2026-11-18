"""Token-level JSON contract for ReliVE-Spatial localization proposals.

The grammar constrains actual decoder tokens.  It deliberately has no repair,
alias, Markdown-fence, or trailing-text path.  Geometry is still checked by
the independent proposal validator after decoding.
"""
from __future__ import annotations

import hashlib
import json

from relive.v2.token_json_constraint import PrefixState, TokenLevelGroundingConstraint


PROPOSAL_GRAMMAR_VERSION = "relive-spatial-proposal-token-json-grammar-v3"
PROPOSAL_CONSTRAINT_IMPLEMENTATION_VERSION = "relive-v2-token-prefix-constraint-v1"
PROPOSAL_REASON = "VISIBLE_EVIDENCE_REGION"
UNRESOLVED_REASON = "NO_CONCRETE_LOCAL_REGION"


class ProposalJsonGrammar:
    """Recognize prefixes of the two canonical proposal JSON objects."""

    grammar_version = PROPOSAL_GRAMMAR_VERSION
    implementation_version = PROPOSAL_CONSTRAINT_IMPLEMENTATION_VERSION

    def __init__(self) -> None:
        self.spec = {
            "grammar_version": self.grammar_version,
            "implementation_version": self.implementation_version,
            "surface": "compact_json_only",
            "top_level": "object",
            "field_order": ["status", "bbox_normalized_0_1000", "reason"],
            "branches": {
                "PROPOSED": {
                    "bbox_normalized_0_1000": "four_integers_0_to_1000",
                    "reason": PROPOSAL_REASON,
                },
                "UNRESOLVED": {
                    "bbox_normalized_0_1000": None,
                    "reason": UNRESOLVED_REASON,
                },
            },
            "allow_markdown_fence": False,
            "allow_trailing_text": False,
            "allow_field_aliases": False,
        }
        encoded = json.dumps(self.spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.spec_sha256 = hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _literal(text: str, pos: int, value: str) -> tuple[int, bool] | None:
        remain = text[pos:]
        if len(remain) < len(value):
            return (pos, True) if value.startswith(remain) else None
        if text.startswith(value, pos):
            return pos + len(value), False
        return None

    @staticmethod
    def _number(text: str, pos: int) -> tuple[int, bool] | None:
        start = pos
        while pos < len(text) and text[pos].isdigit():
            pos += 1
        digits = text[start:pos]
        if not digits:
            return (pos, True) if pos == len(text) else None
        if (len(digits) > 1 and digits.startswith("0")) or len(digits) > 4 or int(digits) > 1000:
            return None
        if pos == len(text):
            return pos, True
        return pos, False

    def _proposed_suffix(self, text: str, pos: int) -> PrefixState:
        result = self._literal(text, pos, '","bbox_normalized_0_1000":[')
        if result is None:
            return PrefixState("INVALID", "PROPOSED_BBOX_FIELD_INVALID")
        pos, need = result
        if need:
            return PrefixState("PREFIX")
        for axis in range(4):
            result = self._number(text, pos)
            if result is None:
                return PrefixState("INVALID", "COORDINATE_OUT_OF_RANGE")
            pos, need = result
            if need:
                return PrefixState("PREFIX")
            result = self._literal(text, pos, "," if axis < 3 else "]")
            if result is None:
                return PrefixState("INVALID", "PROPOSED_BBOX_DELIMITER_INVALID")
            pos, need = result
            if need:
                return PrefixState("PREFIX")
        result = self._literal(text, pos, ',"reason":"' + PROPOSAL_REASON + '"}')
        if result is None:
            return PrefixState("INVALID", "PROPOSED_REASON_INVALID")
        pos, need = result
        if need:
            return PrefixState("PREFIX")
        return PrefixState("COMPLETE" if pos == len(text) else "INVALID",
                           None if pos == len(text) else "TRAILING_TEXT_FORBIDDEN")

    def _unresolved_suffix(self, text: str, pos: int) -> PrefixState:
        result = self._literal(
            text, pos,
            '","bbox_normalized_0_1000":null,"reason":"' + UNRESOLVED_REASON + '"}',
        )
        if result is None:
            return PrefixState("INVALID", "UNRESOLVED_CONTRACT_INVALID")
        pos, need = result
        if need:
            return PrefixState("PREFIX")
        return PrefixState("COMPLETE" if pos == len(text) else "INVALID",
                           None if pos == len(text) else "TRAILING_TEXT_FORBIDDEN")

    def prefix_state(self, text: str) -> PrefixState:
        if not isinstance(text, str) or any(ord(ch) > 127 for ch in text):
            return PrefixState("INVALID", "NON_ASCII_OR_NOT_STRING")
        prefix = '{"status":"'
        result = self._literal(text, 0, prefix)
        if result is None:
            return PrefixState("INVALID", "TOP_LEVEL_OBJECT_REQUIRED")
        pos, need = result
        if need:
            return PrefixState("PREFIX")
        remain = text[pos:]
        branches = ("PROPOSED", "UNRESOLVED")
        if any(branch.startswith(remain) for branch in branches):
            return PrefixState("PREFIX")
        if text.startswith("PROPOSED", pos):
            return self._proposed_suffix(text, pos + len("PROPOSED"))
        if text.startswith("UNRESOLVED", pos):
            return self._unresolved_suffix(text, pos + len("UNRESOLVED"))
        return PrefixState("INVALID", "STATUS_INVALID")

    def next_characters(self, text: str) -> set[str]:
        if self.prefix_state(text).status != "PREFIX":
            return set()
        alphabet = '{}[]":,0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz'
        return {character for character in alphabet
                if self.prefix_state(text + character).status != "INVALID"}


class TokenLevelProposalConstraint(TokenLevelGroundingConstraint):
    """Named adapter for proposal contracts; implementation is shared and audited."""

    def __init__(self, grammar: ProposalJsonGrammar | None = None):
        super().__init__(grammar or ProposalJsonGrammar())


def proposal_constraint_identity() -> dict[str, object]:
    grammar = ProposalJsonGrammar()
    return {
        "enforcement": "ACTUAL_GENERATE_LOGITS_MASK_WITH_FAIL_CLOSED_PREFIX_GRAMMAR",
        "grammar_version": grammar.grammar_version,
        "implementation_version": grammar.implementation_version,
        "grammar_spec_sha256": grammar.spec_sha256,
        "grammar_spec": grammar.spec,
    }
