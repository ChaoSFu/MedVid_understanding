"""Fail-closed token-level compact JSON grammar for Stage 3G-A v3.2.

The grammar has one deliberately narrow canonical surface form.  It constrains
actual ``generate`` logits through :class:`TokenLevelGroundingConstraint`; the
separate Stage 3G validator still validates semantic role completeness and box
geometry after decoding.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Iterable

GRAMMAR_VERSION = "relive-v2-grounding-token-json-grammar-v3.2"
IMPLEMENTATION_VERSION = "relive-v2-token-prefix-constraint-v1"
VISIBILITIES = ("VISIBLE", "NOT_VISIBLE", "AMBIGUOUS")
_TOKEN_PIECE_CACHE: dict[tuple[int, int, tuple[int, ...]], dict[str, list[int]]] = {}


class TokenConstraintError(ValueError):
    pass


@dataclass(frozen=True)
class PrefixState:
    status: str  # PREFIX, COMPLETE, INVALID
    reason: str | None = None


class GroundingJsonGrammar:
    """Recognize prefixes of the one permitted compact JSON object.

    Role order is frozen per task.  Coordinates are syntactically limited to
    integers in [0, 1000]; xy ordering remains an independent post-generation
    validator requirement.
    """

    def __init__(self, roles: Iterable[str]):
        self.roles = tuple(roles)
        if not self.roles or len(self.roles) != len(set(self.roles)):
            raise TokenConstraintError("GRAMMAR_ROLE_CONTRACT_INVALID")
        if any(not isinstance(role, str) or not role or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_" for ch in role)
               for role in self.roles):
            raise TokenConstraintError("GRAMMAR_ROLE_CONTRACT_INVALID")
        self.spec = {
            "grammar_version": GRAMMAR_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "surface": "compact_json_only",
            "top_level": {"components": "fixed_role_order_array"},
            "roles": list(self.roles),
            "visibility": list(VISIBILITIES),
            "visible_bbox": "four_integers_0_to_1000",
            "nonvisible_bbox": "json_null_only",
            "allow_trailing_text": False,
        }
        self.spec_sha256 = hashlib.sha256(
            repr(sorted(self.spec.items())).encode("utf-8")
        ).hexdigest()

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
        if len(digits) > 4 or int(digits) > 1000:
            return None
        # At end the number may still receive a digit, comma, or bracket.
        if pos == len(text):
            return pos, True
        return pos, False

    def prefix_state(self, text: str) -> PrefixState:
        if not isinstance(text, str) or any(ord(ch) > 127 for ch in text):
            return PrefixState("INVALID", "NON_ASCII_OR_NOT_STRING")
        pos = 0
        result = self._literal(text, pos, '{"components":[')
        if result is None:
            return PrefixState("INVALID", "TOP_LEVEL_OBJECT_REQUIRED")
        pos, need = result
        if need:
            return PrefixState("PREFIX")
        for role_index, role in enumerate(self.roles):
            result = self._literal(text, pos, '{"role":"' + role + '","visibility":"')
            if result is None:
                return PrefixState("INVALID", "ROLE_OR_COMPONENT_STRUCTURE_INVALID")
            pos, need = result
            if need:
                return PrefixState("PREFIX")
            suffix = text[pos:]
            matches = [v for v in VISIBILITIES if v.startswith(suffix)]
            if matches:
                return PrefixState("PREFIX")
            visibility = next((v for v in VISIBILITIES if text.startswith(v, pos)), None)
            if visibility is None:
                return PrefixState("INVALID", "VISIBILITY_INVALID")
            pos += len(visibility)
            result = self._literal(text, pos, '","bbox_2d":')
            if result is None:
                return PrefixState("INVALID", "BBOX_FIELD_INVALID")
            pos, need = result
            if need:
                return PrefixState("PREFIX")
            if visibility == "VISIBLE":
                result = self._literal(text, pos, '[')
                if result is None:
                    return PrefixState("INVALID", "VISIBLE_BBOX_REQUIRED")
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
                    result = self._literal(text, pos, ',' if axis < 3 else ']')
                    if result is None:
                        return PrefixState("INVALID", "VISIBLE_BBOX_DELIMITER_INVALID")
                    pos, need = result
                    if need:
                        return PrefixState("PREFIX")
            else:
                result = self._literal(text, pos, 'null')
                if result is None:
                    return PrefixState("INVALID", "NONVISIBLE_BBOX_MUST_BE_NULL")
                pos, need = result
                if need:
                    return PrefixState("PREFIX")
            result = self._literal(text, pos, '}')
            if result is None:
                return PrefixState("INVALID", "COMPONENT_OBJECT_INVALID")
            pos, need = result
            if need:
                return PrefixState("PREFIX")
            result = self._literal(text, pos, ',' if role_index + 1 < len(self.roles) else ']}')
            if result is None:
                return PrefixState("INVALID", "COMPONENT_SEQUENCE_INVALID")
            pos, need = result
            if need:
                return PrefixState("PREFIX")
        return PrefixState("COMPLETE" if pos == len(text) else "INVALID",
                           None if pos == len(text) else "TRAILING_TEXT_FORBIDDEN")

    def next_characters(self, text: str) -> set[str]:
        if self.prefix_state(text).status != "PREFIX":
            return set()
        alphabet = '{}[]":,0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz'
        return {ch for ch in alphabet if self.prefix_state(text + ch).status != "INVALID"}


class TokenLevelGroundingConstraint:
    """A logits processor adapter whose allowed ids are tested against grammar.

    It has no repair branch.  Before completion EOS is masked.  If decoder
    tokenization makes all permitted candidates unavailable, a forced EOS is
    emitted only to stop generation and the resulting record receives an
    explicit constraint failure; the incomplete text remains invalid.
    """

    def __init__(self, grammar: GroundingJsonGrammar):
        self.grammar = grammar
        self.tokenizer: Any | None = None
        self.prompt_token_count: int | None = None
        self.eos_token_ids: set[int] = set()
        self._pieces_by_initial: dict[str, list[int]] = {}
        self._prepared_at: float | None = None
        self._callback_count = 0
        self._candidate_evaluations = 0
        self._constraint_failure: str | None = None
        self._failure_step: int | None = None

    def prepare(self, tokenizer: Any, prompt_token_count: int, eos_token_ids: set[int]) -> dict[str, Any]:
        if not isinstance(prompt_token_count, int) or prompt_token_count < 0 or not eos_token_ids:
            raise TokenConstraintError("CONSTRAINT_BINDING_INVALID")
        decode = getattr(tokenizer, "decode", None)
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not callable(decode) or not isinstance(vocab_size, int) or vocab_size <= 0:
            raise TokenConstraintError("CONSTRAINT_TOKENIZER_UNSUPPORTED")
        started = time.perf_counter()
        cache_key = (id(tokenizer), vocab_size, tuple(sorted(eos_token_ids)))
        mapping = _TOKEN_PIECE_CACHE.get(cache_key)
        if mapping is None:
            mapping = {}
            for token_id in range(vocab_size):
                if token_id in eos_token_ids:
                    continue
                try:
                    piece = decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except Exception as exc:
                    raise TokenConstraintError("CONSTRAINT_TOKEN_DECODING_UNAVAILABLE") from exc
                if isinstance(piece, str) and piece and piece.isascii():
                    mapping.setdefault(piece[0], []).append(token_id)
            _TOKEN_PIECE_CACHE[cache_key] = mapping
        self.tokenizer = tokenizer
        self.prompt_token_count = prompt_token_count
        self.eos_token_ids = set(eos_token_ids)
        self._pieces_by_initial = mapping
        self._prepared_at = time.perf_counter() - started
        return self.metadata()

    def _decode(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            raise TokenConstraintError("CONSTRAINT_NOT_PREPARED")
        value = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not isinstance(value, str):
            raise TokenConstraintError("CONSTRAINT_TOKEN_DECODING_UNAVAILABLE")
        return value

    def allowed_token_ids(self, full_input_ids: list[int]) -> list[int]:
        if self.prompt_token_count is None:
            raise TokenConstraintError("CONSTRAINT_NOT_PREPARED")
        generated = full_input_ids[self.prompt_token_count:]
        non_eos = [token for token in generated if token not in self.eos_token_ids]
        if len(non_eos) != len(generated):
            return list(self.eos_token_ids)
        text = self._decode(non_eos)
        state = self.grammar.prefix_state(text)
        self._callback_count += 1
        if state.status == "COMPLETE":
            return sorted(self.eos_token_ids)
        if state.status != "PREFIX":
            self._constraint_failure = "GENERATED_TOKEN_PREFIX_INVALID"
            self._failure_step = len(generated)
            return sorted(self.eos_token_ids)
        candidates: list[int] = []
        for first in self.grammar.next_characters(text):
            for token_id in self._pieces_by_initial.get(first, []):
                candidate_text = self._decode(non_eos + [token_id])
                self._candidate_evaluations += 1
                if self.grammar.prefix_state(candidate_text).status != "INVALID":
                    candidates.append(token_id)
        if not candidates:
            self._constraint_failure = "NO_LEGAL_SUCCESSOR_TOKEN"
            self._failure_step = len(generated)
            return sorted(self.eos_token_ids)
        return sorted(set(candidates))

    def apply_logits(self, input_ids: Any, scores: Any) -> Any:
        """Mask actual generation logits; only grammar-prefix tokens survive."""
        if getattr(input_ids, "shape", None) is None or int(input_ids.shape[0]) != 1:
            raise TokenConstraintError("CONSTRAINT_BATCH_UNSUPPORTED")
        ids = input_ids[0].detach().to("cpu").tolist() if hasattr(input_ids[0], "detach") else list(input_ids[0])
        allowed = self.allowed_token_ids(ids)
        if not allowed:
            raise TokenConstraintError("NO_LEGAL_SUCCESSOR_TOKEN")
        masked = scores.new_full(scores.shape, float("-inf"))
        masked[0, allowed] = scores[0, allowed]
        return masked

    def metadata(self) -> dict[str, Any]:
        return {
            "grammar_version": GRAMMAR_VERSION,
            "grammar_spec_sha256": self.grammar.spec_sha256,
            "implementation_version": IMPLEMENTATION_VERSION,
            "tokenizer_vocab_size": sum(len(v) for v in self._pieces_by_initial.values()),
            "initialization_overhead_seconds": self._prepared_at,
            "callback_count": self._callback_count,
            "candidate_evaluations": self._candidate_evaluations,
            "constraint_failure": self._constraint_failure,
            "constraint_failure_step": self._failure_step,
        }

    def final_metadata(self, generated_token_ids: list[int]) -> dict[str, Any]:
        non_eos = [token for token in generated_token_ids if token not in self.eos_token_ids]
        text = self._decode(non_eos)
        state = self.grammar.prefix_state(text)
        data = self.metadata()
        data["final_prefix_status"] = state.status
        data["final_prefix_reason"] = state.reason
        data["generated_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if generated_token_ids and generated_token_ids[-1] in self.eos_token_ids and state.status != "COMPLETE" and data["constraint_failure"] is None:
            data["constraint_failure"] = "PREMATURE_EOS"
            data["constraint_failure_step"] = len(generated_token_ids)
        if state.status != "COMPLETE" and data["constraint_failure"] is None:
            data["constraint_failure"] = "INCOMPLETE_OR_INVALID_AT_STOP"
            data["constraint_failure_step"] = len(generated_token_ids)
        return data
