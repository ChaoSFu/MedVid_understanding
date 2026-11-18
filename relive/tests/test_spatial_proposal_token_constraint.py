from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from relive.backends.base import Backend
from relive.spatial_reliability.proposal_constraint import (
    PROPOSAL_GRAMMAR_VERSION,
    ProposalJsonGrammar,
    TokenLevelProposalConstraint,
    proposal_constraint_identity,
)
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference


PROPOSED = (
    '{"status":"PROPOSED","bbox_normalized_0_1000":[100,200,700,800],'
    '"reason":"VISIBLE_EVIDENCE_REGION"}'
)
UNRESOLVED = (
    '{"status":"UNRESOLVED","bbox_normalized_0_1000":null,'
    '"reason":"NO_CONCRETE_LOCAL_REGION"}'
)


class _Tokenizer:
    pieces = [
        "{", "}", "[", "]", ":", ",", '"', "status", "PROPOSED", "UNRESOLVED",
        "bbox_normalized_0_1000", "reason", "VISIBLE_EVIDENCE_REGION",
        "NO_CONCRETE_LOCAL_REGION", "null", "100", "200", "700", "800", "`",
    ]
    vocab_size = len(pieces) + 1

    def decode(self, ids, **_):
        return "".join("" if token == self.vocab_size - 1 else self.pieces[token] for token in ids)


class _ConstrainedBackend(Backend):
    synthetic = False

    def __init__(self):
        super().__init__({"kind": "unit", "model": "constraint-stub", "revision": "v1", "max_retries": 0})

    def infer(self, request):
        raise AssertionError("unconstrained inference must not be called")

    def infer_with_token_constraint(self, request, constraint):
        self.calls += 1
        self.asserted_identity = request["token_constraint_identity"]
        return {
            "raw_response": PROPOSED,
            "generation_metadata": {
                "finish_reason": "EOS_TOKEN", "generated_token_count": 20,
                "max_new_tokens": 256, "reached_max_new_tokens": False,
            },
            "constraint_metadata": {
                "binding": {"grammar_spec_sha256": constraint.grammar.spec_sha256},
                "tokenizer_binding": {"tokenizer_class": "fixture"},
                "execution": {
                    "grammar_version": constraint.grammar.grammar_version,
                    "constraint_failure": None, "final_prefix_status": "COMPLETE",
                },
            },
        }


class SpatialProposalTokenConstraintTests(unittest.TestCase):
    def test_closed_grammar_accepts_only_canonical_surface(self):
        grammar = ProposalJsonGrammar()
        self.assertEqual(grammar.prefix_state(PROPOSED).status, "COMPLETE")
        self.assertEqual(grammar.prefix_state(UNRESOLVED).status, "COMPLETE")
        invalid = (
            "```json\n" + PROPOSED + "\n```",
            PROPOSED + "`",
            PROPOSED.replace("bbox_normalized_0_1000", "bbox_normalized_0_0_1000"),
            PROPOSED.replace("VISIBLE_EVIDENCE_REGION", "free form reason"),
            UNRESOLVED.replace("null", "[null,null,null,null]"),
            UNRESOLVED.replace("UNRESOLVED", "SUPPORTED"),
            '{"status":"PROPOSED","bbox_normalized_0_1000":[010,200,700,800],"reason":"VISIBLE_EVIDENCE_REGION"}',
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                self.assertEqual(grammar.prefix_state(raw).status, "INVALID")

    def test_actual_token_prefix_filter_handles_multichar_tokens_and_eos(self):
        grammar = ProposalJsonGrammar()
        constraint = TokenLevelProposalConstraint(grammar)
        tokenizer = _Tokenizer()
        eos = tokenizer.vocab_size - 1
        constraint.prepare(tokenizer, 0, {eos})
        sequence = [
            "{", '"', "status", '"', ":", '"', "PROPOSED", '"', ",", '"',
            "bbox_normalized_0_1000", '"', ":", "[", "100", ",", "200", ",",
            "700", ",", "800", "]", ",", '"', "reason", '"', ":", '"',
            "VISIBLE_EVIDENCE_REGION", '"', "}",
        ]
        ids = []
        for piece in sequence:
            ids.append(tokenizer.pieces.index(piece))
            self.assertNotEqual(grammar.prefix_state(tokenizer.decode(ids)).status, "INVALID")
        self.assertEqual(grammar.prefix_state(tokenizer.decode(ids)).status, "COMPLETE")
        self.assertEqual(constraint.allowed_token_ids(ids), [eos])
        self.assertNotIn(tokenizer.pieces.index("`"), constraint.allowed_token_ids(ids[:-1]))
        metadata = constraint.final_metadata(ids + [eos])
        self.assertEqual(metadata["grammar_version"], PROPOSAL_GRAMMAR_VERSION)
        self.assertIsNone(metadata["constraint_failure"])

    def test_premature_eos_and_no_legal_successor_fail_closed(self):
        tokenizer = _Tokenizer()
        eos = tokenizer.vocab_size - 1
        early = TokenLevelProposalConstraint()
        early.prepare(tokenizer, 0, {eos})
        self.assertEqual(early.final_metadata([eos])["constraint_failure"], "PREMATURE_EOS")

        class _BlockedTokenizer:
            vocab_size = 2
            def decode(self, ids, **_):
                return "".join("" if token == 1 else "`" for token in ids)

        blocked = TokenLevelProposalConstraint()
        blocked.prepare(_BlockedTokenizer(), 0, {1})
        self.assertEqual(blocked.allowed_token_ids([]), [1])
        self.assertEqual(blocked.final_metadata([1])["constraint_failure"], "NO_LEGAL_SUCCESSOR_TOKEN")

    def test_constrained_cache_is_version_bound_and_exactly_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            Image.new("RGB", (8, 8), "blue").save(image)
            backend = _ConstrainedBackend()
            cache = CachedInference(backend, ArtifactStore(root / "cache"))
            request = {
                "stage": "relive_spatial_proposal", "prompt": "fixed",
                "prompt_version": "proposal-v3", "image_paths": [str(image)],
                "frame_ids": ["frame-1"], "token_constraint_identity": proposal_constraint_identity(),
            }
            first = cache.call_with_token_constraint(request, Budget(2), TokenLevelProposalConstraint)
            replay_budget = Budget(2)
            replay = cache.call_with_token_constraint(request, replay_budget, TokenLevelProposalConstraint)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(replay["cache_hit"])
            self.assertEqual(replay_budget.new_calls, 0)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(replay["raw_text"], PROPOSED)
            self.assertEqual(replay["constraint_metadata"]["execution"]["final_prefix_status"], "COMPLETE")
            raw_record = json.loads(Path(replay["raw_response_ref"]).read_text(encoding="utf-8"))
            self.assertEqual(raw_record["identity"]["request"]["token_constraint_identity"],
                             proposal_constraint_identity())


if __name__ == "__main__":
    unittest.main()
