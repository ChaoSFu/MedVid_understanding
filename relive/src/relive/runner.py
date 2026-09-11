"""Auditable, bounded test-time verification on frozen model services."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid

from PIL import Image, __version__ as pillow_version

from relive.acquisition import acquire, expand_candidate
from relive.adaptation import action_status, choose_action, event
from relive.audits import audit_runtime_imports, audit_strict_result
from relive.backends import make_backend
from relive.certificate import build_certificate, POLICIES, POLICY_VERSION
from relive.claims import PROMPT_VERSIONS, make_contrast, parse_claims, parse_contrasts, prompt, stable_id
from relive.config import validate_config
from relive.contrasts import evaluate_contrast
from relive.coverage import assess_coverage
from relive.data.frames import select_frames, timing_summary
from relive.data.schemas import load_runtime, SUPPORTED_TASKS
from relive.interventions import (apply_spatial_intervention, check_spatial, generate_control_regions,
                                  INTERVENTION_VERSION, CONTROL_VERSION, INTERVENTION_PROTOCOL_VERSION)
from relive.reasoning import forced_answer, strict_answer
from relive.spatial import parse_proposal
from relive.storage.artifacts import ArtifactError, ArtifactStore, stable_hash
from relive.storage.cache import Budget, BudgetExceeded, CachedInference
from relive.types import Claim, ExecutionStatus, ExclusivityStatus, FinalStatus, SemanticStatus, VerificationResult, to_dict
from relive.verification import parse_verification


def _image_file(root: Path, image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    data = stream.getvalue()
    name = hashlib.sha256(data).hexdigest()
    target = root / "images" / f"{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("Image content identity collision")
        return str(target)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, target)
        except FileExistsError:
            if target.read_bytes() != data:
                raise ValueError("Image content identity collision")
    finally:
        os.unlink(tmp)
    return str(target)


class SampleRunner:
    def __init__(self, config, store, inference):
        self.cfg, self.store, self.inference = config, store, inference
        self.intervention = config["spatial"]["intervention"]
        self.budget = Budget(config["budget"]["max_calls"])
        self.events, self.certificates, self.claims, self.candidates, self.proposals = [], [], [], [], []
        self.seen_candidates = set()
        self.verification_memo = {}
        self.contrast_memo = {}
        self.rounds = 0

    def record(self, stage, payload):
        return self.store.append_event(stage, to_dict(payload))

    def infer(self, sample, candidate, claim, stage, variant="ORIGINAL", image_paths=None,
              region=None, proposal_index=0, control_index=None):
        frames = select_frames(sample, candidate)
        frame_ids = list(candidate.frame_ids)
        actual_paths = image_paths or [f.path for f in frames]
        memo_key = None
        if stage == "semantic":
            memo_key = (candidate.candidate_id, claim.claim_id if claim else None, variant,
                        tuple(actual_paths), tuple(region) if region is not None else None, control_index,
                        json.dumps(self.intervention, sort_keys=True, separators=(",", ":")))
            prior = self.verification_memo.get(memo_key)
            if prior is not None:
                self.record("verification_reuse", {"sample_id": sample.sample_id,
                                                   "candidate_id": candidate.candidate_id,
                                                   "claim_id": claim.claim_id if claim else None,
                                                   "variant": variant,
                                                   "raw_response_ref": prior.raw_response_ref,
                                                   "reason": "SAME_INPUT_REUSED_WITHOUT_NEW_INFERENCE"})
                return prior
        # Semantic verdicts bind SUPPORT/CONTRADICTION to one supplied frame.
        # v4 exposes the chronological zero-based index instead of long runtime
        # IDs; the strict parser resolves that index back to an immutable supplied
        # frame ID before certification.
        data = ({"frame_count": len(frame_ids)} if stage == "semantic"
                else {"frame_ids": frame_ids, "timing": timing_summary(frames)})
        if stage != "semantic" and sample.metadata.get("query_timestamps"):
            # These are an explicitly whitelisted public task query, not a hidden
            # annotated action span. Their timebase is never inferred here.
            data["public_query_timestamps"] = list(sample.metadata["query_timestamps"])
        if stage in {"claims"}:
            data.update(question=sample.question, max_claims=self.cfg["claims"]["max_claims"])
        elif claim is not None:
            data["claim"] = {"text": claim.text, "entity": claim.entity}
            if stage != "semantic":
                data["claim"]["time_scope"] = claim.time_scope
        if stage == "contrasts":
            data["max_alternatives"] = self.cfg["claims"]["max_alternatives"]
        context = {"sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
                   "claim_id": claim.claim_id if claim else None, "claim_text": claim.text if claim else None,
                   "variant": variant, "round_index": candidate.round_index,
                   "proposal_index": proposal_index, "region": list(region) if region is not None else None,
                   "control_index": control_index,
                   "intervention_version": self.intervention["operator_version"],
                   "intervention_protocol": self.intervention, "control_version": CONTROL_VERSION,
                   "preprocessing": {"decode": "Pillow_RGB", "resize": False, "pillow": pillow_version}}
        request = {"stage": stage, "prompt": prompt(stage, data), "prompt_version": PROMPT_VERSIONS[stage],
                   "image_paths": actual_paths, "frame_ids": frame_ids,
                   "context": context}
        try:
            result = self.inference.call(request, self.budget)
        except BudgetExceeded:
            raise
        except ArtifactError:
            raw_ref = self.store.put_json("inference_failures", stable_hash({"request": request, "kind": "storage"}),
                                          {"kind": "INFERENCE_ERROR", "stage": stage,
                                           "sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
                                           "claim_id": claim.claim_id if claim else None,
                                           "reason": "INFERENCE_ARTIFACT_ERROR"})
            result = {"execution_status": ExecutionStatus.INFERENCE_ERROR.value, "raw_response_ref": raw_ref,
                      "raw_text": None, "cache_hit": False, "latency_seconds": 0.0,
                      "failure_reason": "INFERENCE_ARTIFACT_ERROR", "cache_key": None, "attempts": []}
        except (OSError, ValueError):
            raw_ref = self.store.put_json("input_failures", stable_hash({"request": request, "kind": "invalid_input"}),
                                          {"kind": "INVALID_INPUT", "stage": stage,
                                           "sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
                                           "claim_id": claim.claim_id if claim else None,
                                           "reason": "INVALID_INFERENCE_INPUT"})
            result = {"execution_status": ExecutionStatus.INVALID_INPUT.value, "raw_response_ref": raw_ref,
                      "raw_text": None, "cache_hit": False, "latency_seconds": 0.0,
                      "failure_reason": "INVALID_INFERENCE_INPUT", "cache_key": None, "attempts": []}
        references = {"sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
                      "claim_id": claim.claim_id if claim else None, "frame_ids": frame_ids,
                      "image_paths": request["image_paths"], "variant": variant,
                      "region": list(region) if region is not None else None,
                      "cache_key": result["cache_key"], "control_index": control_index}
        if stage != "semantic":
            self.record("proposal_inference", {"references": references, "stage": stage,
                                               "raw_response_ref": result["raw_response_ref"],
                                               "execution_status": result["execution_status"]})
            return result
        if result["execution_status"] == "OK":
            verdict = parse_verification(result["raw_text"], raw_response_ref=result["raw_response_ref"],
                                         prompt_version=PROMPT_VERSIONS[stage], input_references=references)
        else:
            verdict = VerificationResult(None, ExecutionStatus(result["execution_status"]),
                                         result["raw_response_ref"], PROMPT_VERSIONS[stage], references,
                                         result.get("failure_reason"))
        self.record("verification", {"sample_id": sample.sample_id, "result": verdict})
        self.verification_memo[memo_key] = verdict
        return verdict

    def contrast(self, sample, candidate, claim, original):
        memo_key = (candidate.candidate_id, claim.claim_id)
        if memo_key in self.contrast_memo:
            prior = self.contrast_memo[memo_key]
            self.record("contrast_reuse", {"sample_id": sample.sample_id,
                                            "candidate_id": candidate.candidate_id,
                                            "claim_id": claim.claim_id,
                                            "reason": "SAME_INPUT_REUSED_WITHOUT_NEW_INFERENCE"})
            return prior
        fixture = next((f for f in self.cfg["claims"]["contrast_fixtures"]
                        if f["sample_id"] == sample.sample_id and f["target_text"] == claim.text), None)
        if fixture:
            if not self.inference.backend.synthetic or sample.provenance["source_kind"] != "synthetic":
                raise ValueError("Human synthetic contrast fixtures require synthetic backend and runtime")
            alternatives, group = make_contrast(claim, fixture["alternatives"], candidate,
                        fixture["comparison_dimension"], ExclusivityStatus(fixture["exclusivity_status"]), fixture["source"])
        else:
            response = self.infer(sample, candidate, claim, "contrasts")
            if response["execution_status"] != "OK":
                outcome = {"pass": False, "status": "CONTRAST_TECHNICAL_FAILURE",
                           "reasons": ["CONTRAST_TECHNICAL_FAILURE"], "raw_response_ref": response["raw_response_ref"]}
                self.contrast_memo[memo_key] = outcome
                return outcome
            try:
                alternatives, group = parse_contrasts(response["raw_text"], claim, candidate,
                                                     self.cfg["claims"]["max_alternatives"])
            except (ValueError, TypeError):
                outcome = {"pass": False, "status": "CONTRAST_PARSE_ERROR", "reasons": ["CONTRAST_PARSE_ERROR"],
                           "raw_response_ref": response["raw_response_ref"]}
                self.contrast_memo[memo_key] = outcome
                return outcome
        self.record("contrasts", {"sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
                                   "group": group, "alternatives": alternatives})
        results = {claim.claim_id: original}
        for alternative in alternatives:
            results[alternative.claim_id] = self.infer(sample, candidate, alternative, "semantic")
        outcome = evaluate_contrast(group, results, claim.claim_id, self.cfg["policy"]["strict_alternatives"])
        self.contrast_memo[memo_key] = outcome
        return outcome

    def spatial(self, sample, candidate, claim, original, proposal_index):
        if len(self.proposals) >= self.cfg["budget"]["max_spatial_proposals"]:
            return {"pass": False, "status": "MAX_SPATIAL_PROPOSALS",
                    "reasons": ["MAX_SPATIAL_PROPOSALS"],
                    "require_controls": POLICIES[self.cfg["policy"]["name"]]["controls"],
                    "proposal_count": len(self.proposals), "intervention_protocol": self.intervention}
        if proposal_index == 0:
            response = self.infer(sample, candidate, claim, "spatial")
            proposal = parse_proposal(response["raw_text"] or "", candidate, claim,
                                      raw_response_ref=response["raw_response_ref"])
            if response["execution_status"] != "OK":
                proposal = replace(proposal, parser_status=ExecutionStatus(response["execution_status"]),
                                   provenance={**proposal.provenance, "failure_reason": response.get("failure_reason")})
        else:
            if not self.inference.backend.synthetic:
                return {"pass": False, "status": "RE_GROUNDING_NOT_IMPLEMENTED",
                        "reasons": ["RE_GROUNDING_NOT_IMPLEMENTED"],
                        "require_controls": POLICIES[self.cfg["policy"]["name"]]["controls"],
                        "proposal_count": len(self.proposals),
                        "intervention_protocol": self.intervention,
                        "interpretation": "No claim-conditioned spatial re-grounding is implemented for real evidence admission."}
            box = self.cfg["spatial"]["alternate_regions"][proposal_index - 1]
            proposal = parse_proposal(json.dumps({"support_region": box}), candidate, claim)
            proposal = replace(proposal, provenance={**proposal.provenance,
                      "source": "preconfigured_alternate_region", "proposal_index": proposal_index},
                      proposal_id=stable_id("region", [candidate.candidate_id, claim.claim_id, box]))
        self.proposals.append(proposal)
        self.record("spatial_proposals", {"sample_id": sample.sample_id, "proposal": proposal,
                                          "intervention_protocol": self.intervention})
        if proposal.parser_status != ExecutionStatus.OK:
            return {"pass": False, "status": "SPATIAL_PROPOSAL_FAILURE", "reasons": ["SPATIAL_PROPOSAL_FAILURE"],
                    "proposal": to_dict(proposal), "require_controls": POLICIES[self.cfg["policy"]["name"]]["controls"],
                    "intervention_protocol": self.intervention}
        box = proposal.support_region
        controls_required = POLICIES[self.cfg["policy"]["name"]]["controls"]
        controls = generate_control_regions(box, self.cfg["spatial"]["control_count"] if controls_required else 0)
        self.record("controls", {"sample_id": sample.sample_id, "proposal_id": proposal.proposal_id,
                                 "intervention_protocol": self.intervention, **controls})
        verdicts = {}
        control_verdicts = []
        variants = [("KEEP_TARGET", box, None), ("DROP_TARGET", box, None)]
        if controls_required:
            variants += [("DROP_MATCHED_CONTROL", r, i) for i, r in enumerate(controls["regions"])]
        pixel_refs = []
        # The original image is audited as a no-op baseline.  It is not sent to
        # the VLM again because `original` already records that inference.
        original_audits = []
        for frame in select_frames(sample, candidate):
            with Image.open(frame.path) as source:
                _, audit = apply_spatial_intervention(source.convert("RGB"), box, "ORIGINAL", self.intervention)
            original_audits.append(audit)
            pixel_refs.append(self.record("pixel_audits", {"sample_id": sample.sample_id,
                    "candidate_id": candidate.candidate_id, "proposal_id": proposal.proposal_id,
                    "frame_id": frame.frame_id, "output_path": None, "control_index": None, **audit}))
        if not all(audit["pixel_audit_pass"] for audit in original_audits):
            return {"pass": False, "status": "INTERVENTION_PIXEL_AUDIT_FAILED",
                    "reasons": ["INTERVENTION_PIXEL_AUDIT_FAILED"],
                    "require_controls": controls_required, "proposal": to_dict(proposal),
                    "controls": controls, "pixel_audit_refs": pixel_refs,
                    "intervention_protocol": self.intervention,
                    "interpretation": "Spatial intervention response under this protocol; not a causal proof."}
        for variant, region, control_index in variants:
            paths = []
            audits = []
            for frame in select_frames(sample, candidate):
                with Image.open(frame.path) as source:
                    altered, audit = apply_spatial_intervention(source.convert("RGB"), region, variant, self.intervention)
                path = _image_file(self.inference.store.root, altered)
                paths.append(path)
                audits.append(audit)
                pixel_refs.append(self.record("pixel_audits", {"sample_id": sample.sample_id,
                        "candidate_id": candidate.candidate_id, "proposal_id": proposal.proposal_id,
                        "frame_id": frame.frame_id, "output_path": path, "control_index": control_index, **audit}))
            invariants_ok = all(audit["unchanged_region_pass"] and audit["resolution_preserved"] for audit in audits)
            any_expected_change = any(audit["expected_region_changed"] for audit in audits)
            if not invariants_ok or not any_expected_change:
                status = "INTERVENTION_PIXEL_AUDIT_FAILED" if not invariants_ok else "INTERVENTION_NO_EFFECT"
                return {"pass": False, "status": status, "reasons": [status],
                        "require_controls": controls_required, "proposal": to_dict(proposal),
                        "controls": controls, "pixel_audit_refs": pixel_refs,
                        "intervention_variant": variant, "intervention_protocol": self.intervention,
                        "interpretation": "Spatial intervention response under this protocol; not a causal proof."}
            verdict = self.infer(sample, candidate, claim, "semantic", variant, paths, region,
                                 proposal_index, control_index)
            if variant == "DROP_MATCHED_CONTROL":
                control_verdicts.append(verdict)
            else:
                verdicts[variant] = verdict
        result = check_spatial(original, verdicts["KEEP_TARGET"], verdicts["DROP_TARGET"], control_verdicts,
                               controls["available"], controls_required,
                               expected_control_count=self.cfg["spatial"]["control_count"] if controls_required else 0)
        area = (box[2] - box[0]) * (box[3] - box[1])
        result.update(proposal=to_dict(proposal), controls=controls, pixel_audit_refs=pixel_refs,
                      intervention_protocol=self.intervention,
                      support_area_fraction=area, large_region_warning=area >= self.cfg["spatial"]["max_area_warning"])
        return result

    def pair(self, sample, candidate, claim, proposal_index):
        policy = self.cfg["policy"]["name"]
        if policy == "acquisition_only":
            original = VerificationResult(None, ExecutionStatus.INVALID_INPUT, None, PROMPT_VERSIONS["semantic"],
                         {"candidate_id": candidate.candidate_id, "claim_id": claim.claim_id,
                          "frame_ids": list(candidate.frame_ids)}, "ACQUISITION_ONLY_NOT_RUN")
        else:
            original = self.infer(sample, candidate, claim, "semantic")
        spatial, contrast = None, None
        try:
            if original.semantic_status == SemanticStatus.SUPPORTED:
                if POLICIES[policy]["contrast"]:
                    contrast = self.contrast(sample, candidate, claim, original)
                if POLICIES[policy]["spatial"]:
                    spatial = self.spatial(sample, candidate, claim, original, proposal_index)
        except BudgetExceeded:
            certificate = self.certificate(sample, candidate, claim, original, spatial, contrast)
            self.certificates.append(certificate)
            raise
        return self.certificate(sample, candidate, claim, original, spatial, contrast)

    def certificate(self, sample, candidate, claim, original, spatial, contrast):
        cert = build_certificate(candidate, claim, original, self.cfg["policy"]["name"], spatial, contrast,
                {"synthetic": self.inference.backend.synthetic, "frame_ids": list(candidate.frame_ids),
                 "source_runtime_sha256": sample.provenance["runtime_sha256"],
                 "config_policy": self.cfg["policy"], "frozen_model": self.inference.backend.fingerprint(),
                 "intervention_protocol": self.intervention})
        self.record("certificates", cert)
        return cert

    def coverage_and_answer(self, sample):
        required = ([sample.target_claim] if sample.task == "claim_verification" else list(sample.required_claims))
        relations = sample.metadata.get("unresolved_relations", [])
        coverage = assess_coverage(required, self.certificates, relations)
        return coverage, strict_answer(sample, self.claims, self.certificates, coverage)

    def run(self, sample):
        if sample.task not in SUPPORTED_TASKS:
            return {"sample_id": sample.sample_id, "task": sample.task, "synthetic": self.inference.backend.synthetic,
                    "status": "UNSUPPORTED", "termination_reason": "UNSUPPORTED_TASK",
                    "strict": {"status": "UNSUPPORTED", "answer": None, "certificate_ids": [], "claim_ids": [],
                               "input_references": []}, "certificates": [], "claims": [], "adaptation_events": [],
                    "coverage": {"status": "UNRESOLVED", "required_claims": [], "missing_claims": [],
                                 "supported_claims": [], "unresolved_relations": ["UNSUPPORTED_TASK"]},
                    "usage": self.budget.snapshot()}
        pool = acquire(sample, self.cfg["acquisition"])
        pool = pool[:self.cfg["budget"]["max_candidates"]]
        self.record("candidate_pool", {"sample_id": sample.sample_id, "candidates": pool})
        supplied = ([sample.target_claim] if sample.target_claim else []) + list(sample.required_claims)
        self.claims = list({c.claim_id: c for c in supplied}.values())
        termination = "CANDIDATES_EXHAUSTED"
        before = None
        current_candidate = current_claim = current_certificate = None
        try:
            if not self.claims:
                # Runtime validation requires a target/required claim before a
                # supported task reaches the runner.  Keep this defensive path
                # model-free so a malformed direct RuntimeSample cannot turn a
                # result into a post-hoc requirement.
                termination = "INVALID_INPUT_NO_FROZEN_REQUIREMENTS"
            self.record("claims", {"sample_id": sample.sample_id, "claims": self.claims})
            for claim_index, claim in enumerate(self.claims):
                current_claim = claim
                candidate, pool_index, proposal_index = pool[0], 0, 0
                seen_inputs = set()
                if claim_index:
                    action = "ACQUIRE_MISSING_CLAIM"
                    if action not in self.cfg["adaptation"]["actions"]:
                        termination = "UNSUPPORTED_ACTION"
                        break
                    ev = event(current_candidate, claim, current_certificate, action,
                               {"missing_claim_id": claim.claim_id, "rank_source": "chronological"}, candidate,
                               self.budget.snapshot())
                    self.events.append(ev)
                    self.record("adaptation", ev)
                while True:
                    current_candidate = candidate
                    if self.rounds >= self.cfg["budget"]["max_rounds"]:
                        termination = "MAX_ROUNDS"
                        break
                    key = (tuple(candidate.frame_ids), claim.claim_id, proposal_index)
                    if key in seen_inputs:
                        termination = "DUPLICATE_INPUT"
                        break
                    if candidate.candidate_id not in self.seen_candidates:
                        if len(self.seen_candidates) >= self.cfg["budget"]["max_candidates"]:
                            termination = "MAX_CANDIDATES"
                            break
                        self.seen_candidates.add(candidate.candidate_id)
                        self.candidates.append(candidate)
                        self.record("candidates", candidate)
                    seen_inputs.add(key)
                    self.rounds += 1
                    cert = self.pair(sample, candidate, claim, proposal_index)
                    self.certificates.append(cert)
                    current_certificate = cert
                    if before is None:
                        before = self.coverage_and_answer(sample)[1]
                    coverage, answer = self.coverage_and_answer(sample)
                    if answer["status"] == "ANSWERED":
                        termination = "COVERAGE_COMPLETE"
                        break
                    expanded = expand_candidate(sample, candidate, self.cfg["adaptation"]["expand_frames"])
                    can_expand = expanded is not None and (tuple(expanded.frame_ids), claim.claim_id, 0) not in seen_inputs
                    configured_alternate = (len(self.proposals) < self.cfg["budget"]["max_spatial_proposals"]
                                             and proposal_index + 1 < 1 + len(self.cfg["spatial"]["alternate_regions"]))
                    action, reason = choose_action(cert, enabled=self.cfg["adaptation"]["enabled"],
                              allowed=self.cfg["adaptation"]["actions"], can_expand=can_expand,
                              can_alternate=(self.inference.backend.synthetic and configured_alternate),
                              can_next=pool_index + 1 < len(pool))
                    previous = candidate
                    params = {"round_index": self.rounds, "decision_reason": reason}
                    if action == "EXPAND_TEMPORAL_CONTEXT":
                        candidate, proposal_index = expanded, 0
                        params["extra_frames_each_side"] = self.cfg["adaptation"]["expand_frames"]
                    elif action == "TRY_ALTERNATE_SUPPORT_REGION":
                        proposal_index += 1
                        params.update(proposal_index=proposal_index,
                                      support_region=self.cfg["spatial"]["alternate_regions"][proposal_index - 1])
                    elif action == "NEXT_CANDIDATE":
                        pool_index += 1
                        candidate, proposal_index = pool[pool_index], 0
                        params["acquisition_rank"] = candidate.acquisition_rank
                    ev = event(previous, claim, cert, action, params, candidate if action != "STOP" else None,
                               self.budget.snapshot(), reason if action == "STOP" else None)
                    self.events.append(ev)
                    self.record("adaptation", ev)
                    if action == "STOP":
                        termination = reason
                        break
                if self.coverage_and_answer(sample)[1]["status"] == "ANSWERED":
                    break
                if termination in {"MAX_ROUNDS", "MAX_CANDIDATES", "UNSUPPORTED_ACTION"}:
                    break
        except BudgetExceeded:
            termination = "CALL_BUDGET_EXHAUSTED"
        coverage, strict = self.coverage_and_answer(sample)
        if strict["status"] == "ANSWERED":
            termination = "COVERAGE_COMPLETE"
        for action in self.cfg["adaptation"]["actions"]:
            if action_status(action) == "UNSUPPORTED_ACTION":
                self.record("unsupported_actions", {"sample_id": sample.sample_id, "action": action,
                                                    "status": "UNSUPPORTED_ACTION"})
        stop = event(current_candidate, current_claim, current_certificate, "STOP", {"rounds": self.rounds},
                     None, self.budget.snapshot(), termination)
        self.events.append(stop)
        self.record("adaptation", stop)
        result = {"sample_id": sample.sample_id, "task": sample.task, "synthetic": self.inference.backend.synthetic,
                  "status": "COMPLETE", "claims": self.claims, "candidates": self.candidates,
                  "spatial_proposals": self.proposals, "certificates": self.certificates, "coverage": coverage,
                  "strict": strict, "before_adaptation_strict": before or strict, "adaptation_events": self.events,
                  "termination_reason": termination, "usage": self.budget.snapshot(),
                  "timing": timing_summary(sample.frames), "provenance": sample.provenance}
        if self.cfg["reasoning"]["mode"] == "benchmark_forced":
            result["benchmark_forced"] = forced_answer(sample, self.claims, pool, strict)
        result = to_dict(result)
        audit = audit_strict_result(result)
        if audit["status"] != "PASS":
            raise RuntimeError(f"STRICT_SOURCE_AUDIT_FAILED: {audit['issues']}")
        self.record("strict_audits", audit)
        return result


def run(config, runtime_path, output_dir, *, cache_dir=None, max_samples=5, phase="smoke", backend=None):
    config = validate_config(config)
    if config["policy"]["version"] != POLICY_VERSION:
        raise ValueError("Policy version must match implemented frozen policy")
    if config["spatial"]["control_version"] != CONTROL_VERSION:
        raise ValueError("Control version must match implemented geometry protocol")
    if phase not in {"smoke", "preflight"}:
        raise ValueError("phase must be smoke or preflight")
    maximum_for_phase = 5 if phase == "smoke" else 10
    if type(max_samples) is not int or not 1 <= max_samples <= maximum_for_phase:
        raise ValueError(f"ReliVE-v1 {phase} is limited to 1–{maximum_for_phase} samples; no full benchmark entry")
    samples = load_runtime(runtime_path)[:max_samples]
    backend = backend or make_backend(config["backend"])
    if not backend.synthetic and any(s.provenance["source_kind"] == "synthetic" for s in samples):
        raise ValueError("Real runs require public_runtime sources; synthetic fixtures stay separate")
    output_parts = Path(output_dir).expanduser().resolve().parts
    if backend.synthetic and "real" in output_parts:
        raise ValueError("Synthetic runs cannot write under a real output directory")
    if not backend.synthetic and "mock" in output_parts:
        raise ValueError("Real runs cannot write under a mock output directory")
    preparation_audit = None
    if not backend.synthetic:
        audit_path = Path(str(Path(runtime_path).expanduser().resolve()) + ".gt_isolation_audit.json")
        try:
            candidate_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Real runs require a readable GT-isolation audit beside the public runtime") from exc
        if (not isinstance(candidate_audit, dict) or candidate_audit.get("status") != "PASS"
                or candidate_audit.get("runtime_sha256") != samples[0].provenance["runtime_sha256"]):
            raise ValueError("Real runtime GT-isolation audit is missing, failed, or does not bind this runtime SHA-256")
        preparation_audit = {"path": str(audit_path), "status": candidate_audit["status"],
                             "adapter": candidate_audit.get("adapter"),
                             "classification": candidate_audit.get("classification")}
    store = ArtifactStore(output_dir)
    cache = ArtifactStore(cache_dir or (store.root / "inference"))
    inference = CachedInference(backend, cache)
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise RuntimeError("Runtime source boundary audit failed")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    frame_hashes = {f.path: hashlib.sha256(Path(f.path).read_bytes()).hexdigest() for s in samples for f in s.frames}
    manifest = {"framework": "ReliVE-v1", "config": config, "config_sha256": stable_hash(config),
                "runtime_path": str(Path(runtime_path).resolve()), "samples": [s.sample_id for s in samples],
                "runtime_sha256": samples[0].provenance["runtime_sha256"] if samples else None,
                "frame_sha256": frame_hashes, "backend": backend.fingerprint(), "synthetic": backend.synthetic,
                "prompt_versions": PROMPT_VERSIONS, "prompt_hashes": {k: stable_hash(prompt(k, {})) for k in PROMPT_VERSIONS},
                "policy_version": POLICY_VERSION, "intervention_version": INTERVENTION_VERSION,
                "intervention_protocol_version": INTERVENTION_PROTOCOL_VERSION,
                "spatial_intervention": config["spatial"]["intervention"],
                "control_version": CONTROL_VERSION, "git_commit": commit, "python": sys.version,
                "platform": platform.platform(), "pillow": pillow_version, "cache_dir": str(cache.root),
                "runtime_source_audit": source_audit, "model_parameters_updated": False,
                "runtime_gt_isolation_audit": preparation_audit,
                "phase": phase, "limits": f"{phase} run only; at most {maximum_for_phase} samples; no GT loading."}
    results = []
    new_calls, cached_samples = 0, 0
    started = time.monotonic()
    with store.run_lock():
        store.put_json("manifest", "run", manifest)
        for sample in samples:
            key = stable_id("sample", sample.sample_id)
            path = store.path("samples", key)
            if path.exists():
                result = store.get_json("samples", key)
                if audit_strict_result(result)["status"] != "PASS":
                    raise ValueError("Completed sample failed strict audit")
                cached_samples += 1
            else:
                engine = SampleRunner(config, store, inference)
                result = engine.run(sample)
                new_calls += engine.budget.new_calls
                store.put_json("samples", key, result)
            results.append(result)
        summary = {"framework": "ReliVE-v1", "synthetic": backend.synthetic, "n_samples": len(results),
                   "strict_answered": sum(r["strict"]["status"] == "ANSWERED" for r in results),
                   "certificate_distribution": dict(Counter(c["final_status"] for r in results for c in r["certificates"])),
                   "termination_distribution": dict(Counter(r["termination_reason"] for r in results)),
                   "logical_model_calls": sum(r["usage"]["calls"] for r in results),
                   "new_model_calls": new_calls, "resumed_completed_samples": cached_samples,
                   "output_dir": str(store.root), "cache_dir": str(cache.root)}
        store.append_event("invocations", {**summary, "invocation_id": uuid.uuid4().hex,
                                           "elapsed_seconds": time.monotonic() - started})
    return summary
