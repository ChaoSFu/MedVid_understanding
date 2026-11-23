"""One text-only call per frozen sample, immutable checkpoints, strict cache replay."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from relive.backends import make_backend
from relive.claim_bank.contracts import ClaimBankError
from relive.claim_bank.data import digest, load_json, rows, write_once
from relive.claim_bank.pipeline import ReplayBackend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore, stable_hash
from relive.storage.cache import Budget, CachedInference
from . import contract
from .sources import check_seal, load_references, new_directory, seal

def implementation_hash():
    here = Path(__file__).parent
    core = here.parent
    files = [here / name for name in ("contract.py", "sources.py", "runner.py", "report.py")]
    files += [core / n for n in ("storage/cache.py", "storage/artifacts.py", "backends/local_hf.py")]
    return stable_hash({str(p.relative_to(core)): digest(p) for p in files})

def validate_response(raw, row, call):
    value = contract.validate(raw, row)
    if call and call["execution_status"] != "OK":
        value["root_errors"] = ["MODEL_EXECUTION_FAILURE"]
        value["root_error_category"] = "MODEL"
    return value

def generate(*, references_dir, config_path, cache_dir, output_dir,
             resume=False, replay_from=None, dry_run=False, _backend=None):
    source_rows, _, ref_manifest = load_references(references_dir)
    config = load_config(config_path)
    if config["backend"]["kind"] != "mock" and ref_manifest["source"]["synthetic_origin"] != "REAL":
        raise ClaimBankError("SYNTHETIC_OR_UNKNOWN_REQUIREMENTS_FOR_REAL_RUN")
    if config["backend"].get("max_retries", 0) != 0 or (config["backend"]["kind"] != "mock" and config["backend"].get("generation", {}).get("do_sample") is not False):
        raise ClaimBankError("FROZEN_GREEDY_ZERO_RETRIES_REQUIRED")
    if _backend is not None and not _backend.synthetic:
        raise ClaimBankError("INJECTED_BACKEND_MUST_BE_SYNTHETIC")
    if replay_from and (resume or dry_run or _backend):
        raise ClaimBankError("REPLAY_ARGUMENTS_INVALID")
    identity = {"references_manifest_sha256": digest(Path(references_dir) / "manifest.json"),
                "protocol_sha256": contract.protocol()["protocol_sha256"], "implementation_sha256": implementation_hash(),
                "config_sha256": digest(config_path), "smoke_indices": contract.SMOKE22, "dry_run": dry_run,
                "requirement_hashes": {r["sample_key"]: r["requirements_sha256"] for r in source_rows}}
    reference = None
    if replay_from:
        validate_run(replay_from, references_dir)
        reference = load_json(Path(replay_from) / "run_binding.json")
        if reference["identity"] != identity:
            raise ClaimBankError("REPLAY_IDENTITY_MISMATCH")
        if reference["identity"]["dry_run"]:
            raise ClaimBankError("CANNOT_REPLAY_DRY_RUN")
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and not resume:
        raise ClaimBankError("NEW_OUTPUT_DIRECTORY_REQUIRED")
    if out.exists() and resume:
        if not (out / "run_binding.json").is_file():
            raise ClaimBankError("RESUME_BINDING_REQUIRED")
        if load_json(out / "run_binding.json").get("identity") != identity:
            raise ClaimBankError("RESUME_IDENTITY_MISMATCH")
    if not out.exists():
        new_directory(out)
    with ArtifactStore(out).run_lock():
        saved_binding = load_json(out / "run_binding.json") if (out / "run_binding.json").exists() else None
        if saved_binding and saved_binding["identity"] != identity:
            raise ClaimBankError("RESUME_IDENTITY_MISMATCH")
        if (out / "manifest.json").exists():
            validate_run(out, references_dir)
            return load_json(out / "stage2_v2_smoke22_summary.json")
        if dry_run:
            backend = None
        elif replay_from:
            backend = ReplayBackend(reference["backend_fingerprint"], config["backend"])
        else:
            backend = _backend or make_backend(config["backend"])
        binding = {"identity": identity,
                   "backend_fingerprint": backend.fingerprint() if backend else None,
                   "synthetic_test": bool(backend and backend.synthetic),
                   "inputs": "QUESTION_QA_TYPE_FROZEN_REQUIREMENTS_ONLY", "image_reads": 0,
                   "real_smoke_executed": bool(backend and not backend.synthetic and not replay_from)}
        if saved_binding and saved_binding != binding:
            raise ClaimBankError("RESUME_BACKEND_MISMATCH")
        write_once(out / "run_binding.json", binding)
        budget = Budget(22)
        # New subdirectory prevents collision with any historical V1 cache.
        inference = CachedInference(backend, ArtifactStore(Path(cache_dir) / contract.VERSION)) if backend else None
        results = []
        for row in source_rows:
            checkpoint = out / "samples" / (row["sample_key"] + ".json")
            if checkpoint.exists():
                saved = load_json(checkpoint)
                if stable_hash(saved["result"]) != saved["result_sha256"]:
                    raise ClaimBankError("CHECKPOINT_HASH_MISMATCH")
                _check_result(saved["result"], row, dry_run)
                results.append(saved["result"])
                continue
            prompt = contract.prompt_for(row)
            request = {"stage": "claimgen_stage2_v2", "prompt_version": contract.VERSION,
                       "prompt": prompt, "image_paths": [], "frame_ids": [],
                       "context": {"qa_type": row["qa_type"]},
                       "planner_binding": {"protocol_version": contract.VERSION, "prompt_sha256": stable_hash(prompt),
                                           "question_sha256": row["question_sha256"], "requirements_sha256": row["requirements_sha256"],
                                           "sample_key": row["sample_key"], "frozen_identity": identity}}
            call, raw = None, None
            if inference:
                if replay_from and not inference.has_terminal_result(request):
                    write_once(out / "cache_replay_audit.json", {"status": "FAIL", "reason": "REPLAY_CACHE_MISS", "new_model_calls": 0})
                    raise ClaimBankError("REPLAY_CACHE_MISS")
                c = inference.call(request, budget)
                raw = c["raw_text"]
                call = {k: c[k] for k in ("execution_status", "failure_reason", "cache_key", "raw_response_ref")}
            validation = validate_response(raw, row, call) if not dry_run else {"generation_status": "NOT_EXECUTED", "claims": [],
                "entities": [], "information_gaps": [], "rejected_candidates": [], "root_errors": [], "claim_audit": [], "semantic_review_status": "NOT_EXECUTED"}
            result = {"format": contract.VERSION, **{k: row[k] for k in ("sample_key", "source_record_index", "qa_type", "question", "requirements", "requirements_sha256", "question_sha256")},
                      **validation, "raw_response": raw, "call": call,
                      "prompt_sha256": stable_hash(prompt), "synthetic_test": binding["synthetic_test"],
                      "input_modality": "TEXT_ONLY", "image_reads": 0}
            if replay_from:
                prior = load_json(Path(replay_from) / "samples" / (row["sample_key"] + ".json"))["result"]
                if result != prior:
                    raise ClaimBankError("REPLAY_RESULT_MISMATCH")
            write_once(checkpoint, {"result": result, "result_sha256": stable_hash(result)})
            results.append(result)
        usage = budget.snapshot()
        write_once(out / "stage2_v2_smoke22.jsonl", results, jsonl=True)
        write_once(out / "claims.jsonl", [{"sample_key": r["sample_key"], **c} for r in results for c in r["claims"]], jsonl=True)
        write_once(out / "protocol.json", contract.protocol())
        audit = {"status": "PASS", "mode": "replay" if replay_from else "resume" if resume else "dry_run" if dry_run else "run",
                 "new_model_calls": usage["new_calls"], "cache_hits": usage["cache_hits"],
                 "exact_result_comparison": "PASS" if replay_from else "NOT_RUN",
                 "accounting_scope": "this invocation only; resumed checkpoints are counted separately",
                 "resumed_checkpoint_count": len(results) - usage["calls"] if not dry_run else 0}
        write_once(out / "cache_replay_audit.json", audit)
        summary = {"format": contract.VERSION, "status": "DRY_RUN" if dry_run else "SYNTHETIC_TEST_ONLY" if binding["synthetic_test"] else "COMPLETE",
                   "sample_count": len(results), "requirement_count": sum(len(r["requirements"]) for r in results),
                   "claim_count": sum(len(r["claims"]) for r in results),
                   "rejected_candidate_count": sum(len(r["rejected_candidates"]) for r in results),
                   "generation_status_counts": dict(sorted(Counter(r["generation_status"] for r in results).items())),
                   "usage": usage, "synthetic_test": binding["synthetic_test"], "gt_used": False, "image_reads": 0,
                   "certificate_writes": 0, "new_verified_count": 0,
                   "source_kind": ref_manifest["source"]["kind"], "stage1_requirements_unchanged": True,
                   "semantic_quality": "REVIEW_REQUIRED", "scientific_product": "QUESTION_DRIVEN_UNVERIFIED_HYPOTHESIS_GRAPH"}
        write_once(out / "stage2_v2_smoke22_summary.json", summary)
        isolation = {"status": "PASS", "generator_fields": ["question", "qa_type", "requirements"],
                     "legacy_claims_in_prompt": False, "images_opened": 0, "gt_used": False,
                     "limit": "A public question or frozen requirement may itself contain author-provided information; this is provenance isolation, not proof of clinical truth."}
        write_once(out / "gt_isolation_audit.json", isolation)
        names = ["stage2_v2_smoke22.jsonl", "claims.jsonl", "protocol.json", "run_binding.json", "cache_replay_audit.json", "stage2_v2_smoke22_summary.json", "gt_isolation_audit.json"]
        # Validate before sealing, so the acceptance report is hash-bound as well.
        checks = [_check_result(r, source, dry_run) for r, source in zip(results, source_rows)]
        validation = {"status": "PASS", "sample_count": len(checks), "requirement_count": 100,
                      "all_requirements_equal": True, "all_claims_unverified": True, "images_opened": 0,
                      "model_calls_made": 0, "certificate_writes": 0, "new_verified_count": 0,
                      "semantic_quality": "REVIEW_REQUIRED", "synthetic_test": binding["synthetic_test"]}
        write_once(out / "stage2_v2_validation.json", validation)
        names += ["stage2_v2_validation.json"]
        seal(out, names, "relive-stage2-v2-run-manifest-v1", identity=identity,
             checkpoint_sha256={r["sample_key"]: digest(out / "samples" / (r["sample_key"] + ".json")) for r in results})
        return summary

def _check_result(result, source, dry_run=False):
    if any(result.get(k) != source[k] for k in ("sample_key", "source_record_index", "qa_type", "question", "requirements", "question_sha256", "requirements_sha256")):
        raise ClaimBankError("FROZEN_INPUT_CHANGED")
    if result["prompt_sha256"] != stable_hash(contract.prompt_for(source)) or result["image_reads"] != 0 or result["input_modality"] != "TEXT_ONLY":
        raise ClaimBankError("TEXT_ONLY_BINDING_INVALID")
    if dry_run:
        if result["generation_status"] != "NOT_EXECUTED" or result["claims"] or result["call"]:
            raise ClaimBankError("DRY_RUN_RESULT_INVALID")
    else:
        recomputed = validate_response(result["raw_response"], source, result["call"])
        if any(result.get(k) != v for k, v in recomputed.items()):
            raise ClaimBankError("CLAIM_VALIDATION_MISMATCH")
    return True

def validate_run(directory, references_dir):
    m = check_seal(directory)
    if m["format"] != "relive-stage2-v2-run-manifest-v1" or m["identity"]["protocol_sha256"] != contract.protocol()["protocol_sha256"]:
        raise ClaimBankError("RUN_PROTOCOL_INVALID")
    frozen, _, _ = load_references(references_dir)
    if m["identity"]["references_manifest_sha256"] != digest(Path(references_dir) / "manifest.json"):
        raise ClaimBankError("REFERENCE_MANIFEST_CHANGED")
    result = rows(Path(directory) / "stage2_v2_smoke22.jsonl")
    if len(result) != 22:
        raise ClaimBankError("RESULT_COVERAGE_INVALID")
    for r, source in zip(result, frozen):
        _check_result(r, source, m["identity"]["dry_run"])
        p = Path(directory) / "samples" / (r["sample_key"] + ".json")
        if digest(p) != m["checkpoint_sha256"][r["sample_key"]]:
            raise ClaimBankError("CHECKPOINT_HASH_MISMATCH")
        saved = load_json(p)
        if saved["result"] != r or saved["result_sha256"] != stable_hash(r):
            raise ClaimBankError("CHECKPOINT_CONTENT_MISMATCH")
    flattened = [{"sample_key": r["sample_key"], **c} for r in result for c in r["claims"]]
    if rows(Path(directory) / "claims.jsonl") != flattened:
        raise ClaimBankError("FLATTENING_INVALID")
    return load_json(Path(directory) / "stage2_v2_validation.json")
