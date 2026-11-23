"""Read-only freezing of public questions and existing Stage 1 requirements."""
from __future__ import annotations

import json
from pathlib import Path

from relive.claim_bank.contracts import ClaimBankError, QA_TYPES
from relive.claim_bank import contracts_v11
from relive.claim_bank.data import digest, load_json, load_runtime, rows, write_once
from relive.claim_bank.pipeline import validate_bank
from relive.storage.artifacts import stable_hash
from .contract import SMOKE22

REFERENCE_KEYS = {"sample_key", "source_record_index", "dataset_sha256", "qa_type", "question",
                  "requirements", "question_sha256", "requirements_sha256", "stage1_source"}

def new_directory(path):
    p = Path(path).expanduser().resolve()
    if p.exists():
        raise ClaimBankError("NEW_OUTPUT_DIRECTORY_REQUIRED")
    p.mkdir(parents=True)
    return p

def seal(root, names, format_name, **fields):
    manifest = {"format": format_name, **fields, "artifact_sha256": {n: digest(root / n) for n in names}}
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    write_once(root / "manifest.json", manifest)
    return manifest

def check_seal(root):
    root = Path(root)
    m = load_json(root / "manifest.json")
    if stable_hash({k: v for k, v in m.items() if k != "manifest_content_sha256"}) != m.get("manifest_content_sha256"):
        raise ClaimBankError("MANIFEST_HASH_MISMATCH")
    for name, h in m["artifact_sha256"].items():
        if Path(name).name != name or digest(root / name) != h:
            raise ClaimBankError("ARTIFACT_HASH_MISMATCH")
    return m

def load_references(root, *, include_legacy=False):
    m = check_seal(root)
    if m["format"] != "relive-stage2-v2-frozen-references-v1":
        raise ClaimBankError("REFERENCE_FORMAT_INVALID")
    data = rows(Path(root) / "frozen_requirements.jsonl")
    if [r["source_record_index"] for r in data] != SMOKE22 or len(data) != 22:
        raise ClaimBankError("SMOKE_SELECTION_CHANGED")
    if sum(len(r["requirements"]) for r in data) != 100:
        raise ClaimBankError("SMOKE_REQUIREMENT_COUNT_CHANGED")
    if len({r["sample_key"] for r in data}) != 22 or len({r["dataset_sha256"] for r in data}) != 1:
        raise ClaimBankError("REFERENCE_IDENTITIES_INVALID")
    for r in data:
        if set(r) != REFERENCE_KEYS or r["question_sha256"] != stable_hash(r["question"]) or r["requirements_sha256"] != stable_hash(r["requirements"]):
            raise ClaimBankError("REFERENCE_HASH_OR_ALLOWLIST_INVALID")
        expected = f'medvidu-{r["dataset_sha256"]}-{r["source_record_index"]:06d}'
        if r["sample_key"] != expected:
            raise ClaimBankError("REFERENCE_SOURCE_IDENTITY_INVALID")
        if r["qa_type"] not in QA_TYPES:
            raise ClaimBankError("REFERENCE_QA_TYPE_INVALID")
        _, errors = contracts_v11.parse(json.dumps({"planning_status": "PLANNED", "planning_reason": None, "requirements": r["requirements"]}), 1)
        if errors:
            raise ClaimBankError("FROZEN_STAGE1_REQUIREMENTS_INVALID")
    baseline = None
    if include_legacy:
        baseline = rows(Path(root) / "legacy_comparison.jsonl")
        if [r["source_record_index"] for r in baseline] != SMOKE22:
            raise ClaimBankError("LEGACY_SELECTION_CHANGED")
        for r, b in zip(data, baseline):
            if any(r[k] != b[k] for k in ("sample_key", "question", "qa_type", "requirements")):
                raise ClaimBankError("LEGACY_STAGE1_REQUIREMENTS_CHANGED")
    return data, baseline, m

def freeze(*, runtime_manifest, legacy_diagnostic, output_dir, stage1_dir=None, allow_stage2_requirements=False):
    """The diagnostic fallback is traceable, not an independent bank attestation.

    If Stage 1 is supplied, compare all 100 requirements against the old Stage 2
    copy. A mismatch is never reconciled. No frame paths/metadata enter references.
    """
    runtime = {r["source_record_index"]: r for r in load_runtime(runtime_manifest)}
    diag = load_json(legacy_diagnostic)
    old = sorted(diag["stage2"]["samples"], key=lambda r: r["source_record_index"])
    if [r["source_record_index"] for r in old] != SMOKE22:
        raise ClaimBankError("SMOKE_SELECTION_CHANGED")
    inputs = {str(Path(runtime_manifest).resolve()): digest(runtime_manifest),
              str(Path(legacy_diagnostic).resolve()): digest(legacy_diagnostic)}
    s1, source = {}, {"kind": "FALLBACK_STAGE2_CARRIED_REQUIREMENTS_ONLY", "source_sha256": digest(legacy_diagnostic),
                       "independent_stage1_bank_checked": False,
                       "synthetic_origin": "REAL" if diag["stage2"].get("summary", {}).get("synthetic_test") is False else "SYNTHETIC" if diag["stage2"].get("summary", {}).get("synthetic_test") is True else "UNKNOWN"}
    if stage1_dir:
        validate_bank(stage1_dir)
        m = load_json(Path(stage1_dir) / "bank_manifest.json")
        if m["stage"] != 1 or m["dry_run"] or m["runtime_manifest_sha256"] != digest(runtime_manifest) or m["protocol_sha256"] != contracts_v11.protocol()["protocol_sha256"]:
            raise ClaimBankError("STAGE1_SOURCE_BINDING_INVALID")
        s1 = {r["source_record_index"]: r for r in rows(Path(stage1_dir) / "claim_bank.jsonl")}
        source = {"kind": "VALIDATED_STAGE1_BANK", "source_sha256": digest(Path(stage1_dir) / "bank_manifest.json"),
                  "independent_stage1_bank_checked": True,
                  "synthetic_origin": "SYNTHETIC" if m["synthetic_test"] else "REAL"}
        inputs[str((Path(stage1_dir) / "bank_manifest.json").resolve())] = source["source_sha256"]
    elif not allow_stage2_requirements:
        raise ClaimBankError("STAGE1_REQUIRED_OR_EXPLICIT_FALLBACK")
    frozen, comparison = [], []
    for old_row in old:
        index = old_row["source_record_index"]
        public = runtime.get(index)
        if public is None or any(public[k] != old_row[k] for k in ("sample_key", "question", "qa_type")):
            raise ClaimBankError("LEGACY_PUBLIC_QUESTION_MISMATCH")
        base = s1.get(index) if stage1_dir else old_row
        if base is None or any(base[k] != old_row[k] for k in ("sample_key", "question", "qa_type", "requirements")):
            raise ClaimBankError("STAGE1_STAGE2_REQUIREMENTS_MISMATCH")
        frozen.append({**{k: public[k] for k in ("sample_key", "source_record_index", "dataset_sha256", "qa_type", "question")},
                       "requirements": base["requirements"], "question_sha256": stable_hash(public["question"]),
                       "requirements_sha256": stable_hash(base["requirements"]), "stage1_source": source})
        # Legacy claims/raw completions are quarantined for the report only.
        comparison.append({k: old_row[k] for k in ("sample_key", "source_record_index", "qa_type", "question", "requirements", "claims", "generation_status", "failure_reasons", "raw_responses")})
    out = new_directory(output_dir)
    write_once(out / "frozen_requirements.jsonl", frozen, jsonl=True)
    write_once(out / "legacy_comparison.jsonl", comparison, jsonl=True)
    report = {"source": source, "requirement_count": sum(len(r["requirements"]) for r in frozen),
              "stage1_content_changed": False, "questions_changed": False,
              "fallback_limit": None if stage1_dir else "Compared to the frozen Stage 2 carried copy and public runtime only; independent Stage 1-bank verification remains unavailable.",
              "runtime_fields": sorted(REFERENCE_KEYS), "generator_fields": ["question", "qa_type", "requirements"],
              "legacy_claims_used_by_generator": False, "images_opened": 0, "model_calls_made": 0,
              "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}
    write_once(out / "reference_audit.json", report)
    seal(out, ["frozen_requirements.jsonl", "legacy_comparison.jsonl", "reference_audit.json"],
         "relive-stage2-v2-frozen-references-v1", inputs=inputs, source=source, sample_indices=SMOKE22)
    load_references(out, include_legacy=True)
    if any(digest(p) != h for p, h in inputs.items()):
        raise ClaimBankError("SOURCE_CHANGED_DURING_FREEZE")
    return {"status": "PASS", **report, "output_dir": str(out)}
