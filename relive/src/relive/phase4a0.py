"""Phase 4A-0: zero-model, blinded human claim--evidence eligibility audit.

This module intentionally has no dependency on a backend, runner, verifier, or
certificate builder.  It packages already frozen public frames and registered
pixel interventions for independent human review before any Phase 4A scorer is
allowed to run.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .interventions import apply_spatial_intervention, resolve_intervention_spec
from .spatial import validate_region

FORMAT = "relive-phase4a0-claim-evidence-eligibility-v1"
SPEC_FORMAT = "relive-phase4a0-candidate-audit-spec-v1"
ELIGIBLE = "ELIGIBLE_FOR_PHASE4A_POSITIVE_CONTROL"
FINAL_STATUSES = {ELIGIBLE, "INELIGIBLE_SINGLE_ROI", "REQUIRES_ADJUDICATION", "AWAITING_HUMAN_REVIEWS", "TECHNICAL_FAILURE"}
VARIANTS = ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL")
BLIND_STATUSES = {"SUPPORTED", "INSUFFICIENT", "CONTRADICTED", "UNREADABLE"}
CLAIM_WORDING = {"CLEAR", "AMBIGUOUS"}
TEMPORAL_SCOPE = {"VALID", "MISMATCH"}
OBSERVABILITY = {"DIRECTLY_VISIBLE", "APPARENT_2D_ONLY", "NOT_VISUALLY_SUPPORTED"}
LABELS = {"SINGLE_ROI_ELIGIBLE", "RELATIONAL_COMPOSITE_REQUIRED", "TEMPORAL_TUBE_REQUIRED", "AMBIGUOUS_VISUAL_FACT", "CLAIM_NOT_SUPPORTED"}
REASONS = {"TEMPORAL_SCOPE_MISMATCH", "UNCLEAR_ENTITY_REFERENCE", "PHYSICAL_CONTACT_NOT_DIRECTLY_OBSERVABLE", "OBJECT_ATTRIBUTE_NOT_VISIBLE", "RELATION_REQUIRES_MULTIPLE_COMPONENTS", "STATIC_ROI_INADEQUATE_ACROSS_FRAMES", "HUMAN_VARIANT_PATTERN_FAILED"}
FORBIDDEN = {"reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "semantic_status", "certificate_status", "keep", "drop", "phase37", "human_public_visual_confirmation_note"}


class Phase4A0Error(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def claim_sha256(text: str) -> str:
    """Bind claim text exactly as the frozen Phase 3.5 manifest does."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4A0Error(f"unreadable JSON: {path}") from exc


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4A0Error(f"unreadable JSONL: {path}") from exc
    if not values or not all(isinstance(v, dict) for v in values):
        raise Phase4A0Error("CANDIDATE_AUDIT_MANIFEST_REQUIRES_NONEMPTY_OBJECT_ROWS")
    return values


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        bad = set(value).intersection(FORBIDDEN)
        if bad:
            raise Phase4A0Error("GT_OR_HISTORICAL_OUTCOME_FIELD_FORBIDDEN:" + ",".join(sorted(bad)))
        for child in value.values(): _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value: _reject_forbidden(child)


def validate_candidate_specs(path: Path) -> list[dict[str, Any]]:
    rows = _rows(path)
    identifiers: set[str] = set()
    required = {"format", "audit_case_id", "source_claim_id", "claim_text", "claim_sha256", "source_record_index", "public_record_sha256", "frame_orders", "frame_paths", "frame_sha256", "frozen_support_region", "coordinate_system", "intervention", "matched_control_regions", "candidate_manifest_sha256", "historical_pilot", "gt_used"}
    for row in rows:
        _reject_forbidden(row)
        if not required.issubset(row): raise Phase4A0Error("CANDIDATE_AUDIT_SPEC_MISSING_REQUIRED_FIELD")
        development_control = row.get("development_control", False)
        if (row["format"] != SPEC_FORMAT or row["gt_used"] is not False
                or not isinstance(row["historical_pilot"], bool) or not isinstance(development_control, bool)
                or row["historical_pilot"] == development_control):
            raise Phase4A0Error("CANDIDATE_AUDIT_SPEC_INVALID_PROVENANCE")
        if not isinstance(row["audit_case_id"], str) or row["audit_case_id"] in identifiers:
            raise Phase4A0Error("CANDIDATE_AUDIT_CASE_ID_INVALID_OR_DUPLICATE")
        identifiers.add(row["audit_case_id"])
        if not isinstance(row["claim_text"], str) or claim_sha256(row["claim_text"]) != row["claim_sha256"]:
            raise Phase4A0Error("CLAIM_SHA256_MISMATCH")
        if (not isinstance(row["frame_orders"], list) or not row["frame_orders"] or len(row["frame_orders"]) != len(row["frame_paths"]) or len(row["frame_paths"]) != len(row["frame_sha256"])):
            raise Phase4A0Error("FROZEN_FRAME_SPEC_INVALID")
        if any(type(o) is not int for o in row["frame_orders"]) or row["frame_orders"] != sorted(row["frame_orders"]):
            raise Phase4A0Error("FROZEN_FRAME_ORDERS_INVALID")
        for frame, expected in zip(row["frame_paths"], row["frame_sha256"]):
            p = Path(frame)
            if not p.is_file(): raise Phase4A0Error(f"FROZEN_ARTIFACT_MISSING:{p}")
            if file_sha(p) != expected: raise Phase4A0Error("PUBLIC_FRAME_SHA256_MISMATCH")
        row["frozen_support_region"] = list(validate_region(row["frozen_support_region"]))
        row["intervention"] = resolve_intervention_spec(row["intervention"], float(row["intervention"].get("parameters", {}).get("blur_radius", 1.0)))
        if not isinstance(row["matched_control_regions"], list) or not row["matched_control_regions"]:
            raise Phase4A0Error("MATCHED_CONTROL_REQUIRED")
        row["matched_control_regions"] = [list(validate_region(r)) for r in row["matched_control_regions"]]
    return rows


def _save_variants(spec: dict[str, Any], target: Path) -> tuple[dict[str, list[Path]], dict[str, list[dict[str, Any]]]]:
    paths: dict[str, list[Path]] = {kind: [] for kind in VARIANTS}
    audits: dict[str, list[dict[str, Any]]] = {kind: [] for kind in VARIANTS}
    for index, source in enumerate(spec["frame_paths"]):
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            regions = {"ORIGINAL": spec["frozen_support_region"], "KEEP_TARGET": spec["frozen_support_region"], "DROP_TARGET": spec["frozen_support_region"], "DROP_MATCHED_CONTROL": spec["matched_control_regions"][0]}
            for kind in VARIANTS:
                rendered, audit = apply_spatial_intervention(image, regions[kind], kind, spec["intervention"])
                out = target / f"{spec['audit_case_id']}_{kind.lower()}_{index:02d}.png"
                out.parent.mkdir(parents=True, exist_ok=True); rendered.save(out)
                paths[kind].append(out); audits[kind].append(audit)
    return paths, audits


def _contact_sheet(images: list[Path], destination: Path, title: str) -> None:
    opened = [Image.open(path).convert("RGB") for path in images]
    try:
        width = max(image.width for image in opened); height = max(image.height for image in opened)
        canvas = Image.new("RGB", (width * len(opened), height + 44), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), title, fill="black", font=ImageFont.load_default())
        for idx, image in enumerate(opened): canvas.paste(image, (idx * width, 44))
        destination.parent.mkdir(parents=True, exist_ok=True); canvas.save(destination)
    finally:
        for image in opened: image.close()


def _packet_schema() -> dict[str, Any]:
    return {"format": FORMAT, "strict": True, "reviewer_fields": ["reviewer_id", "packet_id", "case_blind_id", "variant_assessments", "claim_wording", "temporal_scope", "observability", "eligibility_label", "reason_codes", "notes"], "closed_sets": {"variant_status": sorted(BLIND_STATUSES), "claim_wording": sorted(CLAIM_WORDING), "temporal_scope": sorted(TEMPORAL_SCOPE), "observability": sorted(OBSERVABILITY), "eligibility_label": sorted(LABELS), "reason_codes": sorted(REASONS)}}


def prepare(candidate_manifest: Path, output_dir: Path, *, seed: int = 0) -> dict[str, Any]:
    """Create a blind, zero-model packet from a pre-frozen public candidate manifest."""
    specs = validate_candidate_specs(candidate_manifest)
    if output_dir.exists() and any(output_dir.iterdir()): raise Phase4A0Error("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated, audits = output_dir / "generated_images", {}
    rendered: dict[str, dict[str, list[Path]]] = {}
    for spec in specs:
        rendered[spec["audit_case_id"]], audits[spec["audit_case_id"]] = _save_variants(spec, generated)
    randomizer = random.Random(seed)
    shuffled = sorted(specs, key=lambda row: row["audit_case_id"]); randomizer.shuffle(shuffled)
    mapping: dict[str, Any] = {"format": FORMAT, "seed": seed, "mapping_is_not_for_review": True, "cases": {}}
    packet = output_dir / "review_packet"; template: list[dict[str, Any]] = []
    for number, spec in enumerate(shuffled, 1):
        blind_case = f"case-{number:03d}"; kinds = list(VARIANTS); randomizer.shuffle(kinds)
        variant_map = {chr(ord("A") + index): kind for index, kind in enumerate(kinds)}
        mapping["cases"][blind_case] = {"audit_case_id": spec["audit_case_id"], "variants": variant_map, "pixel_audits": audits[spec["audit_case_id"]], "matched_control_available": True}
        for blind_variant, kind in variant_map.items():
            assets = []
            for index, source in enumerate(rendered[spec["audit_case_id"]][kind]):
                asset = packet / "assets" / blind_case / blind_variant / f"frame_{index:02d}.png"
                asset.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, asset); assets.append(asset)
            _contact_sheet(assets, packet / f"{blind_case}_{blind_variant}.png", f"{blind_case} | variant {blind_variant} | chronological public frames")
        # The packet contains only the blinded case, wording and public order; never source/provenance/history.
        (packet / f"{blind_case}.json").write_text(canonical({"case_blind_id": blind_case, "claim": spec["claim_text"], "frame_orders": spec["frame_orders"], "variants": sorted(variant_map)}) + "\n", encoding="utf-8")
        template.append({"reviewer_id": "", "packet_id": sha({"manifest": sha(specs), "seed": seed})[:16], "case_blind_id": blind_case, "variant_assessments": {key: "" for key in sorted(variant_map)}, "claim_wording": "", "temporal_scope": "", "observability": "", "eligibility_label": "", "reason_codes": [], "notes": ""})
    (output_dir / "candidate_audit_manifest.jsonl").write_text("".join(canonical(row) + "\n" for row in specs), encoding="utf-8")
    (output_dir / "candidate_audit_manifest_sha256.json").write_text(canonical({"sha256": sha(specs), "row_count": len(specs)}) + "\n")
    (output_dir / "blind_mapping.json").write_text(canonical(mapping) + "\n")
    (output_dir / "review_template.jsonl").write_text("".join(canonical(row) + "\n" for row in template))
    (output_dir / "review_schema.json").write_text(canonical(_packet_schema()) + "\n")
    (output_dir / "new_calibration_candidate_manifest.template.json").write_text(canonical({"format": "relive-phase4a0-calibration-candidate-template-v1", "selection_status": "AWAITING_HUMAN_FREEZE", "candidates": []}) + "\n")
    readme = "# Phase 4A-0 blind human eligibility packet\n\nThis is a diagnostic-only, zero-model review. Variant letters are blinded. Do not use model outputs, historical certificates, GT, or prior outcomes. Fill two independent copies of `review_template.jsonl`, then validate them with the Phase 4A-0 CLI. A passing eligibility result is not VERIFIED and does not change any certificate.\n"
    (output_dir / "README.md").write_text(readme)
    _write_initial_reports(output_dir, len(specs))
    return {"format": FORMAT, "status": "PASS", "mode": "prepare", "model_calls_made": 0, "backend_loaded": False, "certificate_created": False, "new_verified_count": 0, "gt_used": False, "historical_artifacts_unchanged": True, "candidate_count": len(specs), "candidate_audit_manifest": str(output_dir / "candidate_audit_manifest.jsonl"), "review_packet": str(packet)}


def _write_initial_reports(output_dir: Path, count: int) -> None:
    summary = {"format": FORMAT, "status": "AWAITING_HUMAN_REVIEWS", "candidate_count": count, "model_calls_made": 0, "backend_loaded": False, "certificate_created": False, "new_verified_count": 0, "gt_used": False, "historical_artifacts_unchanged": True}
    (output_dir / "eligibility_report.jsonl").write_text("", encoding="utf-8")
    (output_dir / "eligibility_summary.json").write_text(canonical(summary) + "\n")
    (output_dir / "phase4a0_audit.json").write_text(canonical(summary) + "\n")


def _control_event(run_dir: Path, sample_id: str, region: list[float]) -> tuple[dict[str, Any] | None, Path | None]:
    """Return an already persisted exact control event; never regenerate geometry."""
    proposals = run_dir / "events" / "spatial_proposals"
    proposal_id = None
    if proposals.is_dir():
        for path in sorted(proposals.glob("*.json")):
            value = _read_json(path); proposal = value.get("proposal", {})
            if value.get("sample_id") == sample_id and proposal.get("support_region") == region:
                proposal_id = proposal.get("proposal_id"); break
    if not isinstance(proposal_id, str): return None, None
    root = run_dir / "events" / "controls"
    if not root.is_dir():
        return None, None
    for path in sorted(root.glob("*.json")):
        value = _read_json(path)
        if value.get("sample_id") == sample_id and value.get("proposal_id") == proposal_id:
            controls = value.get("regions")
            if isinstance(controls, list) and controls:
                return value, path
    return None, None


def export_historical_pilots(*, runtime_path: Path, prospective_manifest_path: Path,
                             phase35_v3_run_dir: Path, phase36_run_dir: Path,
                             output_dir: Path, fresh_source_manifest_path: Path | None = None) -> dict[str, Any]:
    """Export only auditable, immutable Phase 3.5/3.6 geometry to Phase 4A-0.

    The function deliberately looks up persisted ``events/controls`` rather
    than calling the control-placement helper.  Thus a missing historical
    control is reported as unavailable instead of being silently reconstructed.
    """
    fresh_source_manifest_path = fresh_source_manifest_path or runtime_path.parent / "fresh_source_manifest.jsonl"
    required = (runtime_path, prospective_manifest_path, fresh_source_manifest_path, phase35_v3_run_dir / "phase35_v3_trace.jsonl",
                phase36_run_dir / "phase36_regrounding_trace.jsonl")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Phase4A0Error("FROZEN_ARTIFACT_MISSING:" + ",".join(missing))
    if output_dir.exists() and any(output_dir.iterdir()): raise Phase4A0Error("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    runtime_rows = _rows(runtime_path)
    # The strict runtime schema deliberately omits source_record_index.  Its
    # immutable sibling source manifest is the only accepted location for the
    # public-record binding.  We extract only these four fields; confirmation
    # prose and every other source value are discarded before packet creation.
    source_rows = _rows(fresh_source_manifest_path)
    source_by_sample: dict[str, dict[str, Any]] = {}
    for item in source_rows:
        if not {"source_record_index", "sample_id", "public_record_sha256", "target_claim", "claim_text_sha256"}.issubset(item):
            raise Phase4A0Error("FRESH_SOURCE_MANIFEST_PUBLIC_PROVENANCE_MISSING")
        sample_id = item["sample_id"]
        if type(item["source_record_index"]) is not int or not isinstance(sample_id, str) or not isinstance(item["public_record_sha256"], str):
            raise Phase4A0Error("FRESH_SOURCE_MANIFEST_PUBLIC_PROVENANCE_INVALID")
        source_by_sample[sample_id] = {key: item[key] for key in ("source_record_index", "sample_id", "public_record_sha256", "target_claim", "claim_text_sha256")}
    prospective = _read_json(prospective_manifest_path)
    if prospective.get("selection_status") != "FROZEN_PRE_SPATIAL_CERTIFICATE_OUTCOMES" or prospective.get("gt_used") is not False:
        raise Phase4A0Error("PHASE35_PROSPECTIVE_MANIFEST_NOT_FROZEN_OR_GT_SAFE")
    selected = {row.get("claim_id"): row for row in prospective.get("selected", []) if isinstance(row, dict)}
    if set(selected) != {"phase35-local-001", "phase35-local-002", "phase35-local-003"}:
        raise Phase4A0Error("HISTORICAL_PILOT_COHORT_BINDING_MISMATCH")
    by_claim = {row.get("target_claim", {}).get("claim_id"): row for row in runtime_rows if isinstance(row.get("target_claim"), dict)}
    if set(selected) - set(by_claim): raise Phase4A0Error("FROZEN_RUNTIME_CLAIM_MISSING")
    traces35 = {row.get("claim_id"): row for row in _rows(phase35_v3_run_dir / "phase35_v3_trace.jsonl")}
    traces36 = {row.get("claim_id"): row for row in _rows(phase36_run_dir / "phase36_regrounding_trace.jsonl")}
    output_dir.mkdir(parents=True)
    exported, cases = [], []
    source_hash = sha({"runtime": file_sha(runtime_path), "prospective": file_sha(prospective_manifest_path),
                       "phase35_trace": file_sha(phase35_v3_run_dir / "phase35_v3_trace.jsonl"),
                       "phase36_trace": file_sha(phase36_run_dir / "phase36_regrounding_trace.jsonl"), "fresh_source_manifest": file_sha(fresh_source_manifest_path)})
    for claim_id, source in (("phase35-local-001", "PHASE36_FROZEN_R1"), ("phase35-local-002", "PHASE35_FROZEN_R0"), ("phase35-local-003", "NO_FROZEN_AUDITABLE_ROI")):
        runtime = by_claim[claim_id]; frames = runtime.get("frames", []); claim = runtime["target_claim"]
        source_public = source_by_sample.get(runtime.get("sample_id"))
        if source_public is None:
            cases.append({"source_claim_id": claim_id, "status": "NO_FROZEN_AUDITABLE_ROI", "roi_source": source, "reason": "PUBLIC_PROVENANCE_MISSING"}); continue
        if claim.get("text") is None or [frame.get("frame_id") for frame in frames] != claim.get("time_scope", {}).get("frame_ids"):
            raise Phase4A0Error("FROZEN_CLAIM_FRAME_BINDING_MISMATCH")
        expected_hash = selected[claim_id].get("claim_text_sha256")
        if (claim_sha256(claim["text"]) != expected_hash or claim_sha256(claim["text"]) != source_public["claim_text_sha256"]
                or source_public["target_claim"].get("claim_id") != claim_id or source_public["target_claim"].get("text") != claim["text"]):
            raise Phase4A0Error("FROZEN_CLAIM_SHA256_MISMATCH")
        runtime_sha = runtime.get("metadata", {}).get("source_record_sha256")
        if runtime_sha != source_public["public_record_sha256"]:
            raise Phase4A0Error("PUBLIC_RECORD_SHA256_BINDING_MISMATCH")
        frame_paths = [Path(frame["path"]) for frame in frames]
        if not all(path.is_file() for path in frame_paths):
            cases.append({"source_claim_id": claim_id, "status": "FROZEN_ARTIFACT_MISSING", "roi_source": source}); continue
        if source == "PHASE36_FROZEN_R1":
            trace, control_dir = traces36.get(claim_id), phase36_run_dir
            region = trace.get("r1_normalized_0_1_xyxy") if isinstance(trace, dict) else None
        elif source == "PHASE35_FROZEN_R0":
            trace, control_dir = traces35.get(claim_id), phase35_v3_run_dir
            region = trace.get("automatic_support_region") if isinstance(trace, dict) else None
        else:
            trace, control_dir, region = None, None, None
        if not isinstance(region, list) or len(region) != 4:
            cases.append({"source_claim_id": claim_id, "status": "NO_FROZEN_AUDITABLE_ROI", "roi_source": source}); continue
        try: region = list(validate_region(region))
        except ValueError:
            cases.append({"source_claim_id": claim_id, "status": "NO_FROZEN_AUDITABLE_ROI", "roi_source": source}); continue
        control, control_path = _control_event(control_dir, runtime["sample_id"], region)
        if control is None:
            cases.append({"source_claim_id": claim_id, "status": "NO_FROZEN_AUDITABLE_ROI", "roi_source": source, "reason": "MATCHED_CONTROL_ARTIFACT_MISSING"}); continue
        intervention = control.get("intervention_protocol")
        try: intervention = resolve_intervention_spec(intervention, float(intervention.get("parameters", {}).get("blur_radius", 1.0)))
        except (AttributeError, ValueError):
            cases.append({"source_claim_id": claim_id, "status": "NO_FROZEN_AUDITABLE_ROI", "roi_source": source, "reason": "FROZEN_OPERATOR_INVALID"}); continue
        spec = {"format": SPEC_FORMAT, "audit_case_id": "historical-" + claim_id.rsplit("-", 1)[-1], "source_claim_id": claim_id,
                "claim_text": claim["text"], "claim_sha256": claim_sha256(claim["text"]), "source_record_index": source_public["source_record_index"],
                "public_record_sha256": source_public["public_record_sha256"], "frame_orders": [frame["order"] for frame in frames],
                "frame_paths": [str(path) for path in frame_paths], "frame_sha256": [file_sha(path) for path in frame_paths],
                "frozen_support_region": region, "coordinate_system": "normalized_0_1_xyxy", "intervention": intervention,
                "matched_control_regions": control["regions"], "candidate_manifest_sha256": prospective["manifest_sha256"],
                "historical_pilot": True, "development_control": False, "gt_used": False}
        exported.append(spec)
        cases.append({"source_claim_id": claim_id, "status": "EXPORTED", "roi_source": source, "candidate_audit_case_id": spec["audit_case_id"],
                      "frozen_control_artifact": str(control_path), "frozen_control_artifact_sha256": file_sha(control_path), "frame_sha256": spec["frame_sha256"]})
    candidate_path = output_dir / "phase4a0_historical_pilot_candidates.jsonl"
    candidate_path.write_text("".join(canonical(row) + "\n" for row in exported), encoding="utf-8")
    report = {"format": FORMAT, "status": "PASS", "mode": "export_historical_pilots", "model_calls_made": 0,
              "backend_loaded": False, "certificate_created": False, "new_verified_count": 0, "gt_used": False,
              "historical_artifacts_unchanged": True, "source_artifact_binding_sha256": source_hash,
              "fresh_source_manifest": str(fresh_source_manifest_path), "fresh_source_manifest_sha256": file_sha(fresh_source_manifest_path),
              "candidate_manifest": str(candidate_path), "candidate_count": len(exported), "cases": cases,
              "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "phase37"]}
    (output_dir / "phase4a0_historical_pilot_provenance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _validate_review(rows: list[dict[str, Any]], template: list[dict[str, Any]]) -> str:
    expected = {row["case_blind_id"]: set(row["variant_assessments"]) for row in template}
    if len(rows) != len(expected) or {row.get("case_blind_id") for row in rows} != set(expected): raise Phase4A0Error("REVIEW_CASE_SET_MISMATCH")
    ids = {row.get("reviewer_id") for row in rows}
    if len(ids) != 1 or not isinstance(next(iter(ids), None), str) or not next(iter(ids), "").strip(): raise Phase4A0Error("REVIEWER_ID_REQUIRED_AND_CONSISTENT")
    reviewer = next(iter(ids))
    packet_ids = {row.get("packet_id") for row in template}
    if len(packet_ids) != 1 or not isinstance(next(iter(packet_ids)), str):
        raise Phase4A0Error("REVIEW_TEMPLATE_PACKET_ID_INVALID")
    expected_packet_id = next(iter(packet_ids))
    for row in rows:
        if set(row) - {"reviewer_id", "packet_id", "case_blind_id", "variant_assessments", "claim_wording", "temporal_scope", "observability", "eligibility_label", "reason_codes", "notes"}: raise Phase4A0Error("REVIEW_UNKNOWN_FIELD")
        values = row.get("variant_assessments")
        if row.get("packet_id") != expected_packet_id:
            raise Phase4A0Error("REVIEW_PACKET_ID_MISMATCH")
        if not isinstance(values, dict) or set(values) != expected[row["case_blind_id"]] or any(value not in BLIND_STATUSES for value in values.values()): raise Phase4A0Error("REVIEW_VARIANT_SCHEMA_INVALID")
        if row.get("claim_wording") not in CLAIM_WORDING or row.get("temporal_scope") not in TEMPORAL_SCOPE or row.get("observability") not in OBSERVABILITY or row.get("eligibility_label") not in LABELS: raise Phase4A0Error("REVIEW_CLOSED_VALUE_INVALID")
        if not isinstance(row.get("reason_codes"), list) or any(value not in REASONS for value in row["reason_codes"]): raise Phase4A0Error("REVIEW_REASON_CODE_INVALID")
    return reviewer


def validate_reviews(output_dir: Path, review_paths: Iterable[Path]) -> dict[str, Any]:
    template = _rows(output_dir / "review_template.jsonl")
    groups = [_rows(path) for path in review_paths]
    reviewers = [_validate_review(rows, template) for rows in groups]
    if len(set(reviewers)) != len(reviewers): raise Phase4A0Error("DUPLICATE_REVIEWER_ID")
    validated = output_dir / "validated_reviews"; validated.mkdir(exist_ok=True)
    for reviewer, rows in zip(reviewers, groups): (validated / f"{reviewer}.jsonl").write_text("".join(canonical(row) + "\n" for row in rows))
    if len(groups) < 2:
        _write_initial_reports(output_dir, len(template)); return {"status": "AWAITING_HUMAN_REVIEWS", "reviewers": reviewers}
    # No variant identity is read until every supplied review has passed strict validation.
    mapping = _read_json(output_dir / "blind_mapping.json")
    by_case = [{row["case_blind_id"]: row for row in group} for group in groups]
    records = []
    for case in sorted(mapping["cases"]):
        reviews = [group[case] for group in by_case]
        def comparable(review: dict[str, Any]) -> dict[str, Any]:
            # Reviewer identity and free-text notes never decide agreement.
            return {key: value for key, value in review.items() if key not in {"reviewer_id", "notes"}}
        if any(canonical(comparable(review)) != canonical(comparable(reviews[0])) for review in reviews[1:]):
            status = "REQUIRES_ADJUDICATION"; reasons = ["HUMAN_REVIEW_DISAGREEMENT"]
        else:
            review, identity = reviews[0], mapping["cases"][case]["variants"]
            inverse = {kind: blind for blind, kind in identity.items()}
            pattern = (review["variant_assessments"][inverse["ORIGINAL"]] == "SUPPORTED" and review["variant_assessments"][inverse["KEEP_TARGET"]] == "SUPPORTED" and review["variant_assessments"][inverse["DROP_TARGET"]] == "INSUFFICIENT" and review["variant_assessments"][inverse["DROP_MATCHED_CONTROL"]] == "SUPPORTED")
            pixel = all(a["pixel_audit_pass"] for entries in mapping["cases"][case]["pixel_audits"].values() for a in entries)
            meta = review["claim_wording"] == "CLEAR" and review["temporal_scope"] == "VALID" and review["observability"] == "DIRECTLY_VISIBLE" and review["eligibility_label"] == "SINGLE_ROI_ELIGIBLE"
            status = ELIGIBLE if pattern and pixel and meta and mapping["cases"][case]["matched_control_available"] else "INELIGIBLE_SINGLE_ROI"
            reasons = [] if status == ELIGIBLE else ["HUMAN_ELIGIBILITY_GATE_NOT_MET"]
        records.append({"blind_case_id": case, "audit_case_id": mapping["cases"][case]["audit_case_id"], "eligibility_status": status, "reason_codes": reasons, "diagnostic_only": True, "certificate_unchanged": True, "new_verified_count": 0, "gt_used": False})
    overall = "REQUIRES_ADJUDICATION" if any(r["eligibility_status"] == "REQUIRES_ADJUDICATION" for r in records) else "PASS"
    (output_dir / "eligibility_report.jsonl").write_text("".join(canonical(row) + "\n" for row in records))
    summary = {"format": FORMAT, "status": overall, "records": len(records), "eligible_count": sum(r["eligibility_status"] == ELIGIBLE for r in records), "model_calls_made": 0, "backend_loaded": False, "certificate_created": False, "new_verified_count": 0, "gt_used": False, "historical_artifacts_unchanged": True}
    (output_dir / "eligibility_summary.json").write_text(canonical(summary) + "\n")
    (output_dir / "phase4a0_audit.json").write_text(canonical(summary) + "\n")
    return summary
