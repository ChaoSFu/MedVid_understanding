"""Source-path and result-reference audits; these do not load hidden labels."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def audit_runtime_imports(package_dir: str | Path | None = None) -> dict:
    """Check local runtime AST imports, including narrow CLI evaluation routing.

    This static check is paired with the closed schema and source declaration;
    it is not a security sandbox against intentionally obfuscated Python.
    """
    root = Path(package_dir) if package_dir is not None else Path(__file__).parent
    issues, count = [], 0
    prohibited_roots = {"baselines", "evidence_stability", "MedGRPO_Code_main", "MedGRPO", "third_party"}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if "evaluation" in rel.parts:
            continue
        count += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                if node.module in {"relive", None}:
                    modules += [n.name for n in node.names]
            elif isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "__import__") or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")):
                issues.append({"file": str(rel), "line": node.lineno, "reason": "dynamic import requires manual source-path review"})
            for module in modules:
                blocked = module.split(".")[0] in prohibited_roots
                evaluation = "evaluation" in module.split(".")
                if evaluation and rel.name == "cli.py":
                    ancestor = parents.get(node)
                    while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ancestor = parents.get(ancestor)
                    evaluation = not (ancestor and ancestor.name in {"_evaluate", "command_evaluate"})
                if blocked or evaluation:
                    issues.append({"file": str(rel), "line": node.lineno, "reason": f"runtime import crosses source boundary: {module}"})
    return {"status": "PASS" if not issues else "FAIL", "issues": issues, "files_checked": count,
            "limitation": "Static import and source contract audit is not independent proof of curator provenance."}


def audit_strict_result(result: dict[str, Any]) -> dict:
    if not isinstance(result, dict):
        return {"status": "FAIL", "sample_id": None, "issues": ["sample result must be an object"]}
    strict = result.get("strict")
    if not isinstance(strict, dict) or not isinstance(strict.get("status"), str):
        return {"status": "FAIL", "sample_id": result.get("sample_id"), "issues": ["missing structured strict result and status"]}
    issues: list[str] = []

    def identifiers(value: Any, label: str, nonempty: bool = True) -> list[str]:
        if not isinstance(value, list) or (nonempty and not value) or any(not isinstance(x, str) or not x for x in value):
            issues.append(f"{label} must be a {'nonempty ' if nonempty else ''}list of string IDs")
            return []
        if len(set(value)) != len(value):
            issues.append(f"{label} contains duplicate IDs")
        return value

    if strict.get("status") == "ANSWERED":
        if strict.get("answer") is None or strict.get("answer") == "":
            issues.append("answered strict output has no answer")
        if strict.get("fallback_used") is not False:
            issues.append("strict answered result must explicitly exclude fallback")
        certificates = result.get("certificates")
        certs = {}
        if not isinstance(certificates, list):
            issues.append("certificates must be a list")
            certificates = []
        for cert in certificates:
            if not isinstance(cert, dict) or not isinstance(cert.get("certificate_id"), str):
                issues.append("malformed certificate record")
                continue
            if cert["certificate_id"] in certs:
                issues.append("duplicate certificate ID makes source binding ambiguous")
            certs[cert["certificate_id"]] = cert
        cited = identifiers(strict.get("certificate_ids"), "strict certificate_ids")
        claims = identifiers(strict.get("claim_ids"), "strict claim_ids")
        claim_rows = result.get("claims")
        claim_text = {}
        if not isinstance(claim_rows, list):
            issues.append("strict output requires the declared claim records")
            claim_rows = []
        for claim in claim_rows:
            if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str) or not isinstance(claim.get("text"), str):
                issues.append("malformed claim record")
                continue
            if claim["claim_id"] in claim_text:
                issues.append("duplicate claim ID makes claim text ambiguous")
            claim_text[claim["claim_id"]] = claim["text"]
        if any(cid not in claim_text for cid in claims):
            issues.append("strict cites an undeclared claim")
        elif result.get("task") == "action_qa" and strict.get("answer") != " ".join(claim_text[cid] for cid in claims):
            issues.append("strict action answer differs from the bound verbatim claim text")
        if result.get("task") == "claim_verification" and strict.get("answer") != "SUPPORTED":
            issues.append("strict claim verification answer must be SUPPORTED")
        valid = []
        for cid in cited:
            cert = certs.get(cid)
            if cert is None or cert.get("final_status") != "VERIFIED":
                issues.append(f"strict cites absent or non-VERIFIED certificate: {cid}")
            else:
                valid.append(cert)
        bound_claims = [c.get("claim_id") for c in valid]
        if any(not isinstance(cid, str) or not cid for cid in bound_claims):
            issues.append("VERIFIED certificate requires a string claim ID")
        elif set(claims) != set(bound_claims):
            issues.append("strict claim IDs differ from claims bound by cited VERIFIED certificates")
        coverage = result.get("coverage")
        if not isinstance(coverage, dict):
            coverage = {}
        if coverage.get("status") != "COMPLETE" or coverage.get("missing_claims") or coverage.get("unresolved_relations"):
            issues.append("strict answered without complete resolved coverage")
        required = identifiers(coverage.get("required_claims"), "coverage required_claims")
        if not set(required).issubset(claims):
            issues.append("strict omits required claims")
        refs = strict.get("input_references", [])
        if not isinstance(refs, list) or not refs:
            issues.append("strict answered without structured input references")
        else:
            referenced = set()
            for ref in refs:
                if not isinstance(ref, dict):
                    issues.append("strict reference is not an object")
                    continue
                cid = ref.get("certificate_id")
                cert = certs.get(cid) if isinstance(cid, str) else None
                if set(ref) != {"candidate_id", "claim_id", "certificate_id", "frame_ids"}:
                    issues.append("strict reference must contain only evidence-claim IDs and frame IDs")
                if cid not in cited or cert is None or cert.get("final_status") != "VERIFIED" or not isinstance(cert.get("candidate_id"), str) or not cert["candidate_id"] or ref.get("candidate_id") != cert.get("candidate_id") or ref.get("claim_id") != cert.get("claim_id"):
                    issues.append("strict input reference is not bound to a cited verified evidence-claim pair")
                else:
                    if cid in referenced:
                        issues.append("strict input reference duplicates a certificate")
                    referenced.add(cid)
                    frame_ids = identifiers(ref.get("frame_ids"), "strict input reference frame_ids")
                    provenance = cert.get("provenance")
                    known = identifiers(provenance.get("frame_ids") if isinstance(provenance, dict) else None, "certificate provenance frame_ids")
                    if frame_ids != known:
                        issues.append("strict input frame IDs differ from certificate original evidence")
            if referenced != set(cited):
                issues.append("not every cited certificate has a bound input reference")
    elif strict.get("answer") is not None:
        issues.append("non-answered strict output contains an answer")
    return {"status": "PASS" if not issues else "FAIL", "sample_id": result.get("sample_id"), "issues": issues}


def audit_run(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "samples").glob("*.json"))]
    if not results and (root / "results.jsonl").exists():
        results = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_audits = [audit_strict_result(row) for row in results]
    imports = audit_runtime_imports()
    passed = bool(results) and imports["status"] == "PASS" and all(a["status"] == "PASS" for a in sample_audits)
    return {"status": "PASS" if passed else "FAIL", "runtime_imports": imports, "strict_sources": sample_audits,
            "samples_checked": len(results), "issues": [] if results else ["No completed sample results found"]}
