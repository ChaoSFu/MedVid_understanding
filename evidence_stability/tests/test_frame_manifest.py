import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from evidence_stability.frame_manifest import (
    absolute_from_relative,
    build_frame_manifest,
    coverage,
    load_smoke_ids_artifact,
    normalize_manifest_relative_path,
    relative_frame_path,
    selected_interventions,
)
from evidence_stability.utils import write_jsonl


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "09_prepare_d2_frame_manifest.py"
SPEC = importlib.util.spec_from_file_location("frame_manifest_script", SCRIPT_PATH)
frame_manifest_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(frame_manifest_script)


def make_intervention(
    idx: int,
    root: Path,
    *,
    generation_valid: bool = True,
    intervention_type: str = "SHIFT_LEFT_2",
    dataset_name: str = "AVOS",
    frame_names: list[str] | None = None,
    declared_n_frames: int | None = None,
) -> dict:
    frame_names = frame_names or [f"{dataset_name}/video_{idx}/000{i}.jpg" for i in range(3)]
    frames = [str(root / name) for name in frame_names]
    return {
        "qa_id": f"qa-{idx}",
        "clip_id": f"clip-{idx}",
        "window_id": f"win-{idx}",
        "intervention_id": f"iv-{idx}-{intervention_type}",
        "dataset_name": dataset_name,
        "target_field": "action",
        "intervention_family": intervention_type.split("_", 1)[0],
        "intervention_type": intervention_type,
        "intervened_frame_paths": frames,
        "intervened_n_frames": declared_n_frames if declared_n_frames is not None else len(frames),
        "generation_valid": generation_valid,
        "strict_valid": True,
        "original_candidate_label": "TRUE_SUPPORT",
        "original_gt_alignment_class": "STRONG_GT_ALIGNED",
    }


def write_file(path: Path, size: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class FrameManifestTests(unittest.TestCase):
    def test_exact_smoke_id_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [make_intervention(1, root), make_intervention(2, root), make_intervention(3, root)]
            selected, _ = selected_interventions(rows, scope="smoke", smoke_ids=[rows[2]["intervention_id"], rows[0]["intervention_id"]])
            self.assertEqual([row["intervention_id"] for row in selected], [rows[2]["intervention_id"], rows[0]["intervention_id"]])

    def test_full_generation_valid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [make_intervention(1, root), make_intervention(2, root, generation_valid=False)]
            selected, _ = selected_interventions(rows, scope="full")
            self.assertEqual([row["intervention_id"] for row in selected], [rows[0]["intervention_id"]])

    def test_unique_frame_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            names = ["AVOS/v/0001.jpg", "AVOS/v/0002.jpg", "AVOS/v/0002.jpg"]
            for name in set(names):
                write_file(source / name)
            manifest = build_frame_manifest(
                [make_intervention(1, source, frame_names=names)],
                scope="full",
                source_frame_root=source,
                destination_frame_root=dest,
            )
            self.assertEqual(manifest["n_total_frame_references"], 3)
            self.assertEqual(manifest["n_unique_required_frames"], 2)
            self.assertEqual(manifest["duplicate_unique_path_count"], 0)

    def test_relative_path_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "valdata"
            frame = root / "AVOS" / "foo" / "bar.jpg"
            self.assertEqual(relative_frame_path(frame, root), "AVOS/foo/bar.jpg")

    def test_nested_directories_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "valdata"
            rel = relative_frame_path(root / "CholecT50" / "VID01" / "000001.png", root)
            absolute = absolute_from_relative(root, rel)
            self.assertEqual(rel, "CholecT50/VID01/000001.png")
            self.assertEqual(absolute, root / "CholecT50" / "VID01" / "000001.png")

    def test_source_missing_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            manifest = build_frame_manifest(
                [make_intervention(1, source, frame_names=["AVOS/v/missing.jpg"])],
                scope="full",
                source_frame_root=source,
                destination_frame_root=dest,
            )
            self.assertEqual(manifest["n_missing_at_source"], 1)
            self.assertEqual(manifest["source_missing_frames"], ["AVOS/v/missing.jpg"])

    def test_destination_missing_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            write_file(source / "AVOS/v/0001.jpg")
            manifest = build_frame_manifest(
                [make_intervention(1, source, frame_names=["AVOS/v/0001.jpg"])],
                scope="full",
                source_frame_root=source,
                destination_frame_root=dest,
            )
            self.assertEqual(manifest["n_missing_at_destination"], 1)
            self.assertEqual(manifest["destination_missing_frames"], ["AVOS/v/0001.jpg"])

    def test_destination_existing_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            write_file(source / "AVOS/v/0001.jpg")
            write_file(dest / "AVOS/v/0001.jpg")
            manifest = build_frame_manifest(
                [make_intervention(1, source, frame_names=["AVOS/v/0001.jpg"])],
                scope="full",
                source_frame_root=source,
                destination_frame_root=dest,
            )
            self.assertEqual(manifest["n_existing_at_destination"], 1)
            self.assertEqual(manifest["destination_existing_frames"], ["AVOS/v/0001.jpg"])

    def test_coverage_calculation(self):
        self.assertEqual(coverage(0, 0), 1.0)
        self.assertEqual(coverage(1, 4), 0.25)

    def test_file_size_mismatch_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            write_file(source / "AVOS/v/0001.jpg", size=3)
            write_file(dest / "AVOS/v/0001.jpg", size=5)
            manifest = build_frame_manifest(
                [make_intervention(1, source, frame_names=["AVOS/v/0001.jpg"])],
                scope="full",
                source_frame_root=source,
                destination_frame_root=dest,
            )
            self.assertEqual(manifest["n_size_mismatch"], 1)
            self.assertTrue(manifest["required_frames"][0]["size_mismatch"])

    def test_frame_count_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            row = make_intervention(1, source, declared_n_frames=99)
            with self.assertRaisesRegex(RuntimeError, "Frame count mismatch"):
                build_frame_manifest([row], scope="full", source_frame_root=source, destination_frame_root=source)

    def test_duplicate_intervention_id_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            row = make_intervention(1, source)
            with self.assertRaisesRegex(RuntimeError, "Duplicate intervention_id"):
                selected_interventions([row, dict(row)], scope="full")

    def test_path_outside_root_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            outside = Path(tmp) / "outside"
            row = make_intervention(1, source)
            row["intervened_frame_paths"] = [str(outside / "AVOS" / "x.jpg")]
            row["intervened_n_frames"] = 1
            with self.assertRaisesRegex(ValueError, "PATH_OUTSIDE_SOURCE_ROOT"):
                build_frame_manifest([row], scope="full", source_frame_root=source, destination_frame_root=source)

    def test_traversal_rejection(self):
        with self.assertRaisesRegex(ValueError, "Unsafe relative"):
            normalize_manifest_relative_path("../AVOS/x.jpg")
        with self.assertRaisesRegex(ValueError, "absolute"):
            normalize_manifest_relative_path("/AVOS/x.jpg")

    def test_deterministic_sorted_txt_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            rows = [
                make_intervention(1, source, frame_names=["AVOS/v/b.jpg", "AVOS/v/a.jpg"]),
            ]
            for name in ["AVOS/v/a.jpg", "AVOS/v/b.jpg"]:
                write_file(source / name)
            manifest = build_frame_manifest(rows, scope="full", source_frame_root=source, destination_frame_root=dest)
            files = frame_manifest_script.write_manifest_outputs(manifest, Path(tmp) / "out")
            self.assertEqual(files["required_txt"].read_text(encoding="utf-8").splitlines(), ["AVOS/v/a.jpg", "AVOS/v/b.jpg"])

    def test_rsync_files_from_contains_only_missing_destination_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            for name in ["AVOS/v/a.jpg", "AVOS/v/b.jpg"]:
                write_file(source / name)
            write_file(dest / "AVOS/v/a.jpg")
            manifest = build_frame_manifest(
                [make_intervention(1, source, frame_names=["AVOS/v/a.jpg", "AVOS/v/b.jpg"])],
                scope="full",
                source_frame_root=source,
                destination_frame_root=dest,
            )
            files = frame_manifest_script.write_manifest_outputs(manifest, Path(tmp) / "out")
            self.assertEqual(files["rsync_files_from"].read_text(encoding="utf-8").splitlines(), ["AVOS/v/b.jpg"])

    def test_smoke_readiness_true_when_all_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            row = make_intervention(1, source, frame_names=["AVOS/v/a.jpg"])
            write_file(source / "AVOS/v/a.jpg")
            write_file(dest / "AVOS/v/a.jpg")
            manifest = build_frame_manifest(
                [row],
                scope="smoke",
                smoke_ids=[row["intervention_id"]],
                source_frame_root=source,
                destination_frame_root=dest,
            )
            self.assertTrue(manifest["smoke_ready_for_inference"])

    def test_smoke_readiness_false_when_one_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            row = make_intervention(1, source, frame_names=["AVOS/v/a.jpg"])
            write_file(source / "AVOS/v/a.jpg")
            manifest = build_frame_manifest(
                [row],
                scope="smoke",
                smoke_ids=[row["intervention_id"]],
                source_frame_root=source,
                destination_frame_root=dest,
            )
            self.assertFalse(manifest["smoke_ready_for_inference"])

    def test_full_intervention_availability_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            rows = [
                make_intervention(1, source, frame_names=["AVOS/v/a.jpg"]),
                make_intervention(2, source, frame_names=["AVOS/v/b.jpg"]),
            ]
            for name in ["AVOS/v/a.jpg", "AVOS/v/b.jpg"]:
                write_file(source / name)
            write_file(dest / "AVOS/v/a.jpg")
            manifest = build_frame_manifest(rows, scope="full", source_frame_root=source, destination_frame_root=dest)
            self.assertEqual(manifest["n_interventions_fully_available"], 1)
            self.assertEqual(manifest["n_interventions_blocked_by_missing_frames"], 1)

    def test_no_gt_fields_in_output_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            dest = Path(tmp) / "dest"
            row = make_intervention(1, source, frame_names=["AVOS/v/a.jpg"])
            write_file(source / "AVOS/v/a.jpg")
            manifest = build_frame_manifest([row], scope="full", source_frame_root=source, destination_frame_root=dest)
            text = json.dumps(manifest, sort_keys=True)
            forbidden = ["TRUE_SUPPORT", "strict_valid", "candidate_label", "gt_alignment", "evidence_density", "gt_recall", "n_gt_visible"]
            for token in forbidden:
                self.assertNotIn(token, text)

    def test_load_smoke_ids_from_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase_d2_smoke_intervention_ids.json"
            path.write_text(json.dumps({"interventions": [{"intervention_id": "b"}, {"intervention_id": "a"}]}), encoding="utf-8")
            self.assertEqual(load_smoke_ids_artifact(path), ["b", "a"])


if __name__ == "__main__":
    unittest.main()
