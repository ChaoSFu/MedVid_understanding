from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from relive.storage.artifacts import canonical_json,stable_hash
from relive.v2.requirement_freeze import *
from relive.v2.task_selection import *
from relive.v2.temporal_localization import load_event_ontology
class RequirementFreezeTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.question="When does Secure the base happen?";self.row={"source_record_index":0,"sample_id":"sample","public_record_sha256":hashlib.sha256(b"public").hexdigest(),"question_sha256":hashlib.sha256(self.question.encode()).hexdigest(),"qa_type":"tal","question":self.question,"frame_count":9,"first_verified_frame_path":"/does/not/exist.jpg","last_verified_frame_path":"/still/not/exist.jpg","dataset_name":"public"};self.selector=self.root/"selector.jsonl";self.selector.write_text(canonical_json(self.row)+"\n");self.selection=freeze_selector_order(selector_path=self.selector,max_samples=1);self.selection_path=self.root/"selection.json";write_selection(self.selection,self.selection_path);self.ontology=load_event_ontology(Path(__file__).resolve().parents[1]/"configs/v2/tal_event_ontology.yaml")
 def tearDown(self):self.tmp.cleanup()
 def freeze(self,out):return freeze_tal_requirements(selection=self.selection,selector_sha256=hashlib.sha256(self.selector.read_bytes()).hexdigest(),selection_manifest_sha256=hashlib.sha256(self.selection_path.read_bytes()).hexdigest(),ontology=self.ontology,output_dir=out)
 def test_freeze_is_deterministic_and_prevideo(self):
  one,two=self.root/"one",self.root/"two";audit=self.freeze(one);again=self.freeze(two)
  manifest=json.loads((one/"v2_requirement_manifest.json").read_text());self.assertEqual(audit["frames_read"],0);self.assertEqual(manifest["requirement_specs_sha256"],json.loads((two/"v2_requirement_manifest.json").read_text())["requirement_specs_sha256"]);self.assertEqual(manifest["manifest_content_sha256"],stable_hash({key:value for key,value in manifest.items() if key!="manifest_content_sha256"}));self.assertNotIn("VERIFIED",(one/"v2_requirement_specs.jsonl").read_text())
 def test_output_tampering_and_nonempty_output_fail(self):
  out=self.root/"out";self.freeze(out);self.assertEqual(validate_requirement_freeze_artifacts(out)["status"],"PASS");(out/"v2_requirement_specs.jsonl").write_text("tampered")
  with self.assertRaisesRegex(RequirementFreezeError,"BYTES_CHANGED"):validate_requirement_freeze_artifacts(out)
  with self.assertRaisesRegex(RequirementFreezeError,"OUTPUT_DIRECTORY"):self.freeze(out)
 def test_ontology_mutation_changes_requirement_identity(self):
  original=self.freeze(self.root/"orig");mutated=self.root/"ontology.yaml";mutated.write_text((Path(__file__).resolve().parents[1]/"configs/v2/tal_event_ontology.yaml").read_text().replace("HUMAN_AUTHORED_TASK_SEMANTICS","HUMAN_AUTHORED_TASK_SEMANTICS_V2"));other=load_event_ontology(mutated);audit=freeze_tal_requirements(selection=self.selection,selector_sha256=hashlib.sha256(self.selector.read_bytes()).hexdigest(),selection_manifest_sha256=hashlib.sha256(self.selection_path.read_bytes()).hexdigest(),ontology=other,output_dir=self.root/"mutated");self.assertNotEqual(json.loads((self.root/"orig/v2_requirement_manifest.json").read_text())["requirement_specs_sha256"],json.loads((self.root/"mutated/v2_requirement_manifest.json").read_text())["requirement_specs_sha256"])
if __name__=="__main__":unittest.main()

class RequirementFreezeIsolationTests(unittest.TestCase):
 def test_static_sources_exclude_visual_and_certificate_imports(self):
  import ast
  root=Path(__file__).resolve().parents[1]/"src/relive/v2"; imported=[]
  for name in ("task_selection.py","temporal_localization.py","requirement_freeze.py"):
   for node in ast.walk(ast.parse((root/name).read_text())):
    if isinstance(node,ast.Import): imported.extend(alias.name for alias in node.names)
    elif isinstance(node,ast.ImportFrom) and node.module: imported.append(node.module)
  for forbidden in ("relive.backends","relive.runner","relive.verification","relive.certificate","relive.interventions","relive.evaluation","relive.data.medvidu"):
   self.assertFalse(any(name==forbidden or name.startswith(forbidden+".") for name in imported))
