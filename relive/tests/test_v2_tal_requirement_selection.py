from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from relive.storage.artifacts import canonical_json
from relive.v2.task_selection import *
class TALSelectionTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def row(self,index,qa="tal",**more):
  row={"source_record_index":index,"sample_id":f"sample-{index}","public_record_sha256":hashlib.sha256(f"p{index}".encode()).hexdigest(),"question_sha256":hashlib.sha256(f"When does Secure the base happen? {index}".encode()).hexdigest(),"qa_type":qa,"question":f"When does Secure the base happen? {index}","frame_count":1,"first_verified_frame_path":"/does/not/exist/a.jpg","last_verified_frame_path":"/does/not/exist/b.jpg","dataset_name":"public"};row.update(more);return row
 def selector(self,rows):
  path=self.root/"selector.jsonl";path.write_text("".join(canonical_json(row)+"\n" for row in rows));return path
 def test_freezes_tal_in_public_selector_order_without_opening_paths(self):
  path=self.selector([self.row(0,"other"),self.row(1),self.row(2)])
  selection=freeze_selector_order(selector_path=path,max_samples=2)
  self.assertEqual([item.source_record_index for item in selection.items],[1,2]);self.assertEqual(selection.selection_basis,"PUBLIC_SELECTOR_ORDER")
  out=self.root/"selection.json";write_selection(selection,out);loaded=load_selection(path=out,selector_path=path);self.assertEqual(loaded.items,selection.items)
 def test_hash_and_prohibited_input_fail_closed(self):
  row=self.row(0);row["question_sha256"]="0"*64
  with self.assertRaisesRegex(TALSelectionError,"QUESTION_SHA256"):read_public_question_selector(self.selector([row]))
  row=self.row(0);row["reference_answer"]="hidden"
  with self.assertRaisesRegex(TALSelectionError,"PROHIBITED"):read_public_question_selector(self.selector([row]))
 def test_insufficient_tal_is_not_retyped(self):
  with self.assertRaisesRegex(TALSelectionError,"INSUFFICIENT_TAL"):freeze_selector_order(selector_path=self.selector([self.row(0,"stg")]),max_samples=1)
if __name__=="__main__":unittest.main()

class TALSelectionBindingTests(unittest.TestCase):
 def test_selection_cannot_bind_to_changed_selector(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory); question="When does Secure the base happen?"; row={"source_record_index":0,"sample_id":"s","public_record_sha256":"a"*64,"question_sha256":hashlib.sha256(question.encode()).hexdigest(),"qa_type":"tal","question":question}
   selector=root/"selector.jsonl";selector.write_text(canonical_json(row)+"\n");selection=freeze_selector_order(selector_path=selector,max_samples=1);manifest=root/"selection.json";write_selection(selection,manifest)
   row["sample_id"]="changed";selector.write_text(canonical_json(row)+"\n")
   with self.assertRaisesRegex(TALSelectionError,"SOURCE_HASH"):load_selection(path=manifest,selector_path=selector)

class ExplicitIdentitySelectionTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def row(self,index,qa="tal"):
  question=f"When does Secure the base happen? {index}"
  return {"source_record_index":index,"sample_id":f"s-{index}","public_record_sha256":hashlib.sha256(f"p-{index}".encode()).hexdigest(),"question_sha256":hashlib.sha256(question.encode()).hexdigest(),"qa_type":qa,"question":question}
 def test_explicit_identity_preserves_input_order_and_binds_bytes(self):
  rows=[self.row(0),self.row(1),self.row(2)]
  selector=self.root/"selector.jsonl";selector.write_text("".join(canonical_json(row)+"\n" for row in rows))
  identity=self.root/"identity.jsonl";identity.write_text(canonical_json({key:rows[2][key] for key in IDENTITY_FIELDS})+"\n"+canonical_json({key:rows[0][key] for key in IDENTITY_FIELDS})+"\n")
  selection=freeze_explicit_public_identity(selector_path=selector,identity_manifest_path=identity)
  self.assertEqual([item.source_record_index for item in selection.items],[2,0])
  self.assertEqual(selection.identity_manifest_sha256,hashlib.sha256(identity.read_bytes()).hexdigest())
 def test_identity_schema_duplicate_unknown_and_invalid_utf8_fail_closed(self):
  row=self.row(0);selector=self.root/"selector.jsonl";selector.write_text(canonical_json(row)+"\n")
  identity=self.root/"identity.jsonl";good={key:row[key] for key in IDENTITY_FIELDS}
  identity.write_text(canonical_json(good)+"\n"+canonical_json(good)+"\n")
  with self.assertRaisesRegex(TALSelectionError,"DUPLICATE"):freeze_explicit_public_identity(selector_path=selector,identity_manifest_path=identity)
  identity.write_text(canonical_json({**good,"roi":[0,0,1,1]})+"\n")
  with self.assertRaisesRegex(TALSelectionError,"CLOSED_SCHEMA"):freeze_explicit_public_identity(selector_path=selector,identity_manifest_path=identity)
  identity.write_bytes(b'\xff')
  with self.assertRaisesRegex(TALSelectionError,"UNREADABLE"):freeze_explicit_public_identity(selector_path=selector,identity_manifest_path=identity)

class StrictSelectorEncodingTests(unittest.TestCase):
 def test_selector_invalid_utf8_duplicate_key_and_nonfinite_fail_closed(self):
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/"selector.jsonl"
   path.write_bytes(b'\xff')
   with self.assertRaisesRegex(TALSelectionError,"UNREADABLE"):read_public_question_selector(path)
   path.write_text('{"source_record_index":0,"source_record_index":1}\n')
   with self.assertRaisesRegex(TALSelectionError,"DUPLICATE_KEY"):read_public_question_selector(path)
   path.write_text('{"source_record_index":NaN}\n')
   with self.assertRaisesRegex(TALSelectionError,"NONFINITE"):read_public_question_selector(path)
