from __future__ import annotations
import hashlib,inspect,tempfile,unittest
from pathlib import Path
from relive.v2.task_selection import PublicTaskQuestion
from relive.v2.temporal_localization import *
class TALAdapterTests(unittest.TestCase):
 def setUp(self):self.ontology=load_event_ontology(Path(__file__).resolve().parents[1]/"configs/v2/tal_event_ontology.yaml")
 def q(self,text,qa="tal"):return PublicTaskQuestion(1,"sample",hashlib.sha256(b"public").hexdigest(),text,hashlib.sha256(text.encode()).hexdigest(),qa,"dataset")
 def test_closed_template_maps_deterministically(self):
  adapter=TemporalLocalizationTaskAdapter();result=adapter.build_requirement(self.q("  WHEN does   Secure the base happen?!  "),self.ontology)
  self.assertEqual(result.status,RequirementBuildStatus.FROZEN);req=result.requirement;self.assertEqual(req.target_event,"secure_the_base");self.assertEqual(req.answer_schema.to_canonical_dict(),{"type":"INTERVAL","timebase":"REQUIRES_VIDEO_INDEX","cardinality":"ONE_OR_MORE"});self.assertEqual(req.temporal_requirement.value,"EVENT_INTERVAL");self.assertEqual(req.spatial_requirement.value,"LOCAL_DYNAMIC_RELATION")
 def test_unresolved_unsupported_and_no_visual_parameters(self):
  adapter=TemporalLocalizationTaskAdapter();self.assertEqual(adapter.build_requirement(self.q("When does base happen?"),self.ontology).reason_code,"UNRESOLVED_EVENT");self.assertEqual(adapter.build_requirement(self.q("When does Secure the base happen?","stg"),self.ontology).reason_code,"UNSUPPORTED_TASK_TYPE")
  signature=inspect.signature(adapter.build_requirement);self.assertEqual(tuple(signature.parameters),("public_question","ontology"));self.assertFalse(any(param.kind is param.VAR_KEYWORD for param in signature.parameters.values()))
 def test_word_boundary_and_ambiguous_aliases(self):
  self.assertEqual(TemporalLocalizationTaskAdapter().build_requirement(self.q("When does securing the baseless happen?"),self.ontology).reason_code,"UNRESOLVED_EVENT")
  import json
  content=json.loads((Path(__file__).resolve().parents[1]/"configs/v2/tal_event_ontology.yaml").read_text())
  content["events"]["secure_the_site"]={**content["events"]["secure_the_base"], "aliases":["the base"]}
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/"ontology.yaml";path.write_text(json.dumps(content));ontology=load_event_ontology(path)
   self.assertEqual(TemporalLocalizationTaskAdapter().build_requirement(self.q("When does secure the base happen?"),ontology).reason_code,"AMBIGUOUS_EVENT_MATCH")
if __name__=="__main__":unittest.main()
