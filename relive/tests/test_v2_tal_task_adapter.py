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
  content["events"]["secure_the_site"]={**content["events"]["secure_the_base"], "aliases":["secure the base"]}
  with tempfile.TemporaryDirectory() as directory:
   path=Path(directory)/"ontology.yaml";path.write_text(json.dumps(content));ontology=load_event_ontology(path)
   self.assertEqual(TemporalLocalizationTaskAdapter().build_requirement(self.q("When does secure the base happen?"),ontology).reason_code,"AMBIGUOUS_EVENT_MATCH")
if __name__=="__main__":unittest.main()

class TALTerminalQueryCompatibilityTests(unittest.TestCase):
 def setUp(self):
  self.ontology=load_event_ontology(Path(__file__).resolve().parents[1]/"configs/v2/tal_event_ontology.yaml")
  self.adapter=TemporalLocalizationTaskAdapter()
 def q(self,text):
  return PublicTaskQuestion(1,"sample",hashlib.sha256(b"public").hexdigest(),text,hashlib.sha256(text.encode()).hexdigest(),"tal","dataset")
 def test_medvidu_style_prefix_only_uses_terminal_query(self):
  question=("Procedure description: Secure the base may be one possible action. "
            "Possible actions: Secure the base; irrigate the field.\n"
            "Question: When does securing the base happen?")
  result=self.adapter.build_requirement(self.q(question),self.ontology)
  self.assertEqual(result.status,RequirementBuildStatus.FROZEN)
  self.assertEqual(result.template_id,"when_does_happen")
  self.assertEqual(result.normalized_event_phrase,"securing the base")
  self.assertEqual(result.requirement.question_text,question)
  self.assertEqual(result.requirement.question_sha256,hashlib.sha256(question.encode()).hexdigest())
 def test_possible_actions_do_not_override_terminal_target(self):
  question="Possible actions include Secure the base. Find the segment(s) where irrigate the field happens."
  result=self.adapter.build_requirement(self.q(question),self.ontology)
  self.assertEqual(result.status,RequirementBuildStatus.UNRESOLVED)
  self.assertEqual(result.reason_code,"UNRESOLVED_EVENT")
 def test_registered_terminal_templates(self):
  templates=("When does Secure the base happen?", "Find the segment(s) where Secure the base happens.",
             "What is the time span of Secure the base?", "When can I see Secure the base in the video?",
             "When is Secure the base performed?", "During what interval does Secure the base occur?")
  for question in templates:
   with self.subTest(question=question):
    self.assertEqual(self.adapter.build_requirement(self.q(question),self.ontology).status,RequirementBuildStatus.FROZEN)
 def test_unknown_ambiguous_and_multiple_queries_fail_closed(self):
  self.assertEqual(self.adapter.build_requirement(self.q("When does unknown action happen?"),self.ontology).reason_code,"UNRESOLVED_EVENT")
  self.assertEqual(self.adapter.build_requirement(self.q("When does Secure the base happen? When is Secure the base performed?"),self.ontology).reason_code,"UNSUPPORTED_OR_AMBIGUOUS_TERMINAL_QUERY")
  first=self.ontology.events[0]
  ambiguous=EventOntology(self.ontology.ontology_version,self.ontology.ontology_sha256,(first,EventDefinition("secure_the_base_again",first.aliases,first.answer_schema,first.required_evidence,first.temporal_requirement,first.spatial_requirement,first.ontology_source)))
  self.assertEqual(self.adapter.build_requirement(self.q("When does Secure the base happen?"),ambiguous).reason_code,"AMBIGUOUS_EVENT_MATCH")
