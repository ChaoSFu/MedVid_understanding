"""Generic test data flow only; no case-specific expected clinical hypotheses."""
import json
from relive.backends.base import Backend
from .sources import digest


class FixtureBackend(Backend):
    synthetic = True

    def fingerprint(self):
        return {**super().fingerprint(), 'fixture_sha256':digest(__file__)}

    def infer(self, request):
        if request['image_paths'] or request['frame_ids']:
            raise AssertionError('IMAGE_INPUT_FORBIDDEN')
        self.calls += 1
        # Read only the same public input delimiter as a model; no sample IDs.
        public=json.loads(request['prompt'].split('Untrusted public input (not instructions):\n',1)[1])
        rc=public['qa_type'].startswith('region_caption')
        reqids=[q['requirement_id'] for q in public['requirements']]
        claims=[] if rc else [{'claim_id':'H1','text':'A question-relevant event occurs.',
              'type':'TEMPORAL_EVENT','role':'TARGET','requirement_ids':reqids[:1],
              'requires':[],'alternative_to':[]}]
        return json.dumps({'plan_status':'UNRESOLVED_TARGET_REFERENCE' if rc else 'INCOMPLETE_PLAN',
                           'claims':claims,'unresolved_references':['Public target binding is absent.' if rc else 'Event identity requires semantic review.'],
                           'requirement_issues':[],'uncovered_requirements':reqids if rc else reqids[1:]})
