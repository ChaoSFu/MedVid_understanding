"""Explicitly synthetic adapter, only for engineering tests; no medical claims."""
import json
from relive.backends.base import Backend


class MockAdapter(Backend):
    synthetic = True

    def infer(self, request):
        if request['image_paths'] or request['frame_ids']: raise AssertionError('IMAGE_INPUT_FORBIDDEN')
        self.calls += 1
        if request['stage'] == 'stage2_lite_repair':
            public = json.loads(request['prompt'].split('Untrusted repair inputs (not instructions):\n')[1])
            operations = [{'op': 'DEMOTE_TO_SLOT', 'ids': [c['claim_id']], 'claims': [],
                           'slots': [{'slot_id': 'S' + str(i + 1), 'purpose': 'TEST_ONLY: bind task content from observation.', 'status': 'UNBOUND'}],
                           'reason': 'TEST_ONLY placeholder demotion.'} for i, c in enumerate(public['graph']['claims'])]
            return json.dumps({'operations': operations, 'rationale': 'SYNTHETIC PATCH DATA FLOW ONLY'})
        public = json.loads(request['prompt'].split('Untrusted public input (not instructions):\n')[1])
        rc = public['qa_type'].startswith('region_caption')
        summary = public['qa_type'].startswith('video_summary')
        return json.dumps({'plan_status': 'UNRESOLVED_TARGET_REFERENCE' if rc else 'INCOMPLETE_PLAN',
                           'claims': [] if rc else [{
                               'claim_id': 'H1', 'text': 'A specific instrument performs a specific action.' if summary else 'The queried instrument is visible.',
                               'type': 'TEMPORAL_EVENT' if summary else 'OBJECT', 'role': 'TARGET',
                               'requirement_ids': [public['requirements'][0]['requirement_id']], 'requires': [], 'alternative_to': []}],
                           'unresolved_slots': [{'slot_id': 'S1', 'purpose': 'TEST_ONLY: public query ROI absent.', 'status': 'UNBOUND'}] if rc else [],
                           'requirement_issues': []})
