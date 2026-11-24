"""Synthetic transport only; no expected clinical answers or case-ID branches."""
import json
from relive.backends.base import Backend


class MockAdapter(Backend):
    synthetic = True

    def infer(self, request):
        if request['stage'] != 'stage2_v41_generation' or request['image_paths'] or request['frame_ids']:
            raise AssertionError('ONLY_SINGLE_TEXT_GENERATION_ALLOWED')
        self.calls += 1
        row = json.loads(request['prompt'].split('Untrusted public input (not instructions):\n')[1])
        rc = row['qa_type'].startswith('region_caption'); summary = row['qa_type'].startswith('video_summary')
        claims = [] if rc or summary else [{
            'claim_id': 'H1', 'text': 'The queried tool is visible at time t.', 'type': 'OBJECT', 'role': 'TARGET',
            'requirement_ids': [row['requirements'][0]['requirement_id']], 'requires': [], 'alternative_to': []}]
        return json.dumps({'plan_status': 'UNRESOLVED_TARGET_REFERENCE' if rc else 'INCOMPLETE_PLAN', 'claims': claims,
                           'unresolved_slots': [{'slot_id': 'S1', 'purpose': 'TEST_ONLY: bind an unresolved observation target.', 'status': 'UNBOUND'}] if not claims else [],
                           'requirement_issues': []})
