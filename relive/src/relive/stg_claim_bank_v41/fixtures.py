"""TEST_ONLY transport for batch binding, including empty upstream requirements."""
import json
from relive.stage2_single_pass_v41.fixtures import MockAdapter as V41Mock


class MockAdapter(V41Mock):
    def infer(self, request):
        public = json.loads(request['prompt'].split('Untrusted public input (not instructions):\n')[1])
        if public['requirements']: return super().infer(request)
        self.calls += 1
        return json.dumps({'plan_status': 'INCOMPLETE_PLAN', 'claims': [{
            'claim_id': 'H1', 'text': 'The queried tool is visible at time t.', 'type': 'OBJECT', 'role': 'TARGET',
            'requirement_ids': [], 'requires': [], 'alternative_to': []}], 'unresolved_slots': [], 'requirement_issues': []})
