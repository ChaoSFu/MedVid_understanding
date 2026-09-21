from __future__ import annotations
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from relive.storage.artifacts import canonical_json
from relive.v2.human_overlay_adjudication import HumanOverlayReviewError, adjudicate


def write_rows(path, rows):
    path.write_text(''.join(canonical_json(x)+'\n' for x in rows), encoding='utf-8')


def raw(anchor, role='BASE_PLATE', visibility='VISIBLE', *, observation_role='ACTION_CORE_PRESSING', frame='f'*64, box=None):
    box = [0.1,0.1,0.4,0.4] if box is None and visibility == 'VISIBLE' else box
    component = {'role':role, 'visibility':visibility, 'bbox_2d_raw':[100,100,400,400] if box else None,
                 'bbox_normalized_xyxy':box, 'bbox_pixel_xyxy':[10,10,40,40] if box else None}
    return {'anchor_candidate_id':anchor, 'spatial_grounding_task_id':'task-'+anchor,
            'observation_claim_id':'obs-'+anchor, 'observation_claim':'visible statement',
            'observation_role':observation_role, 'geometry_type':'DYNAMIC_SUPPORT_TUBE',
            'required_component_roles':[role], 'contextual_requirements':[], 'image_path':'/tmp/unused.png',
            'frame_sha256':frame, 'timestamp_seconds':1.0, 'source_frame_reference':1, 'components':[component]}


def review(anchor, role, visibility, label):
    return {'anchor_candidate_id':anchor, 'role':role, 'model_visibility':visibility,
            'human_overlay_label':label, 'reason_code':''}


class HumanOverlayAdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def adjudicate_rows(self, raws, reviews, expected=None, name='out'):
        rp=self.root/'raw.jsonl'; vp=self.root/'reviews.jsonl'; write_rows(rp,raws);write_rows(vp,reviews)
        return adjudicate(raw_anchor_manifest=rp, review_jsonl=vp, output_dir=self.root/name, expected_anchor_count=expected or len(raws))

    def test_all_adjudication_mappings(self):
        cases=[('VISIBLE','ACCEPTED'),('VISIBLE','BOX_TOO_BROAD'),('VISIBLE','BOX_TOO_TIGHT'),('VISIBLE','WRONG_OBJECT'),('VISIBLE','INTERFACE_NOT_LOCALIZED'),('VISIBLE','SHOULD_BE_NOT_VISIBLE'),('NOT_VISIBLE','NOT_VISIBLE_CONFIRMED'),('AMBIGUOUS','AMBIGUOUS_CONFIRMED'),('VISIBLE','UNRESOLVED')]
        raws=[raw(f'a{i}', visibility=v) for i,(v,_) in enumerate(cases)]
        reviews=[review(f'a{i}','BASE_PLATE',v,label) for i,(v,label) in enumerate(cases)]
        self.adjudicate_rows(raws,reviews)
        rows=[json.loads(x) for x in (self.root/'out'/'adjudicated_anchor_manifest.jsonl').read_text().splitlines()]
        by_label={x['adjudicated_components'][0]['human_overlay_label']:x['adjudicated_components'][0] for x in rows}
        self.assertTrue(by_label['ACCEPTED']['bbox_usable_for_intervention'])
        self.assertEqual(by_label['BOX_TOO_BROAD']['component_review_status'],'INVALID_GEOMETRY')
        self.assertEqual(by_label['WRONG_OBJECT']['component_review_status'],'INVALID_OBJECT')
        self.assertEqual(by_label['INTERFACE_NOT_LOCALIZED']['component_review_status'],'INVALID_INTERFACE')
        self.assertEqual(by_label['SHOULD_BE_NOT_VISIBLE']['adjudicated_visibility'],'NOT_VISIBLE')
        self.assertEqual(by_label['NOT_VISIBLE_CONFIRMED']['component_review_status'],'VALID_NONVISIBLE')
        self.assertEqual(by_label['AMBIGUOUS_CONFIRMED']['component_review_status'],'VALID_AMBIGUOUS')
        self.assertEqual(by_label['UNRESOLVED']['component_review_status'],'UNRESOLVED')

    def test_missing_duplicate_unknown_and_incompatible_reviews_fail_closed(self):
        rows=[raw('a')]
        for bad in ([], [review('a','BASE_PLATE','VISIBLE','ACCEPTED'),review('a','BASE_PLATE','VISIBLE','ACCEPTED')], [review('unknown','BASE_PLATE','VISIBLE','ACCEPTED')], [review('a','BASE_PLATE','VISIBLE','NOT_VISIBLE_CONFIRMED')]):
            with self.assertRaises(HumanOverlayReviewError): self.adjudicate_rows(rows,bad,name='bad'+str(len(bad))+str(len(str(bad))))

    def test_raw_role_omission_and_coordinate_abnormality_fail_closed(self):
        omitted=raw('a'); omitted['required_component_roles']=['MISSING_ROLE']
        with self.assertRaisesRegex(HumanOverlayReviewError,'OMISSION'):
            self.adjudicate_rows([omitted],[review('a','BASE_PLATE','VISIBLE','ACCEPTED')])
        abnormal=raw('b'); abnormal['components'][0]['bbox_normalized_xyxy']=[.2,.1,.1,.4]
        with self.assertRaisesRegex(HumanOverlayReviewError,'BBOX'):
            self.adjudicate_rows([abnormal],[review('b','BASE_PLATE','VISIBLE','ACCEPTED')],name='abnormal')

    def test_required_context_separation_and_routing_priority(self):
        item=raw('a',role='BASE_PLATE'); item['required_component_roles']=['BASE_PLATE']; item['contextual_requirements']=['BASE_REMAINS_ALIGNED_OR_ATTACHED_AFTER_INTERACTION']
        item['components'].append({'role':'BASE_REMAINS_ALIGNED_OR_ATTACHED_AFTER_INTERACTION','visibility':'VISIBLE','bbox_2d_raw':[1,1,2,2],'bbox_normalized_xyxy':[.1,.1,.2,.2],'bbox_pixel_xyxy':[1,1,2,2]})
        self.adjudicate_rows([item],[review('a','BASE_PLATE','VISIBLE','ACCEPTED'),review('a','BASE_REMAINS_ALIGNED_OR_ATTACHED_AFTER_INTERACTION','VISIBLE','UNRESOLVED')])
        row=json.loads((self.root/'out'/'adjudicated_anchor_manifest.jsonl').read_text().splitlines()[0])
        self.assertEqual(row['spatial_anchor_status'],'SPATIAL_COMPONENTS_VALID')
        self.assertEqual(row['claim_evidence_route'],'TEMPORAL_REACQUIRE')
        relation=raw('b', observation_role='POSTCONDITION_ATTACHMENT')
        self.adjudicate_rows([relation],[review('b','BASE_PLATE','VISIBLE','ACCEPTED')],name='relation')
        row=json.loads((self.root/'relation'/'adjudicated_anchor_manifest.jsonl').read_text().splitlines()[0])
        self.assertEqual(row['claim_evidence_route'],'RELATIONAL_COMPOSITE_REQUIRED')

    def test_observation_selection_determinism_and_raw_unchanged(self):
        rows=[raw('b'),raw('a')]; rows[0]['observation_claim_id']=rows[1]['observation_claim_id']='shared'
        reviews=[review('b','BASE_PLATE','VISIBLE','BOX_TOO_TIGHT'),review('a','BASE_PLATE','VISIBLE','ACCEPTED')]
        rp=self.root/'raw.jsonl'; vp=self.root/'review.jsonl';write_rows(rp,rows);write_rows(vp,reviews); original=rp.read_bytes()
        first=adjudicate(raw_anchor_manifest=rp,review_jsonl=vp,output_dir=self.root/'one',expected_anchor_count=2)
        second=adjudicate(raw_anchor_manifest=rp,review_jsonl=vp,output_dir=self.root/'two',expected_anchor_count=2)
        self.assertEqual(original,rp.read_bytes()); self.assertEqual(first['report_content_sha256'],second['report_content_sha256'])
        for name in ('adjudicated_anchor_manifest.jsonl','observation_anchor_decisions.jsonl','eligible_anchor_manifest.jsonl','schema_issue_candidates.jsonl','second_pass_review_queue.jsonl','role_label_summary.csv'):
            self.assertEqual((self.root/'one'/name).read_bytes(),(self.root/'two'/name).read_bytes())
        d=json.loads((self.root/'one'/'observation_anchor_decisions.jsonl').read_text())
        self.assertEqual(d['selected_anchor_candidate_id'],'a')

    def test_duplicate_grounding_disagreement_warns_not_blocks(self):
        rows=[raw('a'),raw('b')]; reviews=[review('a','BASE_PLATE','VISIBLE','ACCEPTED'),review('b','BASE_PLATE','VISIBLE','BOX_TOO_BROAD')]
        report=self.adjudicate_rows(rows,reviews,expected=2)
        self.assertEqual(report['warning_count'],1)
        issue=(self.root/'out'/'schema_issue_candidates.jsonl').read_text()
        self.assertIn('DUPLICATE_GROUNDING_INCONSISTENT_REVIEW',issue)

    def test_cli_and_no_backend_certificate_import(self):
        rows=[raw(f'a{i}', frame=f'{i:064x}') for i in range(75)]; reviews=[review(f'a{i}','BASE_PLATE','VISIBLE','ACCEPTED') for i in range(75)]
        rp=self.root/'raw.jsonl';vp=self.root/'review.jsonl';write_rows(rp,rows);write_rows(vp,reviews)
        script=Path(__file__).resolve().parents[1]/'scripts'/'adjudicate_v2_tal_human_overlay_reviews.py'
        result=subprocess.run([sys.executable,str(script),'--raw-anchor-manifest',str(rp),'--review-jsonl',str(vp),'--output-dir',str(self.root/'cli')],env={**__import__('os').environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},capture_output=True,text=True)
        self.assertEqual(result.returncode,0, result.stderr + result.stdout)
        audit=subprocess.run([sys.executable,'-c','import sys; import relive.v2.human_overlay_adjudication; assert "relive.backends" not in sys.modules; assert "relive.certificate" not in sys.modules'],env={**__import__('os').environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},capture_output=True,text=True)
        self.assertEqual(audit.returncode,0,audit.stderr)

if __name__=='__main__': unittest.main()
