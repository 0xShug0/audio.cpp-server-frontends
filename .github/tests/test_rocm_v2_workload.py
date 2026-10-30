import csv,json,tempfile,unittest
from pathlib import Path
import rocm_v2_workload as workload

ROOT=Path(__file__).resolve().parent

class WorkloadTests(unittest.TestCase):
    def test_all_captured_q8_dimensions_cover_every_column(self):
        cases=json.loads((ROOT/'workload-probes/cases.json').read_text())
        captured={(r['sources'][0]['ne'][0],r['sources'][0]['ne'][1]) for r in cases
                  if r['op']=='MUL_MAT' and r['sources'][0]['type']==8 and r['models']}
        self.assertEqual(len(captured),57)
        for k,m in captured:
            columns={r['sources'][1]['ne'][1] for r in cases if r['op']=='MUL_MAT'
                     and r['sources'][0]['type']==8 and r['sources'][0]['ne'][:2]==[k,m]}
            self.assertTrue(set(range(1,9))<=columns,(k,m,columns))
        for k,m in [(1024,1024),(1024,4096),(4096,1024),(1024,64)]:
            self.assertIn((k,m),captured)

    def test_manifest_covers_all_fixtures_and_builds_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.github/tests').mkdir(parents=True)
            (root/'.github/tests/workload-probes').symlink_to(ROOT/'workload-probes',target_is_directory=True)
            plan=workload.manifest(root)
        self.assertEqual(len(plan['builds']),5)
        for build in plan['builds'][1:]:
            self.assertIn('-DCMAKE_BUILD_TYPE=Debug',build['commands'][0])
            self.assertIn('-DCMAKE_CXX_STANDARD=17',build['commands'][0])
            self.assertIn('test-backend-ops',build['commands'][1])
            self.assertIn('--parallel',build['commands'][1])
        all_ids=set(workload.fixture_ids(ROOT/'workload-probes/all.ops'))
        check=plan['suites'][0]['cases']
        native=set()
        for c in check:
            if c['id'].startswith('native-'):
                native.update(workload.fixture_ids(ROOT/'workload-probes'/Path(c['argv'][c['argv'].index('--test-file')+1]).name))
        self.assertEqual(native,all_ids)
        for c in plan['suites'][1]['cases']:
            self.assertTrue(c['depends_on'])
            self.assertIn('--duration-ms',c['argv'])
        self.assertNotIn('conda',json.dumps(plan))

    def test_structured_results_reject_skips_duplicates_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'case.log'
            def write(rows):
                with path.open('w') as f:
                    w=csv.DictWriter(f,['op_params','supported','passed','time_us','n_runs'],quoting=csv.QUOTE_ALL)
                    w.writeheader();w.writerows(rows)
            row=dict(op_params='name=fixture-1',supported='1',passed='1',time_us='4.25',n_runs='100')
            write([row]);self.assertEqual(set(workload.read_rows(path)),{'fixture-1'})
            for rows in [[dict(row,supported='0')],[dict(row,passed='0')],[row,row],[dict(row,op_params='')]]:
                write(rows)
                with self.assertRaises(ValueError):workload.read_rows(path)

    def test_selection_requires_actual_timing_and_rejects_other_op_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';fixtures=source/'.github/tests/workload-probes'
            fixtures.mkdir(parents=True);(root/'case/correctness').mkdir(parents=True);(root/'case/operators').mkdir(parents=True)
            catalog=[dict(id='q8',models=['ming3_tts'],group='q8',reasons=['weight bank']),
                     dict(id='copy',models=['yue2'],group='copy',reasons=[])]
            (fixtures/'cases.json').write_text(json.dumps(catalog))
            plan=dict(suites=[dict(id='correctness',cases=[]),dict(id='operators',cases=[])])
            for variant in workload.VARIANTS:
                ids=['q8','copy'] if variant in ['native','full1'] else ['q8']
                fixture=fixtures/(variant+'.ops');fixture.write_text(''.join('fixture '+key+'\n' for key in ids))
                def log(path,times):
                    with path.open('w') as f:
                        w=csv.DictWriter(f,['op_params','supported','passed','time_us','n_runs'],quoting=csv.QUOTE_ALL);w.writeheader()
                        for key,time in times.items():w.writerow(dict(op_params='name='+key,supported=1,passed=1,time_us=time,n_runs=10))
                name=variant+'-check'
                plan['suites'][0]['cases'].append(dict(id=name,argv=['--test-file',str(fixture.relative_to(source))]))
                log(root/'case/correctness'/(name+'.log'),{key:0 for key in ids})
                for repeat in range(2):
                    name=f'r{repeat}-{variant}-perf'
                    plan['suites'][1]['cases'].append(dict(id=name,argv=['--test-file',str(fixture.relative_to(source))]))
                    log(root/'case/operators'/(name+'.log'),{key:10 if variant=='native' else 5 for key in ids})
            (source/'campaign.json').write_text(json.dumps(plan));(root/'summary.json').write_text(json.dumps(dict(results=[])))
            workload.select(root,source)
            self.assertTrue(json.loads((root/'selected-variant.json').read_text())['accepted'])
            for repeat in range(2):
                log(root/'case/operators'/f'r{repeat}-full1-perf.log',dict(q8=5,copy=20))
            with self.assertRaises(ValueError):workload.select(root,source)
            selected=json.loads((root/'selected-variant.json').read_text())
            self.assertFalse(selected['accepted']);self.assertEqual(selected['non_q8_regressions'],['copy'])
            log(root/'case/operators'/'r0-full2-perf.log',dict(q8=0))
            with self.assertRaises(ValueError):workload.select(root,source)

    def test_model_abba_keeps_requests_and_natural_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';(source/'inputs').mkdir(parents=True)
            (root/'selected-variant.json').write_text(json.dumps(dict(candidate='full1',correctness_verified=True)))
            (source/'inputs/workload-requests.json').write_bytes((ROOT/'workload-requests.json').read_bytes())
            plan=workload.model_manifest(root,source)
        requests=json.loads((ROOT/'workload-requests.json').read_text())
        self.assertEqual(set(requests),{'breeze_tts','ming3_tts','omnivoice','qwen3_tts','irodori_tts','yue2'})
        for family in requests:
            cmds=[]
            for label in ['a1','b1','b2','a2']:
                c=next(s for s in plan['suites'] if s['id']==family+'-'+label)['cases'][0]['argv'].copy()
                self.assertIn('--log',c);self.assertIn('--metrics',c)
                if label.startswith('b'):
                    idx=c.index('--backend-impl');del c[idx:idx+2]
                else:self.assertNotIn('--backend-impl',c)
                c[0]='<binary>';c[c.index('--out-dir')+1]='<out>';cmds.append(c)
            self.assertTrue(all(c==cmds[0] for c in cmds))
        self.assertNotIn('max_tokens',json.dumps(requests))
        self.assertNotIn('max_new_tokens',json.dumps(requests))
        self.assertEqual(requests['yue2'][1],dict(requests['yue2'][-1],id=requests['yue2'][1]['id']))

if __name__=='__main__':unittest.main()
