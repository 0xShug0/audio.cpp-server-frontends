"""CPU-only checks of native Halo campaign staging and selection commands."""
from pathlib import Path
import json
import unittest
import tempfile
import csv
import re
from unittest.mock import patch
import rocm_v2_halo
from rocm_v2_campaign import validate_manifest
from rocm_v2_halo import manifest, operator_manifest, model_manifest, model_requests, followup_manifest, select_candidate, hip_baseline_manifest


class HaloCampaignTests(unittest.TestCase):
    def test_generic_gates_isolated_probes_and_matched_builds(self):
        plan = rocm_v2_halo.generic_gates_manifest(Path('.'))
        validate_manifest(plan)
        native, candidate = plan['builds'][1:]
        self.assertEqual(native['commands'], candidate['commands'])
        self.assertEqual(native['cwd'], '../source-baseline')
        self.assertIn('-DCMAKE_BUILD_TYPE=Debug', candidate['commands'][0])
        text = json.dumps(plan)
        for forbidden in ['audiocpp_cli', '.gguf', 'conda', 'pip install']:
            self.assertNotIn(forbidden, text)
        self.assertEqual([s['id'] for s in plan['suites']],
                         ['correctness', 'candidate-checks', 'framework'])
        timing = [c for c in plan['suites'][-1]['cases'] if not c['id'].startswith('trace-')]
        self.assertEqual(len(timing), 78 * 12)
        self.assertTrue(all(len(c['depends_on']) == 1 for c in timing))
        for suite in plan['suites'][:2]:
            self.assertEqual(len(suite['cases']), 216)
            self.assertTrue(all('--test-file' in c['argv'] or 'strict-f32' in c['id']
                                for c in suite['cases']))

    def test_tts_main_reuses_build_and_selects_only_candidate_impl(self):
        plan = rocm_v2_halo.tts_main_manifest(Path('.'))
        validate_manifest(plan)
        existing = rocm_v2_halo.irodori_main_manifest(Path('.'))
        self.assertEqual(plan['builds'][0], existing['builds'][0])
        self.assertEqual(plan['builds'][1]['commands'][1], existing['builds'][1]['commands'][1])
        families = list(rocm_v2_halo.tts_requests())
        self.assertEqual(families, ['breeze_tts', 'ming3_tts', 'omnivoice', 'qwen3_tts', 'irodori_tts'])
        for family in families:
            commands = []
            for label in ['a1', 'b1', 'b2', 'a2']:
                suite = next(s for s in plan['suites'] if s['id'] == family + '-' + label)
                argv = suite['cases'][0]['argv'].copy()
                if label.startswith('b'):
                    index = argv.index('--backend-impl')
                    self.assertEqual(argv[index + 1], 'gfx1151_v1')
                    del argv[index:index + 2]
                else:
                    self.assertNotIn('--backend-impl', argv)
                argv[argv.index('--out-dir') + 1] = '<out>'
                commands.append(argv)
            self.assertTrue(all(c == commands[0] for c in commands))
            requests = rocm_v2_halo.tts_requests()[family]
            self.assertEqual([r['id'] for r in requests], ['warmup', 'short', 'long', 'short_repeat'])
            self.assertEqual(requests[1]['text'], requests[-1]['text'])
            self.assertNotIn('voice_ref', json.dumps(requests))
            self.assertNotIn('max_tokens', json.dumps(requests))
        self.assertEqual(plan['suites'][-1]['id'], 'framework')
        self.assertTrue(all('language' not in r for r in rocm_v2_halo.tts_requests()['breeze_tts']))

    def test_ggml_v1_uses_real_pr_tests_without_models(self):
        plan = rocm_v2_halo.ggml_v1_manifest(Path('.'))
        validate_manifest(plan)
        build = next(b for b in plan['builds'] if b['id'] == 'operators')
        self.assertEqual(build['commands'][0][2], 'external/ggml')
        self.assertIn('-DCMAKE_BUILD_TYPE=Debug', build['commands'][0])
        self.assertEqual(build['commands'][1][-2:], ['--parallel', '8'])
        cases = plan['suites'][0]['cases']
        self.assertEqual(len(cases), 10)
        self.assertEqual({c['argv'][c['argv'].index('--backend-impl')+1] for c in cases},
                         {'default', 'gfx1151_v1'})
        self.assertTrue(all(c['argv'][c['argv'].index('--suite')+1] == 'gfx1151_v1' for c in cases))
        self.assertNotIn('audiocpp_cli', json.dumps(plan))
        self.assertNotIn('https://', json.dumps(plan))
        self.assertNotIn('conda', json.dumps(plan))

    def test_irodori_current_main_has_one_build_and_serial_abba(self):
        plan = rocm_v2_halo.irodori_main_manifest(Path('.'))
        validate_manifest(plan)
        self.assertEqual(len(plan['builds']), 2)
        model = plan['builds'][1]
        self.assertIn('-DAUDIOCPP_MODELS=irodori_tts', model['commands'][0])
        self.assertIn('-DCMAKE_BUILD_TYPE=Debug', model['commands'][0])
        self.assertEqual(model['commands'][1][-2:], ['--parallel', '8'])
        normalized = []
        for label in ['a1', 'b1', 'b2', 'a2']:
            suite = next(s for s in plan['suites'] if s['id'] == 'irodori_tts-' + label)
            argv = suite['cases'][0]['argv'].copy()
            self.assertEqual(argv[0], 'build/debug/bin/audiocpp_cli')
            self.assertIn('--log', argv)
            self.assertIn('--metrics', argv)
            if label.startswith('a'):
                self.assertNotIn('--backend-impl', argv)
            else:
                idx = argv.index('--backend-impl')
                self.assertEqual(argv[idx+1], 'gfx1151_v1')
                del argv[idx:idx+2]
            argv[argv.index('--out-dir')+1] = '<out>'
            normalized.append(argv)
        self.assertTrue(all(c == normalized[0] for c in normalized))
        self.assertNotIn('conda', json.dumps(plan))
        self.assertNotIn('pip install', json.dumps(plan))
        self.assertNotIn('max_tokens', json.dumps(model_requests()['irodori_tts']))
        self.assertEqual(plan['suites'][-1]['cases'][0]['argv'][-1], 'gfx1151_v1')

    def test_native_and_portable(self):
        plan = manifest(Path('/hosted/source'))
        self.assertEqual(plan, manifest(Path('/runner/source')))
        validate_manifest(plan)
        text = json.dumps(plan).lower()
        self.assertNotIn('conda', text)
        self.assertNotIn('pip install', text)
        self.assertNotIn('miniforge', text)
        build = next(row for row in plan['builds'] if row['id'] == 'model')
        self.assertIn('-DCMAKE_BUILD_TYPE=Debug', build['commands'][0])
        self.assertIn('-DCMAKE_CXX_STANDARD=17', build['commands'][0])
        self.assertEqual(build['commands'][1][-2:], ['--parallel', '8'])

    def test_operators_do_not_run_or_download_models(self):
        plan = operator_manifest(Path('.'))
        validate_manifest(plan)
        text = json.dumps(plan)
        self.assertNotIn('audiocpp_cli', text)
        self.assertNotIn('inputs/', text)
        self.assertNotIn('conda', text)
        self.assertEqual(len(plan['suites'][0]['cases']), 10)
        self.assertEqual(len(plan['suites'][1]['cases']), 24)
        self.assertTrue(all('fused' not in x['id'] for x in plan['suites'][1]['cases']))

    def test_model_pairs_use_separate_binaries_and_identical_controls(self):
        plan = model_manifest(Path('.'))
        validate_manifest(plan)
        for build in plan['builds'][1:]:
            self.assertIn('-DCMAKE_BUILD_TYPE=Debug', build['commands'][0])
            self.assertIn('-DCMAKE_CXX_STANDARD=17', build['commands'][0])
            self.assertIn('-DAUDIOCPP_MODELS=qwen3_tts,irodori_tts,yue2', build['commands'][0])
        for family, requests in model_requests().items():
            self.assertEqual(requests[0]['id'], 'warmup')
            self.assertEqual(requests[-1]['text'], requests[1]['text'])
            self.assertTrue(all('max_tokens' not in r['options'] for r in requests))
            cases = [next(s for s in plan['suites'] if s['id'] == family + '-' + v)['cases'][0]['argv']
                     for v in ['a1','b1','b2','a2']]
            self.assertTrue(cases[0][0].startswith('../source-baseline/'))
            self.assertFalse(cases[1][0].startswith('../source-baseline/'))
            normalized = []
            for argv in cases:
                argv = argv.copy()
                self.assertEqual(argv[argv.index('--backend-impl') + 1], 'halo')
                argv[0] = '<binary>'
                argv[argv.index('--out-dir') + 1] = '<out>'
                normalized.append(argv)
            self.assertTrue(all(row == normalized[0] for row in normalized))

    def test_followup_times_actual_one_token_work_and_copy(self):
        plan = followup_manifest(Path('.'))
        validate_manifest(plan)
        self.assertEqual(len(plan['suites'][0]['cases']),6)
        self.assertEqual(len(plan['suites'][1]['cases']),17)
        build = plan['builds'][1]
        self.assertEqual(build['commands'][0][2],'.')
        self.assertIn('halo_backend_ops',build['commands'][1])
        self.assertIn('--kernel-trace',plan['suites'][1]['cases'][-1]['argv'])

    def test_selection_rejects_weak_results_and_preserves_proven_copy(self):
        for accept in [False,True]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, copy, results = root/'source', root/'source-copy', root/'results'
                names = ['external/ggml/src/ggml-cuda/mmvq.cu',
                         'external/ggml/src/ggml-cuda/ggml-cuda.cu',
                         'external/ggml/src/ggml-hip/implementation.cuh',
                         'external/ggml/src/ggml-hip/halo/halo.cuh',
                         'external/ggml/src/ggml-hip/halo/cpy.cu']
                for name in names:
                    (source/name).parent.mkdir(parents=True,exist_ok=True)
                    (copy/name).parent.mkdir(parents=True,exist_ok=True)
                    (source/name).write_text('candidate: src0->ne[0] < 512 || src0->ne[1] < 1024; src->ne[0] < 64')
                    (copy/name).write_text('proven copy-only baseline')
                results.mkdir()
                (results/'summary.json').write_text(json.dumps(dict(status='PASS',complete=True)))
                profile=results/'profile/q8';profile.mkdir(parents=True)
                with (profile/'test_kernel_trace.csv').open('w') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['Kernel_Name']);writer.writeheader()
                    writer.writerow(dict(Kernel_Name='void mul_mat_vec_q<(ggml_type)8, 1, false, false, false, 4>(...)'))
                    writer.writerow(dict(Kernel_Name='halo_copy_rows(...)'))
                records=[]
                speed=2.0 if accept else 0.9
                for k in [512,1024,2048,3072,8192]:
                    for m in [1024,3072]:
                        records.append(dict(op='MUL_MAT',params=f'type_a=q8_0,m={m},n=1,k={k}',
                            speedup=speed,default_min=10.,halo_max=10./speed))
                for w in [64,128]:
                    for i in range(4):
                        records.append(dict(op='CPY',params=f'ne=[{w},16,4,1],perm={i}',
                            speedup=speed,default_min=10.,halo_max=10./speed))
                with patch.object(rocm_v2_halo,'summarize_operators',return_value=records):
                    select_candidate(results,source)
                decision=json.loads((results/'selection.json').read_text())
                self.assertEqual(decision['min_k'],512 if accept else None)
                self.assertEqual(decision['copy_min_columns'],64 if accept else 256)
                validate_manifest(json.loads((source/'model-campaign.json').read_text()))
                for name,digest in decision['selected_source_sha256'].items():
                    self.assertEqual((source/name).read_bytes(),(results/'selected-source'/name).read_bytes())
                if not accept:
                    self.assertTrue(all((source/name).read_bytes()==(copy/name).read_bytes() for name in names))

    def test_original_hip_baseline_omits_selector_and_uses_same_binary(self):
        plan = hip_baseline_manifest(Path('.'))
        validate_manifest(plan)
        self.assertEqual([b['id'] for b in plan['builds']],['device-health','candidate'])
        for family in model_requests():
            paired=[]
            for v in ['a1','b1','b2','a2']:
                suite=next(s for s in plan['suites'] if s['id']==family+'-'+v)
                self.assertEqual(suite['build'],'candidate')
                argv=suite['cases'][0]['argv'].copy()
                self.assertEqual(argv[0],'./build/debug/bin/audiocpp_cli')
                if v.startswith('a'):
                    self.assertNotIn('--backend-impl',argv)
                else:
                    index=argv.index('--backend-impl')
                    self.assertEqual(argv[index+1],'halo')
                    del argv[index:index+2]
                argv[argv.index('--out-dir')+1]='<out>'
                self.assertIn('../results/wavs/'+family+'-',suite['cases'][0]['argv'][suite['cases'][0]['argv'].index('--out-dir')+1])
                paired.append(argv)
            self.assertTrue(all(x==paired[0] for x in paired))
        summary=plan['suites'][-1]['cases'][0]['argv']
        self.assertEqual(summary[-2:],['--baseline-implementation','default'])

    def test_selection_changes_only_implementation(self):
        plan = manifest(Path('.'))
        cases = next(row for row in plan['suites'] if row['id'] == 'correctness')['cases']
        normalized = []
        for row, impl in zip(cases, ['default', 'halo', 'halo', 'default']):
            argv = row['argv'].copy()
            self.assertEqual(argv[argv.index('--backend') + 1], 'hip')
            self.assertEqual(argv[argv.index('--backend-impl') + 1], impl)
            for flag in ['--backend-impl', '--out-dir']:
                argv[argv.index(flag) + 1] = '<paired>'
            self.assertIn('--log', argv)
            self.assertIn('--metrics', argv)
            normalized.append(argv)
        self.assertTrue(all(row == normalized[0] for row in normalized))
        report = plan['suites'][-1]['cases'][0]['argv']
        self.assertEqual(report[-2:], ['--implementation', 'halo'])

    def test_operator_and_profile_cases_are_serial_and_distinct(self):
        plan = manifest(Path('.'))
        checks = plan['suites'][0]
        self.assertEqual(checks['id'], 'operators')
        self.assertEqual(len(checks['cases']), 9)
        trace = next(row for row in plan['suites'] if row['id'] == 'framework')
        self.assertEqual([row['id'] for row in trace['cases']], ['default', 'halo'])
        self.assertIn('../results/profile/halo', trace['cases'][1]['argv'])


if __name__ == '__main__':
    unittest.main()
