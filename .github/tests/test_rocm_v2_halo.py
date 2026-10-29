"""CPU-only checks of native Halo campaign staging and selection commands."""
from pathlib import Path
import json
import unittest
from rocm_v2_campaign import validate_manifest
from rocm_v2_halo import manifest


class HaloCampaignTests(unittest.TestCase):
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
