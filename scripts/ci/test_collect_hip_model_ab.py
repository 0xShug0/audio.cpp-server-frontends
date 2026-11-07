"""Harness-only tests: these make no model correctness/performance claims."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_hip_model_ab as collector


class CollectorTest(unittest.TestCase):
    def exercise(self, failed_variant=None, download_fails=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {'id': 'chatterbox', 'family': 'chatterbox', 'repo': 'fixture/model',
                     'revision': '0' * 40, 'files': [{'path': 'model.gguf',
                     'sha256': hashlib.sha256(b'weights').hexdigest()}]}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'source_sha': '0' * 40, 'threads': 8, 'models': [model]}))
            args = argparse.Namespace(manifest=manifest, model='chatterbox', binary=root / 'cli',
                                      workspace=root / 'weights', results=root / 'results', timeout=1)
            calls = []

            def fake_run(command, stem, env, cwd, timeout):
                calls.append((command, env))
                if stem.name == 'download':
                    if download_fails:
                        return {'exit_code': 1}
                    (args.workspace / 'chatterbox/model.gguf').write_bytes(b'weights')
                else:
                    output = Path(command[command.index('--out-dir') + 1])
                    output.mkdir()
                    (output / 'measured.wav').write_bytes(b'fixture, not actual audio')
                return {'exit_code': 1 if stem.name == failed_variant else 0}

            with patch.object(collector, 'run_raw', side_effect=fake_run), patch.object(
                    collector.urllib.request, 'urlopen', side_effect=lambda *a, **kw: io.BytesIO(b'reference')):
                code = collector.collect(args)
                status = json.loads((args.results / 'chatterbox/status.json').read_text())
                self.assertFalse((args.workspace / 'chatterbox').exists())
                self.assertTrue(status['weights_and_download_cache_removed'])
                if not download_fails:
                    self.assertEqual(len(calls), 5)
                    self.assertEqual([v['label'] for v in status['variants']], ['a1', 'b1', 'b2', 'a2'])
                    for command, env in calls:
                        self.assertTrue(env['HF_HOME'].startswith(str(args.workspace.resolve())))
                        self.assertTrue(env['HF_XET_CACHE'].startswith(str(args.workspace.resolve())))
                    for command, _ in (calls[1], calls[4]):
                        self.assertNotIn('--backend-impl', command)
                    for command, _ in (calls[2], calls[3]):
                        self.assertEqual(command[-2:], ['--backend-impl', 'gfx1151_v1'])
                if code == 0:
                    before = len(calls)
                    self.assertEqual(collector.collect(args), 0)
                    self.assertEqual(len(calls), before, 'completed collection must not rerun')
                return code, status

    def test_success_and_skip(self):
        self.assertEqual(self.exercise()[0], 0)

    def test_failed_variant_does_not_stop_remaining_variants(self):
        code, status = self.exercise(failed_variant='b1')
        self.assertEqual(code, 1)
        self.assertEqual(status['variants'][-1]['label'], 'a2')

    def test_failed_download_is_cleaned(self):
        self.assertEqual(self.exercise(download_fails=True)[0], 1)


if __name__ == '__main__':
    unittest.main()
