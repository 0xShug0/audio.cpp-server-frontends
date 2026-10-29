"""CPU-only harness tests. Generated WAVs are parser fixtures, not model evidence."""
import array
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import wave

import rocm_v2_qwen as qwen
from rocm_v2_campaign import validate_manifest


class NativeCampaignTests(unittest.TestCase):
    def test_native_portable_paired_commands(self):
        plan = qwen.manifest(Path('/hosted/prepare'))
        self.assertEqual(plan, qwen.manifest(Path('/amd/runner')))
        validate_manifest(plan)
        encoded = json.dumps(plan).lower()
        self.assertNotIn('conda', encoded)
        self.assertNotIn('miniforge', encoded)
        model = next(build for build in plan['builds'] if build['id'] == 'model')
        self.assertEqual(len(model['commands']), 2)
        self.assertIn('-DCMAKE_BUILD_TYPE=Debug', model['commands'][0])
        self.assertIn('-DCMAKE_CXX_STANDARD=17', model['commands'][0])
        self.assertIn('-DAUDIOCPP_MODELS=qwen3_tts', model['commands'][0])
        self.assertEqual(model['commands'][1][-2:], ['--parallel', '8'])
        normalized = []
        cases = plan['suites'][0]['cases']
        self.assertEqual([case['id'] for case in cases], ['hip-a1', 'v2-b1', 'v2-b2', 'hip-a2'])
        for case, backend in zip(cases, ['hip', 'rocm_v2', 'rocm_v2', 'hip']):
            argv = case['argv'].copy()
            self.assertEqual(argv[0], 'build/debug/bin/audiocpp_cli')
            self.assertEqual(argv[argv.index('--backend') + 1], backend)
            self.assertIn('--log', argv)
            self.assertIn('--metrics', argv)
            for flag in ['--backend', '--out-dir']:
                argv[argv.index(flag) + 1] = '<paired-value>'
            normalized.append(argv)
        self.assertTrue(all(argv == normalized[0] for argv in normalized))
        self.assertFalse(plan['suites'][-1]['gpu'])
        self.assertEqual(plan['suites'][-1]['id'], 'report')

    def test_default_sampling_and_natural_stop_inputs(self):
        self.assertEqual([row['id'] for row in qwen.REQUESTS], ['warmup', 'short', 'medium', 'long', 'short_repeat'])
        for row in qwen.REQUESTS:
            self.assertEqual(row['seed'], 1234)
            self.assertEqual(row['voice_ref'], 'reference.wav')
            self.assertEqual(row['options'], {'x_vector_only_mode': True})
            self.assertNotIn('max_tokens', row)
            self.assertNotIn('temperature', row)
        self.assertEqual(qwen.REQUESTS[1]['text'], qwen.REQUESTS[-1]['text'])


class ResultCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='native-qwen-parser-')
        self.root = Path(self.temp.name)
        for variant in ['hip-a1', 'v2-b1', 'v2-b2', 'hip-a2']:
            v2 = variant.startswith('v2')
            lines = ['ROCm0: implementation=rocm_v2'] if v2 else []
            for row in qwen.REQUESTS:
                name = row['id']
                path = self.root / 'wavs' / variant / (name + '.wav')
                path.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(path), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(array.array('h', [100, -100] * 12000).tobytes())
                lines.extend(['[TIMING] qwen3_tts.talker_ms 200', '[TIMING] qwen3_tts.speech_decoder_ms 300',
                    f'metrics[{name}].wall_ms={9000 if name == "warmup" else 500 if v2 else 1000}',
                    f'metrics[{name}].audio_duration_ms=1000'])
            log = self.root / 'case/correctness' / (variant + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('\n'.join(lines))

    def tearDown(self):
        self.temp.cleanup()

    def summarize(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return qwen.summarize(self.root)

    def test_warmup_excluded_and_repeats_checked(self):
        result = self.summarize()
        self.assertEqual(len(result['rows']), 16)
        self.assertEqual(result['aggregate']['hip']['rtf'], 1)
        self.assertEqual(result['aggregate']['rocm_v2']['rtf'], 0.5)
        self.assertTrue(all(result['repeat_byte_identical'].values()))
        self.assertTrue(all(row['byte_identical'] for row in result['parity']))
        self.assertTrue((self.root / 'qwen-summary.json').exists())

    def test_summary_cli_entrypoint(self):
        result = subprocess.run([sys.executable, str(Path(qwen.__file__).resolve()),
            '--summarize', str(self.root)], text=True, capture_output=True, check=True)
        parsed = json.loads(result.stdout)
        self.assertEqual(parsed['aggregate']['hip']['rtf'], 1)
        self.assertEqual(parsed['aggregate']['rocm_v2']['rtf'], 0.5)

    def test_missing_request_metrics_fails(self):
        log = self.root / 'case/correctness/hip-a1.log'
        log.write_text('\n'.join(line for line in log.read_text().splitlines() if not line.startswith('metrics[long]')))
        with self.assertRaises(AssertionError):
            self.summarize()

    def test_missing_wav_fails(self):
        (self.root / 'wavs/v2-b1/long.wav').unlink()
        with self.assertRaises(FileNotFoundError):
            self.summarize()

    def test_backend_selection_cannot_be_ignored(self):
        log = self.root / 'case/correctness/v2-b1.log'
        log.write_text(log.read_text().replace('implementation=rocm_v2', 'ordinary HIP'))
        with self.assertRaises(AssertionError):
            self.summarize()


if __name__ == '__main__':
    unittest.main()
