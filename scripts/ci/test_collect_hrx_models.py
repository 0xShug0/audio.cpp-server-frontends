"""Collector lifecycle/cache tests. No model or performance claims."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import collect_hrx_models as collector


class CollectorTest(unittest.TestCase):
    def test_saved_output_and_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = collector.run([sys.executable, '-u', '-c', "print('raw output')"], root / 'normal', root, 5)
            self.assertEqual(result['exit_code'], 0)
            self.assertEqual((root / 'normal.log').read_text(), 'raw output\n')

    def test_timeout_terminates_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = collector.run([sys.executable, '-c', 'import time; time.sleep(60)'], root / 'timeout', root, 0.2)
            self.assertEqual(result['exit_code'], 'timeout')
            self.assertFalse((root / 'cleanup-failed.json').exists())

    def test_failed_kill_is_bounded_and_prevents_more_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = Mock(pid=12345)
            child.poll.return_value = None
            child.wait.side_effect = [subprocess.TimeoutExpired('child', n) for n in (1, 10, 5)]
            follow = Mock()
            with patch.object(collector.subprocess, 'Popen', side_effect=[child, follow]), patch.object(
                    collector.os, 'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError, 'survived SIGKILL'):
                    collector.run(['child'], root / 'c1', root, 1)
            self.assertEqual([c.kwargs['timeout'] for c in child.wait.call_args_list], [1, 10, 5])
            self.assertEqual([c.args[1] for c in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])
            model = root / 'breeze'
            model.mkdir()
            (root / 'cleanup-failed.json').rename(model / 'cleanup-failed.json')
            with patch.object(sys, 'argv', ['collector', '--model', 'breeze', '--source', tmp, '--results', tmp]):
                with self.assertRaisesRegex(RuntimeError, 'Previous child cleanup failed'):
                    collector.main()

    def test_log_follower_start_failure_cleans_up_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = Mock(pid=12345)
            child.poll.return_value = None
            with patch.object(collector.subprocess, 'Popen', side_effect=[child, OSError('tail failed')]), patch.object(
                    collector.os, 'killpg') as kill:
                with self.assertRaisesRegex(OSError, 'tail failed'):
                    collector.run(['child'], root / 'c1', root, 1)
            kill.assert_called_once_with(12345, signal.SIGKILL)
            child.wait.assert_called_once_with(timeout=5)

    def test_cancel_stops_detached_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_dir = str(Path(collector.__file__).resolve().parent)
            child_code = ('import os,sys,time; from pathlib import Path; '
                          'Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)')
            worker_code = ('import sys,signal; from pathlib import Path; '
                           f'sys.path.insert(0,{module_dir!r}); import collect_hrx_models as c; '
                           'signal.signal(signal.SIGTERM,signal.default_int_handler); '
                           f'c.run([sys.executable,"-u","-c",{child_code!r},sys.argv[1]],'
                           'Path(sys.argv[2]),Path(sys.argv[3]),60)')
            worker = subprocess.Popen([sys.executable, '-c', worker_code, str(root / 'child.pid'),
                                       str(root / 'cancel'), tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not (root / 'child.pid').exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue((root / 'child.pid').exists(), 'child never started')
                child_pid = int((root / 'child.pid').read_text())
                worker.send_signal(signal.SIGTERM)
                self.assertNotEqual(worker.wait(timeout=8), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_verified_runner_cache_and_mirror_failure(self):
        payload = b'cache fixture'
        sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = []
            mirror_fails = False
            corrupt_download = False

            def run(command, stem, cwd, timeout):
                if command[0] == 'curl':
                    downloads.append(command[-1])
                    Path(command[command.index('--output') + 1]).write_bytes(b'bad' if corrupt_download else payload)
                    return {'exit_code': 1 if mirror_fails and 'mirror.invalid' in command[-1] else 0}
                return {'exit_code': 0}

            def invoke(label):
                with patch.object(sys, 'argv', ['collector', '--model', 'breeze', '--source', tmp,
                                               '--results', str(root / label), '--backends', 'hrx', '--blocks', 'c1']):
                    return collector.main()

            with patch.dict(os.environ, {'RUNNER_TEMP': str(root / 'runner'), 'XDG_CACHE_HOME': str(root / 'cache'),
                                         'HF_ENDPOINT': 'https://mirror.invalid'}), patch.dict(
                    collector.MODELS, {'breeze': ('breeze_tts', 'repo', 'revision', 'model.gguf', sha)}), patch.object(
                    collector, 'run', side_effect=run):
                self.assertEqual(invoke('cold'), 0)
                self.assertEqual(invoke('warm'), 0)
                self.assertEqual(len(downloads), 1)
                cached = root / 'cache/audio-cpp-ci' / sha
                cached.write_bytes(b'corrupt')
                mirror_fails = True
                self.assertEqual(invoke('corrupt'), 0)
                self.assertEqual(len(downloads), 3)
                self.assertEqual(cached.read_bytes(), payload)
                cached.unlink()
                mirror_fails = False
                corrupt_download = True
                with self.assertRaisesRegex(RuntimeError, 'SHA-256 mismatch'):
                    invoke('bad')
                self.assertFalse(cached.exists())
                self.assertFalse(list(cached.parent.glob('*.part')))


if __name__ == '__main__':
    unittest.main()
