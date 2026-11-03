"""Collect streaming timings and raw GPU-memory samples; retain final WAVs."""
import base64
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request
import wave

root = Path.cwd()
out = root / 'results'
with wave.open(str(root / 'models/sample.wav')) as audio:
    duration = audio.getnframes() / audio.getframerate()
config = {'host': '127.0.0.1', 'port': 8035, 'backend': 'hip', 'threads': 8,
          'lazy_load': False, 'ui': False, 'max_loaded_models': 1,
          'models': [{'id': 'realtime-reuse', 'family': 'realtime_reuse',
                      'path': str(root / 'models/realtime-reuse-f32.gguf'),
                      'task': 's2s', 'mode': 'streaming'}]}
(out / 'config.json').write_text(json.dumps(config, indent=2))
command = [str(root / 'source/build/debug/bin/audiocpp_server'), '--config',
           str(out / 'config.json'), '--log', '--log-file', str(out / 'server.log')]
(out / 'command.json').write_text(json.dumps(command, indent=2))
stop = threading.Event()
samples = []
with (out / 'console.log').open('w') as log:
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    def monitor():
        files = list(Path('/sys/class/drm').glob('card[0-9]*/device/mem_info_vram_used'))
        while not stop.is_set():
            row = {'monotonic_s': time.monotonic(), 'device_vram_bytes': {}, 'fdinfo': {}}
            for path in files:
                row['device_vram_bytes'][str(path)] = int(path.read_text())
            for path in Path('/proc/' + str(server.pid) + '/fdinfo').glob('*'):
                try:
                    text = path.read_text()
                except FileNotFoundError:
                    continue  # File descriptors close while the server is running.
                if 'drm-client-id:' in text:
                    row['fdinfo'][path.name] = text
            samples.append(row)
            stop.wait(.025)
    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        for _ in range(600):
            if server.poll() is not None:
                raise RuntimeError('server exited: ' + str(server.returncode))
            try:
                with urllib.request.urlopen('http://127.0.0.1:8035/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('server health timeout')
        rows = []
        for iteration in range(3):
            payload = {'model': 'realtime-reuse', 'stream_format': 'sse',
                       'request': {'audio': str(root / 'models/sample.wav')}}
            request = urllib.request.Request('http://127.0.0.1:8035/v1/tasks/stream',
                                            data=json.dumps(payload).encode(),
                                            headers={'Content-Type': 'application/json'})
            started = time.perf_counter()
            first = None
            result = None
            events = 0
            with urllib.request.urlopen(request, timeout=300) as response:
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        continue
                    event = json.loads(data)
                    if event.get('type') == 'task.stream.event' and event.get('event', {}).get('audio'):
                        if first is None:
                            first = (time.perf_counter() - started) * 1000
                        events += 1
                    elif event.get('type') == 'task.stream.done':
                        result = event['result']
                    elif 'error' in event:
                        raise RuntimeError(str(event))
            wall = (time.perf_counter() - started) * 1000
            if result is None or first is None:
                raise RuntimeError('missing final result or streaming audio')
            (out / ('request-' + str(iteration) + '.wav')).write_bytes(base64.b64decode(result['audio']))
            row = {'iteration': iteration, 'client_wall_ms': wall, 'ttfa_ms': first,
                   'rtf': wall / (duration * 1000), 'audio_events': events,
                   'timing': result.get('timing')}
            rows.append(row)
            (out / 'requests.json').write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
        summary = {'audio_seconds': duration, 'warm_rtf': statistics.mean(r['rtf'] for r in rows[1:]),
                   'warm_ttfa_ms': statistics.mean(r['ttfa_ms'] for r in rows[1:]),
                   'memory_evidence': 'memory-samples.json contains raw process DRM and device memory counters'}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stop.set()
        thread.join()
        (out / 'memory-samples.json').write_text(json.dumps(samples))
