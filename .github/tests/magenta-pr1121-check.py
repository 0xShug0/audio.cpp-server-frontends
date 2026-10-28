import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf

root = Path(sys.argv[1])
report = {"backend": "hip", "requests": {}}
digests = []
for mode in ("streaming", "offline"):
    path = root / f"{mode}.wav"
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == 48000 and samples.shape == (384000, 2), (rate, samples.shape)
    assert np.isfinite(samples).all(), f"{mode}: nonfinite output"
    energy = float(np.mean(samples.astype(np.float64) ** 2))
    assert energy > 1e-8, f"{mode}: silent output"
    peak = float(np.abs(samples).max())
    assert peak <= 1, (mode, peak)
    log = (root / f"{mode}-framework.log").read_text()
    def timing(name):
        matches = re.findall(r"\] " + re.escape(name) + r" ([0-9.eE+-]+)", log)
        assert matches, f"Missing {name} in {mode} logs"
        return float(matches[-1])
    duration = samples.shape[0] / rate
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    digests.append(digest)
    report["requests"][mode] = {
        "duration_sec": duration,
        "sample_rate": rate,
        "channels": samples.shape[1],
        "generation_ms": timing("session.wall_ms"),
        "rtf": timing("session.wall_ms") / (1000 * duration),
        "ttfa_ms": timing("session.ttfa_ms"),
        "ar_ms": timing("magenta_realtime_2.ar.total_ms"),
        "codec_ms": timing("magenta_realtime_2.codec.total_ms"),
        "peak": peak,
        "rms": energy ** 0.5,
        "clipped_fraction": float(np.mean(np.abs(samples) >= 0.9999)),
        "sha256": digest,
    }
report["streaming_offline_byte_identical"] = digests[0] == digests[1]
print(json.dumps(report, indent=2), flush=True)
assert report["streaming_offline_byte_identical"], "Streaming/offline WAV mismatch"
