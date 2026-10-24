import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import urllib.request

root = Path.cwd()
results = root / "results"
results.mkdir(exist_ok=True)
harness = root / "harness/.github/tests/gguf-pr1072"
bins = {}


def run(command, log):
    with (results / log).open("w", encoding="utf-8") as output:
        subprocess.run([str(x) for x in command], stdout=output, stderr=subprocess.STDOUT, check=True)


for variant in ("main", "pr"):
    build = root / variant / "build/debug"
    run(["cmake", "-S", harness, "-B", build,
         f"-DAUDIOCPP_SOURCE={root / variant}", "-DCMAKE_BUILD_TYPE=Release"], f"{variant}-configure.log")
    run(["cmake", "--build", build, "--config", "Release", "--target", "gguf_read_benchmark",
         "--parallel", str(min(os.cpu_count() or 2, 8))], f"{variant}-build.log")
    bins[variant] = build / ("Release/gguf_read_benchmark.exe" if platform.system() == "Windows" else "gguf_read_benchmark")
    if not bins[variant].is_file():
        raise RuntimeError(f"Missing executable: {bins[variant]}")

fixture = root / "numeric-16m.gguf"
run([bins["main"], "--log", "--create", fixture], "fixture.log")
model = root / "moonshine-tiny-q8_0.gguf"
urllib.request.urlretrieve(
    "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf",
    model,
)
model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
if model_hash != "0f795f14924dbfe40d2ee0e220382c9b23cd9e89064c6ffb97e236d81f1d7020":
    raise RuntimeError(f"Model differs from the pinned test artifact: {model_hash}")
report = {"platform": platform.platform(), "model_sha256": model_hash, "cases": []}
for name, file in (("moonshine_tiny", model), ("numeric_metadata_16m", fixture)):
    times = {"main": [], "pr": []}
    hashes = set()
    for iteration in range(8):
        order = ("main", "pr") if iteration % 2 == 0 else ("pr", "main")
        for variant in order:
            log = f"{name}-{iteration}-{variant}.log"
            meta = results / f"{variant}-meta.bin"
            run([bins[variant], "--log", file, meta], log)
            hashes.add(hashlib.sha256(meta.read_bytes()).hexdigest())
            if iteration:
                entries = dict(line.split("=", 1) for line in (results / log).read_text().splitlines() if "=" in line)
                times[variant].append(float(entries["gguf.read_ms"]))
        if (results / "main-meta.bin").read_bytes() != (results / "pr-meta.bin").read_bytes():
            raise RuntimeError(f"Metadata differs: {name}, iteration {iteration}")
    if len(hashes) != 1:
        raise RuntimeError(f"Metadata is unstable: {name}")
    for variant in ("main", "pr"):
        (results / f"{variant}-meta.bin").unlink()
    medians = {key: statistics.median(value) for key, value in times.items()}
    report["cases"].append({"case": name, "input_bytes": file.stat().st_size,
                            "read_ms": times, "median_ms": medians,
                            "speedup": medians["main"] / medians["pr"],
                            "metadata_identical": True, "metadata_sha256": hashes.pop()})
text = json.dumps(report, indent=2)
(results / "summary.json").write_text(text + "\n")
print(text)
with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as output:
    output.write("Release CPU build; one discarded warmup, seven sequential reads per variant.\n\n")
    output.write("| Case | Main median ms | PR median ms | Speedup | Metadata identical |\n|---|---:|---:|---:|---|\n")
    for case in report["cases"]:
        output.write(f"| {case['case']} | {case['median_ms']['main']:.3f} | {case['median_ms']['pr']:.3f} | {case['speedup']:.2f}x | Yes |\n")
