"""Failure-isolated, serial ROCm CI orchestration for POSIX runners (stdlib only).

Usage: python rocm_v2_campaign.py --manifest campaign.json --output-dir results --log
Add --resume to preserve ALL terminal results, including failures. An interrupted
START becomes a failure on resume; retrying requires a new output directory.
--validate-only checks the full manifest without launching any subprocesses.
--through-suite builds stops after all builds; --through-suite ID stops after
that suite (in manifest order), including on resume when it is already complete.
Omit --through-suite to finish the campaign. Partial runs still exit nonzero if
any required result recorded so far failed, so CI can upload between phases.
--deadline-epoch UNIX_SECONDS bounds command execution across phases; pass the
same finite positive deadline on every invocation. Expired commands are blocked.

Manifest v1 (paths are relative to the manifest directory):
  {"version": 1,
   "builds": [{"id": "hip", "build_type": "Debug",
               "commands": [["cmake", "--build", "build/debug", "--parallel", "8"]]}],
   "suites": [{"id": "math", "build": "hip", "gpu": true,
               "cases": [{"id": "small", "argv": ["build/debug/probe", "--log"]}]}],
   "health_check": {"argv": ["build/debug/bin/test-backend-ops", "test",
                             "--suite", "rocm-v2", "--case-index", "0",
                             "-b", "ROCm0", "--log"], "timeout_s": 10}}

Builds, cases and health_check accept cwd, env and timeout_s. Suite defaults for
cwd/env/timeout_s/required are inherited by cases. Required defaults to true;
gpu defaults to false. Build depends_on lists earlier build IDs; case depends_on
lists earlier "suite/case" IDs. Failed dependencies block their dependents. Exit 0
means PASS, positive exits mean INCORRECT, signals/launch errors mean CRASH.
All build commands finish before any case starts. No shell is used. GPU crashes
and timeouts trigger the optional health check; without one, GPU work is blocked.
Only a passing health check permits further GPU cases. CPU work can continue.
JSONL START/DONE events and individual logs are flushed as work progresses.
"""

import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


STATUSES = {"PASS", "INCORRECT", "CRASH", "TIMEOUT", "BUILD_FAILED",
            "BLOCKED_BY_DEPENDENCY"}
COMMON = {"cwd", "env", "timeout_s", "required"}


def validate_manifest(data):
    """Reject ambiguous manifests before executing anything."""
    def fields(item, allowed, label):
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError(f"{label}: expected object with fields {sorted(allowed)}")
        if "cwd" in item and (not isinstance(item["cwd"], str) or not item["cwd"] or "\0" in item["cwd"]):
            raise ValueError(f"{label}: cwd must be a nonempty string")
        if "env" in item:
            env = item["env"]
            if not isinstance(env, dict) or any(
                not isinstance(k, str) or not k or "=" in k or "\0" in k
                or not isinstance(v, str) or "\0" in v for k, v in env.items()
            ):
                raise ValueError(f"{label}: env must map valid names to strings")
        if "required" in item and type(item["required"]) is not bool:
            raise ValueError(f"{label}: required must be boolean")
        if "timeout_s" in item and (
            type(item["timeout_s"]) not in (int, float)
            or not math.isfinite(item["timeout_s"]) or item["timeout_s"] <= 0
        ):
            raise ValueError(f"{label}: timeout_s must be finite and positive")

    def identifier(item):
        value = item.get("id")
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("id must contain only letters, digits, underscores, dots or hyphens")
        return value

    def argv(value):
        if not isinstance(value, list) or not value or any(
            not isinstance(arg, str) or "\0" in arg for arg in value
        ) or not value[0]:
            raise ValueError("command must be a nonempty argv array of strings")

    fields(data, {"version", "builds", "suites", "health_check"}, "manifest")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("manifest version must be 1")
    if not isinstance(data.get("builds"), list) or not isinstance(data.get("suites"), list):
        raise ValueError("manifest requires builds and suites arrays")
    builds, suites, cases = set(), set(), set()
    for build in data["builds"]:
        fields(build, COMMON | {"id", "build_type", "commands", "depends_on"}, "build")
        name = identifier(build)
        if name in builds:
            raise ValueError(f"duplicate build: {name}")
        deps = build.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(dep, str) or dep not in builds for dep in deps):
            raise ValueError(f"build {name}: depends_on must reference earlier builds")
        builds.add(name)
        if build.get("build_type") != "Debug":
            raise ValueError(f"build {name}: build_type must be Debug")
        if not isinstance(build.get("commands"), list) or not build["commands"]:
            raise ValueError(f"build {name}: commands must be a nonempty array")
        for command in build["commands"]:
            argv(command)
    for suite in data["suites"]:
        fields(suite, COMMON | {"id", "build", "gpu", "cases"}, "suite")
        name = identifier(suite)
        if name == "builds":
            raise ValueError("suite id 'builds' is reserved for --through-suite builds")
        if name in suites:
            raise ValueError(f"duplicate suite: {name}")
        suites.add(name)
        if "build" in suite and (not isinstance(suite["build"], str) or suite["build"] not in builds):
            raise ValueError(f"suite {name}: unknown build")
        if "gpu" in suite and type(suite["gpu"]) is not bool:
            raise ValueError(f"suite {name}: gpu must be boolean")
        if not isinstance(suite.get("cases"), list) or not suite["cases"]:
            raise ValueError(f"suite {name}: cases must be a nonempty array")
        for case in suite["cases"]:
            fields(case, COMMON | {"id", "argv", "depends_on"}, "case")
            key = name + "/" + identifier(case)
            if key in cases:
                raise ValueError(f"duplicate case: {key}")
            argv(case.get("argv"))
            deps = case.get("depends_on", [])
            if not isinstance(deps, list) or any(not isinstance(dep, str) or dep not in cases for dep in deps):
                raise ValueError(f"case {key}: depends_on must reference earlier cases")
            cases.add(key)
    if "health_check" in data:
        fields(data["health_check"], {"argv", "cwd", "env", "timeout_s"}, "health_check")
        argv(data["health_check"].get("argv"))
    return data


class Campaign:
    def __init__(self, manifest, output, resume=False, deadline_epoch=None):
        if deadline_epoch is not None and (
            type(deadline_epoch) not in (int, float)
            or not math.isfinite(deadline_epoch) or deadline_epoch <= 0
        ):
            raise ValueError("--deadline-epoch must be finite and positive")
        self.deadline_epoch = deadline_epoch
        self.path = Path(manifest).resolve()
        self.data = validate_manifest(json.loads(self.path.read_text(encoding="utf-8")))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.results = {}
        self.healthy = True
        self.termination_failed = False

    def record(self, event, kind, key, **details):
        row = dict(event=event, kind=kind, id=key, time_ns=time.time_ns(), **details)
        line = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        self.journal.write(line)
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.log.write(line)
        self.log.flush()
        print(line, end="", flush=True)
        if event == "DONE":
            self.results[kind, key] = row
            if row.get("termination_error"):
                self.termination_failed = True
            if kind == "case" and row["gpu"] and row["status"] in {"CRASH", "TIMEOUT"}:
                self.healthy = False
            elif kind == "health":
                self.healthy = row["status"] == "PASS" and not self.termination_failed
        return row

    def settings(self, item, parent=None):
        parent = parent or {}
        return dict(cwd=str(self.path.parent / item.get("cwd", parent.get("cwd", "."))),
                    env={**os.environ, **parent.get("env", {}), **item.get("env", {})},
                    timeout_s=item.get("timeout_s", parent.get("timeout_s", 300)))

    def execute(self, argv, settings, log):
        """Bound the whole command group, including children retaining log handles."""
        log.write((json.dumps({"argv": argv, "cwd": settings["cwd"]}) + "\n").encode())
        log.flush()
        started = time.monotonic()
        termination_error = None
        if self.deadline_epoch is not None and time.time() >= self.deadline_epoch:
            result = {"status": "BLOCKED_BY_DEPENDENCY", "reason": "campaign deadline",
                      "session.wall_ms": (time.monotonic() - started) * 1000}
            log.write((json.dumps(result) + "\n").encode())
            log.flush()
            return result
        try:
            proc = subprocess.Popen(argv, cwd=settings["cwd"], env=settings["env"],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.write((f"launch failed: {exc}\n").encode())
            log.flush()
            return {"status": "CRASH", "error": str(exc), "returncode": None,
                    "session.wall_ms": (time.monotonic() - started) * 1000}
        try:
            try:
                timeout = settings["timeout_s"]
                if self.deadline_epoch is not None:
                    timeout = min(timeout, max(0, self.deadline_epoch - time.time()))
                code = proc.wait(timeout=timeout)
                status = "PASS" if code == 0 else "CRASH" if code < 0 else "INCORRECT"
            except subprocess.TimeoutExpired:
                status = "TIMEOUT"
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The group exited between the timeout and the signal.
                try:
                    proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass  # The unconditional group SIGKILL below completes termination.
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # The command group has already exited.
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                termination_error = "process did not exit within 1s after group SIGKILL"
        result = {"status": status, "returncode": proc.returncode,
                  "session.wall_ms": (time.monotonic() - started) * 1000}
        if termination_error:
            result["termination_error"] = termination_error
        log.write((json.dumps(result) + "\n").encode())
        log.flush()
        return result

    def run_item(self, kind, key, commands, settings, required=True, gpu=False):
        if (kind, key) in self.results:
            return
        log_path = self.output / kind / (key + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.record("START", kind, key, required=required, gpu=gpu, log=str(log_path))
        started = time.monotonic()
        with log_path.open("ab", buffering=0) as log:
            for command in commands:
                result = self.execute(command, settings, log)
                if result["status"] != "PASS":
                    break
        if kind == "build" and result["status"] not in {"PASS", "BLOCKED_BY_DEPENDENCY"}:
            result["reason"] = result["status"]
            result["status"] = "BUILD_FAILED"
        result["session.wall_ms"] = (time.monotonic() - started) * 1000
        self.record("DONE", kind, key, required=required, gpu=gpu, log=str(log_path), **result)

    def restore(self, fingerprint):
        journal = self.output / "events.jsonl"
        if not journal.exists():
            if self.resume:
                raise ValueError("--resume requires an existing events.jsonl")
            return []
        if not self.resume:
            raise ValueError("events.jsonl already exists; use --resume or a new output directory")
        rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        if not rows or rows[0] != {"event": "MANIFEST", "sha256": fingerprint}:
            raise ValueError("resume manifest differs from the recorded manifest")
        pending = {}
        for row in rows[1:]:
            key = row["kind"], row["id"]
            if row["event"] == "START":
                pending[key] = row
            elif row["event"] == "DONE" and row["status"] in STATUSES:
                pending.pop(key, None)
                self.results[key] = row
                if row.get("termination_error"):
                    self.termination_failed = True
                if row["kind"] == "case" and row["gpu"] and row["status"] in {"CRASH", "TIMEOUT"}:
                    self.healthy = False
                elif row["kind"] == "health":
                    self.healthy = row["status"] == "PASS" and not self.termination_failed
            else:
                raise ValueError("invalid resume journal event")
        return list(pending.values())

    def run(self, through_suite=None):
        started = time.monotonic()
        suites = self.data["suites"]
        if through_suite == "builds":
            suites = []
        elif through_suite is not None:
            names = [suite["id"] for suite in suites]
            if through_suite not in names:
                raise ValueError(f"unknown --through-suite: {through_suite}")
            suites = suites[:names.index(through_suite) + 1]
        # A second controller must not corrupt the journal or overlap its cases.
        with (self.output / "campaign.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fingerprint = hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()
            pending = self.restore(fingerprint)
            with (self.output / "events.jsonl").open("a", encoding="utf-8") as self.journal, \
                    (self.output / "controller.log").open("a", encoding="utf-8") as self.log:
                if not self.resume:
                    line = json.dumps({"event": "MANIFEST", "sha256": fingerprint}) + "\n"
                    self.journal.write(line)
                    self.journal.flush()
                    os.fsync(self.journal.fileno())
                for row in pending:
                    self.record("DONE", row["kind"], row["id"],
                                status="BUILD_FAILED" if row["kind"] == "build" else "CRASH",
                                required=row["required"], gpu=row["gpu"],
                                reason="controller interrupted after START; no implicit retry",
                                **{"session.wall_ms": 0})
                for build in self.data["builds"]:
                    key = build["id"]
                    if ("build", key) in self.results:
                        continue
                    failed = [dep for dep in build.get("depends_on", [])
                              if self.results["build", dep]["status"] != "PASS"]
                    if failed:
                        required = build.get("required", True)
                        self.record("START", "build", key, required=required, gpu=False)
                        self.record("DONE", "build", key, status="BLOCKED_BY_DEPENDENCY",
                                    required=required, gpu=False, dependencies=failed,
                                    **{"session.wall_ms": 0})
                        continue
                    self.run_item("build", build["id"], build["commands"], self.settings(build),
                                  required=build.get("required", True))
                for suite in suites:
                    for case in suite["cases"]:
                        key = suite["id"] + "/" + case["id"]
                        if ("case", key) in self.results:
                            continue
                        required = case.get("required", suite.get("required", True))
                        gpu = suite.get("gpu", False)
                        dependencies = [("case", dep) for dep in case.get("depends_on", [])]
                        if "build" in suite:
                            dependencies.append(("build", suite["build"]))
                        failed = [dep[1] for dep in dependencies if self.results[dep]["status"] != "PASS"]
                        if gpu and (not self.healthy or self.termination_failed):
                            failed.append("device health")
                        if failed:
                            self.record("START", "case", key, required=required, gpu=gpu)
                            self.record("DONE", "case", key, status="BLOCKED_BY_DEPENDENCY",
                                        required=required, gpu=gpu, dependencies=failed,
                                        **{"session.wall_ms": 0})
                            continue
                        self.run_item("case", key, [case["argv"]], self.settings(case, suite), required, gpu)
                        if gpu and self.results["case", key]["status"] in {"CRASH", "TIMEOUT"}:
                            if "health_check" in self.data:
                                check = self.data["health_check"]
                                self.run_item("health", key, [check["argv"]],
                                              self.settings({"timeout_s": 10, **check}), required=False)
                results = list(self.results.values())
                failed = any(row["required"] and row["status"] != "PASS" for row in results)
                expected = {("build", build["id"]) for build in self.data["builds"]}
                expected.update(("case", suite["id"] + "/" + case["id"])
                                for suite in self.data["suites"] for case in suite["cases"])
                summary = {"status": "FAIL" if failed else "PASS", "results": results,
                           "complete": expected <= self.results.keys(), "through_suite": through_suite,
                           "counts": dict(collections.Counter(row["status"] for row in results)),
                           "session.wall_ms": (time.monotonic() - started) * 1000}
                temporary = self.output / "summary.json.tmp"
                temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                temporary.replace(self.output / "summary.json")
                return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--through-suite", help="run through builds or a suite ID, then persist an interim summary")
    parser.add_argument("--deadline-epoch", type=float, help="absolute UNIX seconds deadline shared by all phases")
    parser.add_argument("--log", action="store_true", required=True, help="enable persisted controller and child logs")
    args = parser.parse_args()
    try:
        controller = Campaign(args.manifest, args.output_dir, args.resume, args.deadline_epoch)
        if args.validate_only:
            line = json.dumps({"event": "VALIDATED", "manifest": str(controller.path),
                               "session.wall_ms": 0}) + "\n"
            with (controller.output / "controller.log").open("a", encoding="utf-8") as log:
                log.write(line)
            print(line, end="", flush=True)
            return 0
        return controller.run(args.through_suite)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        message = f"campaign error: {exc}\n"
        print(message, file=sys.stderr, end="", flush=True)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "controller.log").open("a", encoding="utf-8") as log:
            log.write(message)
        return 2


if __name__ == "__main__":
    sys.exit(main())
