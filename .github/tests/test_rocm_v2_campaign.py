"""CPU-only campaign tests; no compiler, model, device or measurement is used.

Run: python .github/tests/test_rocm_v2_campaign.py --log
The controller and each synthetic command retain logs in temporary directories.
The test runner's output can be redirected to a persistent CI log.
"""

import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CONTROLLER = Path(__file__).with_name("rocm_v2_campaign.py").resolve()
spec = importlib.util.spec_from_file_location("rocm_v2_campaign", CONTROLLER)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rocm-campaign-cpu-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "results"

    @staticmethod
    def command(code):
        return [sys.executable, "-c", code, "--log"]

    def case(self, name, code="print('ok')", **options):
        return dict(id=name, argv=self.command(code), **options)

    def build(self, name, code="print('built')", **options):
        return dict(id=name, build_type="Debug", commands=[self.command(code)], **options)

    @staticmethod
    def manifest_data(builds=None, suites=None, **options):
        return dict(version=1, builds=builds or [], suites=suites or [], **options)

    def run_campaign(self, data=None, resume=False, through_suite=None, validate_only=False, deadline_epoch=None):
        if data is not None:
            self.manifest.write_text(json.dumps(data), encoding="utf-8")
        argv = [sys.executable, str(CONTROLLER), "--manifest", str(self.manifest),
                "--output-dir", str(self.output), "--log"]
        if resume:
            argv.append("--resume")
        if through_suite is not None:
            argv.extend(["--through-suite", through_suite])
        if validate_only:
            argv.append("--validate-only")
        if deadline_epoch is not None:
            argv.append(f"--deadline-epoch={deadline_epoch}")
        result = subprocess.run(argv, cwd=self.root, capture_output=True, text=True, timeout=15)
        with (self.root / "invocations.log").open("a", encoding="utf-8") as log:
            log.write(result.stdout + result.stderr)
        return result

    def events(self):
        return [json.loads(line) for line in (self.output / "events.jsonl").read_text().splitlines()]

    def statuses(self, kind="case"):
        return {row["id"]: row["status"] for row in self.events()
                if row["event"] == "DONE" and row["kind"] == kind}

    def test_failed_build_isolated_and_all_builds_before_cases(self):
        data = self.manifest_data(
            builds=[self.build("bad", "import sys; print('compile failed'); sys.exit(2)"),
                    self.build("good", "from pathlib import Path; Path('built').touch()")],
            suites=[dict(id="bad-suite", build="bad", cases=[self.case("blocked")]),
                    dict(id="good-suite", build="good", cases=[self.case(
                        "works", "from pathlib import Path; assert Path('built').exists()")])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses("build"), {"bad": "BUILD_FAILED", "good": "PASS"})
        self.assertEqual(self.statuses(), {"bad-suite/blocked": "BLOCKED_BY_DEPENDENCY",
                                          "good-suite/works": "PASS"})
        events = self.events()[1:]
        last_build = max(i for i, row in enumerate(events) if row["kind"] == "build")
        first_case = min(i for i, row in enumerate(events) if row["kind"] == "case")
        self.assertLess(last_build, first_case)
        self.assertIn("compile failed", (self.output / "build/bad.log").read_text())

    def test_build_timeout_and_failed_command_stop_only_their_build(self):
        broken = self.build("broken", "raise SystemExit(1)")
        broken["commands"].append(self.command("from pathlib import Path; Path('should-not-run').touch()"))
        data = self.manifest_data(builds=[
            self.build("hang", "import time; time.sleep(30)", timeout_s=0.1), broken,
            self.build("good")], suites=[dict(id="cpu", build="good", cases=[self.case("works")])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses("build"), {
            "hang": "BUILD_FAILED", "broken": "BUILD_FAILED", "good": "PASS"})
        self.assertEqual(self.statuses(), {"cpu/works": "PASS"})
        self.assertFalse((self.root / "should-not-run").exists())

    def test_failed_ggml_blocks_framework_but_independent_vulkan_runs(self):
        data = self.manifest_data(builds=[
            self.build("ggml", "raise SystemExit(1)"),
            self.build("framework", "from pathlib import Path; Path('framework-built').touch()",
                       depends_on=["ggml"]),
            self.build("vulkan", "from pathlib import Path; Path('vulkan-built').touch()")],
            suites=[dict(id="ggml-suite", build="ggml", cases=[self.case("blocked")]),
                    dict(id="framework-suite", build="framework", cases=[self.case("blocked")]),
                    dict(id="vulkan-suite", build="vulkan", cases=[self.case(
                        "works", "from pathlib import Path; assert Path('vulkan-built').exists()")])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses("build"), {"ggml": "BUILD_FAILED",
            "framework": "BLOCKED_BY_DEPENDENCY", "vulkan": "PASS"})
        self.assertEqual(self.statuses(), {"ggml-suite/blocked": "BLOCKED_BY_DEPENDENCY",
            "framework-suite/blocked": "BLOCKED_BY_DEPENDENCY", "vulkan-suite/works": "PASS"})
        self.assertFalse((self.root / "framework-built").exists())
        self.assertTrue((self.root / "vulkan-built").exists())
        events = self.events()[1:]
        framework = [row for row in events if row["kind"] == "build" and row["id"] == "framework"]
        self.assertEqual([row["event"] for row in framework], ["START", "DONE"])
        self.assertEqual(framework[-1]["dependencies"], ["ggml"])
        self.assertLess(max(i for i, row in enumerate(events) if row["kind"] == "build"),
                        min(i for i, row in enumerate(events) if row["kind"] == "case"))
        before = (self.output / "events.jsonl").read_bytes()
        self.assertEqual(self.run_campaign(resume=True).returncode, 1)
        self.assertEqual((self.output / "events.jsonl").read_bytes(), before)

    def test_passing_build_dependency_allows_dependent_commands(self):
        data = self.manifest_data(builds=[self.build("ggml"), self.build("framework",
            "from pathlib import Path; Path('framework-built').touch()", depends_on=["ggml"])],
            suites=[dict(id="framework-suite", build="framework", cases=[self.case("works")])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.statuses("build"), {"ggml": "PASS", "framework": "PASS"})
        self.assertTrue((self.root / "framework-built").exists())

    def test_crash_incorrect_launch_error_and_independent_success(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[
            self.case("crash", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
            self.case("wrong", "raise SystemExit(1)"),
            dict(id="missing", argv=[str(self.root / "missing-executable"), "--log"]),
            self.case("good"),
        ])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses(), {"cpu/crash": "CRASH", "cpu/wrong": "INCORRECT",
                                          "cpu/missing": "CRASH", "cpu/good": "PASS"})
        for key in self.statuses():
            pair = [row["event"] for row in self.events() if row.get("id") == key]
            self.assertEqual(pair, ["START", "DONE"])
        for row in self.events():
            if row["event"] == "DONE":
                self.assertIn("session.wall_ms", row)

    def test_timeout_kills_group_and_independent_success(self):
        # The grandchild ignores SIGTERM and would create a marker if left alive.
        grandchild = "import time; from pathlib import Path; time.sleep(0.9); Path('escaped').touch()"
        parent = ("import subprocess, sys, signal, time; "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  f"subprocess.Popen([sys.executable, '-c', {grandchild!r}, '--log']); "
                  "print('before timeout', flush=True); time.sleep(30)")
        data = self.manifest_data(suites=[dict(id="cpu", cases=[
            self.case("hang", parent, timeout_s=0.3),
            self.case("independent", "import time; time.sleep(1)"),
        ])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses(), {"cpu/hang": "TIMEOUT", "cpu/independent": "PASS"})
        self.assertFalse((self.root / "escaped").exists(), "timed-out grandchild survived")
        self.assertIn("before timeout", (self.output / "case/cpu/hang.log").read_text())

    def test_unreapable_process_has_bounded_wait_and_explicit_error(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[self.case("first")])])
        self.manifest.write_text(json.dumps(data))
        controller = campaign.Campaign(self.manifest, self.output)
        process = mock.Mock(pid=12345, returncode=None)
        process.wait.side_effect = subprocess.TimeoutExpired("synthetic", 1)
        with mock.patch.object(campaign.subprocess, "Popen", return_value=process), \
                mock.patch.object(campaign.os, "killpg"):
            result = controller.execute(["synthetic", "--log"],
                                        controller.settings({"timeout_s": 0.1}), io.BytesIO())
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertIn("SIGKILL", result["termination_error"])
        self.assertTrue(all("timeout" in call.kwargs for call in process.wait.call_args_list))

    def test_expired_deadline_blocks_builds_and_cases_without_children(self):
        command = "from pathlib import Path; Path('executed').touch()"
        data = self.manifest_data(builds=[self.build("compile", command)], suites=[
            dict(id="built", build="compile", cases=[self.case("blocked", command)]),
            dict(id="independent", cases=[self.case("blocked", command)])])
        result = self.run_campaign(data, deadline_epoch=1)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses("build"), {"compile": "BLOCKED_BY_DEPENDENCY"})
        self.assertEqual(self.statuses(), {"built/blocked": "BLOCKED_BY_DEPENDENCY",
                                          "independent/blocked": "BLOCKED_BY_DEPENDENCY"})
        self.assertFalse((self.root / "executed").exists())
        direct = [row for row in self.events() if row["event"] == "DONE"
                  and row["id"] in {"compile", "independent/blocked"}]
        self.assertTrue(all(row["reason"] == "campaign deadline" for row in direct))
        controller = campaign.Campaign(self.manifest, self.root / "no-children", deadline_epoch=1)
        with mock.patch.object(campaign.subprocess, "Popen") as spawn, \
                mock.patch.object(campaign.sys, "stdout", io.StringIO()):
            self.assertEqual(controller.run(), 1)
            spawn.assert_not_called()

    def test_deadline_caps_wait_and_is_checked_before_every_command(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[self.case("first")])])
        self.manifest.write_text(json.dumps(data))
        controller = campaign.Campaign(self.manifest, self.output, deadline_epoch=110)
        process = mock.Mock(pid=12345, returncode=0)
        process.wait.return_value = 0
        with mock.patch.object(campaign.subprocess, "Popen", return_value=process) as spawn, \
                mock.patch.object(campaign.os, "killpg"), \
                mock.patch.object(campaign.time, "time", side_effect=[100, 100, 111]):
            settings = controller.settings({"timeout_s": 300})
            self.assertEqual(controller.execute(["synthetic", "--log"], settings, io.BytesIO())["status"], "PASS")
            blocked = controller.execute(["synthetic", "--log"], settings, io.BytesIO())
        self.assertEqual(process.wait.call_args_list[0].kwargs["timeout"], 10)
        self.assertEqual(blocked["status"], "BLOCKED_BY_DEPENDENCY")
        self.assertEqual(blocked["reason"], "campaign deadline")
        self.assertEqual(spawn.call_count, 1)

    def test_deadline_must_be_finite_and_positive(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[self.case("first")])])
        for value in [0, -1, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(deadline=value):
                result = self.run_campaign(data, deadline_epoch=value, validate_only=True)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("--deadline-epoch must be finite and positive", result.stderr)
        self.assertEqual(self.run_campaign(data, deadline_epoch=1, validate_only=True).returncode, 0)
        self.assertFalse((self.output / "events.jsonl").exists())

    def test_gpu_fault_without_health_check_blocks_gpu_only(self):
        data = self.manifest_data(suites=[
            dict(id="gpu", gpu=True, cases=[
                self.case("crash", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
                self.case("later")]),
            dict(id="cpu", cases=[self.case("independent")])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses(), {"gpu/crash": "CRASH",
                                          "gpu/later": "BLOCKED_BY_DEPENDENCY",
                                          "cpu/independent": "PASS"})

    def test_bounded_failed_health_check_blocks_gpu(self):
        data = self.manifest_data(
            suites=[dict(id="gpu", gpu=True, cases=[
                self.case("hang", "import time; time.sleep(30)", timeout_s=0.1),
                self.case("later")])],
            health_check=dict(argv=self.command("import time; time.sleep(30)"), timeout_s=0.1))
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses("health"), {"gpu/hang": "TIMEOUT"})
        self.assertEqual(self.statuses()["gpu/later"], "BLOCKED_BY_DEPENDENCY")

    def test_passing_health_check_allows_next_gpu_case(self):
        data = self.manifest_data(
            suites=[dict(id="gpu", gpu=True, cases=[
                self.case("crash", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
                self.case("later")])],
            health_check=dict(argv=self.command("print('healthy')"), timeout_s=1))
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.statuses("health"), {"gpu/crash": "PASS"})
        self.assertEqual(self.statuses()["gpu/later"], "PASS")

    def test_case_dependencies_and_optional_failure(self):
        data = self.manifest_data(suites=[dict(id="cpu", required=False, cases=[
            self.case("wrong", "raise SystemExit(1)"),
            self.case("blocked", depends_on=["cpu/wrong"]),
            self.case("good", required=True),
        ])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.statuses()["cpu/blocked"], "BLOCKED_BY_DEPENDENCY")

    def test_exit_77_is_incorrect_and_fails_required_campaign(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[
            self.case("unsupported", "raise SystemExit(77)"), self.case("good")])])
        self.assertEqual(self.run_campaign(data).returncode, 1)
        self.assertEqual(self.statuses(), {"cpu/unsupported": "INCORRECT", "cpu/good": "PASS"})

    def test_validate_only_launches_zero_subprocesses_and_leaves_no_journal(self):
        command = "from pathlib import Path; Path('executed').touch()"
        data = self.manifest_data(builds=[self.build("compile", command)], suites=[dict(
            id="cpu", cases=[self.case("first", command)])])
        self.assertEqual(self.run_campaign(data, validate_only=True).returncode, 0)
        self.assertFalse((self.root / "executed").exists())
        self.assertFalse((self.output / "events.jsonl").exists())
        self.assertIn("VALIDATED", (self.output / "controller.log").read_text())
        with mock.patch.object(campaign.subprocess, "Popen", side_effect=AssertionError("unexpected child")), \
                mock.patch.object(sys, "argv", [str(CONTROLLER), "--manifest", str(self.manifest),
                                                "--output-dir", str(self.output), "--validate-only", "--log"]):
            self.assertEqual(campaign.main(), 0)
        self.assertEqual(self.run_campaign(data).returncode, 0)

    def test_phases_resume_to_named_boundary_even_if_already_complete(self):
        data = self.manifest_data(builds=[self.build("compile")], suites=[
            dict(id="first", cases=[self.case("wrong", "raise SystemExit(1)")]),
            dict(id="second", cases=[self.case("good")]),
            dict(id="last", cases=[self.case("good")])])
        self.assertEqual(self.run_campaign(data, through_suite="builds").returncode, 0)
        self.assertEqual(self.statuses(), {})
        self.assertFalse(json.loads((self.output / "summary.json").read_text())["complete"])
        self.assertEqual(self.run_campaign(resume=True, through_suite="first").returncode, 1)
        before = (self.output / "events.jsonl").read_bytes()
        self.assertEqual(self.run_campaign(resume=True, through_suite="first").returncode, 1)
        self.assertEqual(before, (self.output / "events.jsonl").read_bytes())
        self.assertEqual(self.run_campaign(resume=True, through_suite="second").returncode, 1)
        self.assertEqual(self.statuses(), {"first/wrong": "INCORRECT", "second/good": "PASS"})
        self.assertEqual(self.run_campaign(resume=True).returncode, 1)
        self.assertEqual(self.statuses()["last/good"], "PASS")
        self.assertTrue(json.loads((self.output / "summary.json").read_text())["complete"])

    def test_unknown_phase_rejected_before_builds(self):
        data = self.manifest_data(builds=[self.build("compile")], suites=[dict(
            id="cpu", cases=[self.case("first")])])
        self.assertEqual(self.run_campaign(data, through_suite="typo").returncode, 2)
        self.assertFalse((self.output / "events.jsonl").exists())

    def test_resume_preserves_failed_and_successful_results(self):
        data = self.manifest_data(builds=[self.build("compile")], suites=[dict(
            id="cpu", build="compile", cases=[
                self.case("wrong", "from pathlib import Path; Path('attempt').touch(); raise SystemExit(1)"),
                self.case("good")])])
        self.assertEqual(self.run_campaign(data).returncode, 1)
        before = (self.output / "events.jsonl").read_bytes()
        (self.root / "attempt").unlink()
        self.assertEqual(self.run_campaign(resume=True).returncode, 1)
        self.assertEqual((self.output / "events.jsonl").read_bytes(), before)
        self.assertFalse((self.root / "attempt").exists())

    def test_resume_runs_only_unstarted_cases(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[self.case("first"), self.case("later")])])
        self.assertEqual(self.run_campaign(data).returncode, 0)
        rows = [row for row in self.events() if row.get("id") != "cpu/later"]
        (self.output / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(self.run_campaign(resume=True).returncode, 0)
        self.assertEqual(self.statuses(), {"cpu/first": "PASS", "cpu/later": "PASS"})
        self.assertEqual(sum(row.get("id") == "cpu/first" for row in self.events()), 2)

    def test_resume_interrupted_start_is_failure_not_retry(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[
            self.case("interrupted", "from pathlib import Path; Path('attempt').touch()"),
            self.case("later")])])
        self.assertEqual(self.run_campaign(data).returncode, 0)
        rows = [row for row in self.events() if row.get("id") != "cpu/later"
                and not (row.get("id") == "cpu/interrupted" and row["event"] == "DONE")]
        (self.output / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.root / "attempt").unlink()
        self.assertEqual(self.run_campaign(resume=True).returncode, 1)
        self.assertEqual(self.statuses(), {"cpu/interrupted": "CRASH", "cpu/later": "PASS"})
        self.assertFalse((self.root / "attempt").exists())

    def test_resume_rejects_changed_manifest_and_corrupt_journal(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[self.case("first")])])
        self.assertEqual(self.run_campaign(data).returncode, 0)
        changed = copy.deepcopy(data)
        changed["suites"][0]["cases"][0]["timeout_s"] = 1
        self.assertEqual(self.run_campaign(changed, resume=True).returncode, 2)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write('{"event":')
        self.assertEqual(self.run_campaign(data, resume=True).returncode, 2)

    def test_resume_restores_failed_device_health(self):
        data = self.manifest_data(suites=[dict(id="gpu", gpu=True, cases=[
            self.case("crash", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"),
            self.case("later")])])
        self.assertEqual(self.run_campaign(data).returncode, 1)
        rows = [row for row in self.events() if row.get("id") != "gpu/later"]
        (self.output / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.assertEqual(self.run_campaign(resume=True).returncode, 1)
        self.assertEqual(self.statuses()["gpu/later"], "BLOCKED_BY_DEPENDENCY")

    def test_fresh_run_refuses_existing_journal(self):
        data = self.manifest_data(suites=[dict(id="cpu", cases=[self.case("first")])])
        self.assertEqual(self.run_campaign(data).returncode, 0)
        before = (self.output / "events.jsonl").read_bytes()
        self.assertEqual(self.run_campaign().returncode, 2)
        self.assertEqual((self.output / "events.jsonl").read_bytes(), before)

    def test_malformed_manifests_do_not_execute(self):
        valid = self.manifest_data(builds=[self.build("compile")], suites=[dict(
            id="cpu", cases=[self.case("first", "from pathlib import Path; Path('executed').touch()")])])
        mutations = [
            lambda d: d.update(version=2),
            lambda d: d.update(unknown=True),
            lambda d: d["builds"][0].update(build_type="Release"),
            lambda d: d["builds"][0].update(depends_on="compile"),
            lambda d: d["builds"][0].update(depends_on=["compile"]),
            lambda d: d["builds"][0].update(depends_on=["missing"]),
            lambda d: d["builds"].insert(0, self.build("forward", depends_on=["compile"])),
            lambda d: d["suites"][0].update(build="missing"),
            lambda d: d["suites"][0]["cases"][0].update(argv="echo unsafe"),
            lambda d: d["suites"][0]["cases"][0].update(timeout_s=0),
            lambda d: d["suites"][0]["cases"][0].update(timeout_s=float("inf")),
            lambda d: d["suites"][0]["cases"][0].update(required="false"),
            lambda d: d["suites"][0]["cases"][0].update(cwd="bad\0path"),
            lambda d: d["suites"][0]["cases"][0].update(id="../escape"),
            lambda d: d["suites"][0]["cases"][0].update(depends_on=["cpu/future"]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = copy.deepcopy(valid)
                mutate(data)
                result = self.run_campaign(data)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("campaign error:", result.stderr)
                self.assertFalse((self.root / "executed").exists())
                self.assertFalse((self.output / "events.jsonl").exists())
        self.manifest.write_text("{not json")
        self.assertEqual(self.run_campaign().returncode, 2)

    def test_argv_is_literal_and_environment_is_inherited(self):
        literal = "$(touch injected); `touch injected2`"
        command = self.command("import sys, os; from pathlib import Path; "
                               "assert os.environ['SUITE_ENV'] == 'inherited'; "
                               "assert os.environ['CASE_ENV'] == 'case'; "
                               "Path('literal').write_text(sys.argv[2])") + [literal]
        data = self.manifest_data(suites=[dict(id="cpu", env={"SUITE_ENV": "inherited"},
            cases=[dict(id="literal", argv=command, env={"CASE_ENV": "case"})])])
        result = self.run_campaign(data)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "literal").read_text(), literal)
        self.assertFalse((self.root / "injected").exists())
        self.assertFalse((self.root / "injected2").exists())


if __name__ == "__main__":
    # Accept the repository's explicit log switch; unittest writes to stderr.
    if "--log" in sys.argv:
        sys.argv.remove("--log")
    unittest.main(verbosity=2)
