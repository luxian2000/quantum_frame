import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


QAS_ROOT = Path(__file__).resolve().parents[1]
RUNNER = QAS_ROOT / "demos" / "run_ch4_18q_lineb_npu4.sh"


def _bash_executable():
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
        if git_bash.is_file():
            return str(git_bash)
    return shutil.which("bash")


def _shell_path(path):
    resolved = Path(path).resolve()
    if os.name != "nt":
        return str(resolved)
    posix = resolved.as_posix()
    return f"/{posix[0].lower()}{posix[2:]}"


class Ch4LineBRunnerTests(unittest.TestCase):
    @unittest.skipUnless(_bash_executable(), "bash is required for the CH4 runner test")
    def test_resume_mode_copies_completed_labels_without_invoking_p0(self):
        with tempfile.TemporaryDirectory(dir=QAS_ROOT) as tmp:
            root = Path(tmp)
            source = root / "completed_labels.csv"
            source.write_text("architecture_id,fair_best_energy\np0,-53.2\n", encoding="utf-8")
            output_dir = root / "resume_output"
            env = os.environ.copy()
            env.update(
                {
                    "HAM_PATH": "unused-in-resume-mode.json",
                    "OUT_DIR": _shell_path(output_dir),
                    "P1_BOOTSTRAP_LABELS_CSV": _shell_path(source),
                    "PYTHON": "python-command-that-must-not-run",
                    "ROUNDS": "0",
                    "SKIP_P0": "1",
                }
            )

            completed = subprocess.run(
                [_bash_executable(), _shell_path(RUNNER)],
                cwd=QAS_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            self.assertEqual(
                (output_dir / "current_labeled_rows.csv").read_text(encoding="utf-8"),
                source.read_text(encoding="utf-8"),
            )
            self.assertFalse((output_dir / "chemistry_excitation_bootstrap_queue.csv").exists())
            self.assertIn("p0=skipped", completed.stdout)

    @unittest.skipUnless(_bash_executable(), "bash is required for the CH4 runner test")
    def test_resume_mode_accepts_an_alias_of_current_labels(self):
        with tempfile.TemporaryDirectory(dir=QAS_ROOT) as tmp:
            root = Path(tmp)
            output_dir = root / "resume_output"
            output_dir.mkdir()
            current_labels = output_dir / "current_labeled_rows.csv"
            current_labels.write_text("architecture_id,fair_best_energy\np0,-53.2\n", encoding="utf-8")
            aliased_labels = f"{_shell_path(output_dir)}/../{output_dir.name}/current_labeled_rows.csv"
            env = os.environ.copy()
            env.update(
                {
                    "HAM_PATH": "unused-in-resume-mode.json",
                    "OUT_DIR": _shell_path(output_dir),
                    "P1_BOOTSTRAP_LABELS_CSV": aliased_labels,
                    "PYTHON": "python-command-that-must-not-run",
                    "ROUNDS": "0",
                    "SKIP_P0": "1",
                }
            )

            completed = subprocess.run(
                [_bash_executable(), _shell_path(RUNNER)],
                cwd=QAS_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            self.assertIn("p0=skipped", completed.stdout)


if __name__ == "__main__":
    unittest.main()
