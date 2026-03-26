#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args, timeout=180):
    completed = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr


def assert_ok(name, code, stdout, stderr, required_text=None):
    if code != 0:
        raise RuntimeError(
            f"{name} failed with exit code {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    if required_text and required_text not in stdout:
        raise RuntimeError(
            f"{name} output missing expected text: {required_text}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def main():
    python = sys.executable

    code, out, err = run_cmd(
        [
            python,
            "main.py",
            "--quickstart",
            "--question",
            "What is retrieval augmented generation?",
        ]
    )
    assert_ok("Quickstart single question", code, out, err, required_text="retrieval")

    code, out, err = run_cmd([python, "evaluate.py", "--quickstart"])
    assert_ok("Quickstart evaluation", code, out, err, required_text="avg_semantic_similarity")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
