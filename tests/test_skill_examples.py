"""Smoke tests for the runnable examples of the cavity-calculations skill.

Every .py file in .claude/skills/cavity-calculations/examples/ is executed; the examples end
with asserts on physical quantities, so an API change that breaks or silently distorts an
example fails here. New example files are picked up automatically.
"""

import runpy
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)  # plt.show() must not block/open windows during tests
import matplotlib.pyplot as plt

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[1]
    / ".claude"
    / "skills"
    / "cavity-calculations"
    / "examples"
)
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


def test_examples_dir_is_populated():
    assert EXAMPLE_SCRIPTS, f"no example scripts found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_skill_example_runs(script):
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        plt.close("all")
