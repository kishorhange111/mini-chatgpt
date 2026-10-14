"""The numbers quoted in the README must match the raw result files in results/.

This guards against the README drifting from the data (a typo, or a re-run
whose numbers were not copied over).
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
BENCH = json.loads((ROOT / "results" / "benchmarks.json").read_text())
DPO = json.loads((ROOT / "results" / "dpo_summary.json").read_text())

# README row label -> metric key in benchmarks.json
ROWS = {
    "ARC-Easy": "arc_easy/acc_norm",
    "ARC-Challenge": "arc_challenge/acc_norm",
    "IFEval — instruction-level strict": "ifeval/inst_level_strict_acc",
    "IFEval — prompt-level strict": "ifeval/prompt_level_strict_acc",
    "HellaSwag": "hellaswag/acc_norm",
    "Winogrande": "winogrande/acc",
    "MMLU": "mmlu/acc",
}
COLUMNS = ["base", "sft", "dpo", "reference"]


def table_row(label):
    """Return the numeric cells of the README table row starting with `label`."""
    for line in README.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return [float(c.replace("*", "")) for c in cells[1:5]]  # the 5th cell is the CI half-width
    raise AssertionError(f"row {label!r} not found in README")


@pytest.mark.parametrize("label,key", ROWS.items())
def test_benchmark_table_matches_results_json(label, key):
    expected = [round(100 * BENCH[model][key], 1) for model in COLUMNS]
    assert table_row(label) == expected


def test_dpo_reward_accuracy_matches_summary():
    before = round(100 * DPO["eval_before"]["reward_accuracy"], 1)
    after = round(100 * DPO["eval_after"]["reward_accuracy"], 1)
    assert re.search(rf"Reward accuracy.*\|\s*{before} %.*\|\s*\*\*{after} %\*\*", README)


def test_dpo_reward_margin_matches_summary():
    before = round(DPO["eval_before"]["reward_margin"], 3)
    after = round(DPO["eval_after"]["reward_margin"], 3)
    assert re.search(rf"Mean reward margin.*\|\s*{before}\s*\|\s*\*\*{after}\*\*", README)
