"""
Collect lm-evaluation-harness results into one summary file.

eval.sh runs lm-evaluation-harness once per model and task group; each run
writes a results_<timestamp>.json somewhere under <eval_dir>/<model_name>/.
This module walks those files and keeps only the headline numbers (accuracy
and IFEval "strict" scores, no standard errors), producing

    {"base": {"arc_easy/acc_norm": 0.58, ...}, "sft": {...}, ...}

which is what results/benchmarks.json contains.

Usage (called by eval.sh):
    python -m mini_chatgpt.eval_summary [EVAL_DIR]      # default /content/eval
"""

import glob
import json
import os
import sys

MODEL_NAMES = ["base", "sft", "dpo", "reference"]


def extract_metrics(results):
    """Pick the headline metrics out of one lm-eval "results" section.

    Args:
        results: {task: {"acc,none": 0.5, "acc_stderr,none": 0.01, ...}}, the
            "results" field of an lm-evaluation-harness output file.

    Returns:
        {"task/metric": value}: only float metrics whose name starts with
        "acc" or contains "strict", excluding stderr entries. The ",none"
        filter suffix lm-eval adds to metric names is stripped.
    """
    out = {}
    for task, metrics in results.items():
        for k, v in metrics.items():
            if isinstance(v, float) and "stderr" not in k and (k.startswith("acc") or "strict" in k):
                out[f"{task}/{k.split(',')[0]}"] = v
    return out


def summarize(eval_dir, names=MODEL_NAMES):
    """Merge every results_*.json under eval_dir/<name>/ for each model name.

    Args:
        eval_dir: the folder eval.sh wrote to.
        names: model sub-folders to read.

    Returns:
        {name: {"task/metric": value}}.
    """
    rows = {}
    for name in names:
        res = {}
        for f in glob.glob(os.path.join(eval_dir, name, "**", "results_*.json"), recursive=True):
            with open(f) as fh:
                res.update(extract_metrics(json.load(fh)["results"]))
        rows[name] = res
    return rows


def main(argv=None):
    """Write <eval_dir>/summary.json and print it."""
    argv = sys.argv[1:] if argv is None else argv
    eval_dir = argv[0] if argv else "/content/eval"
    rows = summarize(eval_dir)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
