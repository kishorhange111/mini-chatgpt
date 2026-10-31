"""Collecting lm-evaluation-harness output files into one summary."""

import json

from mini_chatgpt.eval_summary import extract_metrics, main, summarize


def test_extract_metrics_keeps_headline_scores_only():
    results = {
        "arc_easy": {"alias": "arc_easy", "acc,none": 0.6, "acc_stderr,none": 0.01, "acc_norm,none": 0.58},
        "ifeval": {
            "prompt_level_strict_acc,none": 0.24,
            "prompt_level_loose_acc,none": 0.3,
            "inst_level_strict_acc_stderr,none": 0.02,
        },
        "mmlu": {"acc,none": 0.47, "samples": 100},
    }
    assert extract_metrics(results) == {
        "arc_easy/acc": 0.6,
        "arc_easy/acc_norm": 0.58,
        "ifeval/prompt_level_strict_acc": 0.24,
        "mmlu/acc": 0.47,
    }


def write_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": results}))


def test_summarize_merges_nested_result_files(tmp_path):
    # scripts/evaluate.sh writes multiple-choice and IFEval runs as separate files under each model folder.
    write_results(tmp_path / "sft" / "x" / "results_1.json", {"mmlu": {"acc,none": 0.45}})
    write_results(tmp_path / "sft" / "y" / "results_2.json", {"ifeval": {"inst_level_strict_acc,none": 0.36}})
    write_results(tmp_path / "base" / "results_3.json", {"mmlu": {"acc,none": 0.47}})
    rows = summarize(str(tmp_path), names=["base", "sft", "dpo"])
    assert rows == {
        "base": {"mmlu/acc": 0.47},
        "sft": {"mmlu/acc": 0.45, "ifeval/inst_level_strict_acc": 0.36},
        "dpo": {},
    }


def test_main_writes_summary_json(tmp_path, capsys):
    write_results(tmp_path / "dpo" / "results_1.json", {"winogrande": {"acc,none": 0.557}})
    main([str(tmp_path)])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["dpo"] == {"winogrande/acc": 0.557}
    assert '"winogrande/acc": 0.557' in capsys.readouterr().out
