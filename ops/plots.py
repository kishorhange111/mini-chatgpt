# (runs on the LLM VM) Training curves + benchmark chart for the README.
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("/content/plots", exist_ok=True)
sft = [r for r in json.load(open("/content/out/log_history.json")) if "loss" in r and "step" in r]
dpo = json.load(open("/content/dpo/dpo_log.json"))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot([r["step"] for r in sft], [r["loss"] for r in sft], lw=1)
axes[0].set_title("SFT: training loss (assistant tokens)"); axes[0].set_xlabel("step"); axes[0].grid(alpha=.3)
w = 10
acc = [sum(r["reward_accuracy"] for r in dpo[max(0, i - w):i + 1]) / len(dpo[max(0, i - w):i + 1]) for i in range(len(dpo))]
axes[1].plot([r["step"] for r in dpo], acc, lw=1.5, color="#2a7ab9")
axes[1].axhline(0.5, ls="--", color="gray", lw=1)
axes[1].set_title("DPO: reward accuracy (chosen > rejected), moving avg"); axes[1].set_xlabel("step"); axes[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig("/content/plots/training_curves.png", dpi=110)

b = json.load(open("/content/eval/summary.json"))
tasks = [("mmlu/acc", "MMLU"), ("arc_easy/acc_norm", "ARC-E"), ("arc_challenge/acc_norm", "ARC-C"),
         ("hellaswag/acc_norm", "HellaSwag"), ("winogrande/acc", "Winogrande"), ("ifeval/inst_level_strict_acc", "IFEval (inst)")]
models = [("base", "Qwen2.5-0.5B base", "#bbb"), ("sft", "ours: SFT", "#7fb3d5"), ("dpo", "ours: SFT+DPO", "#2a7ab9"),
          ("reference", "Qwen2.5-0.5B-Instruct", "#e39c4b")]
fig, ax = plt.subplots(figsize=(11, 3.8))
for i, (k, label, color) in enumerate(models):
    ax.bar([j + (i - 1.5) * 0.2 for j in range(len(tasks))], [100 * b[k].get(t, 0) for t, _ in tasks], 0.2, label=label, color=color)
ax.set_xticks(range(len(tasks)), [n for _, n in tasks]); ax.set_ylabel("score (%)"); ax.legend(fontsize=8, ncol=4)
ax.set_title("Benchmarks: base -> SFT -> DPO vs. Qwen's official instruct model")
fig.tight_layout(); fig.savefig("/content/plots/benchmarks.png", dpi=110)
print("plots saved")
