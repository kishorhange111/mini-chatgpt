#!/bin/bash
# Stage 4 of Mini-ChatGPT: evaluate every stage of the pipeline on standard benchmarks with
# EleutherAI's lm-evaluation-harness (the tool behind the Open LLM Leaderboard).
#
#   base      Qwen/Qwen2.5-0.5B              (pre-trained only)
#   sft       /content/out/final             (our SFT on 160k SmolTalk chats)
#   dpo       /content/dpo/final             (our DPO on UltraFeedback preferences)
#   reference Qwen/Qwen2.5-0.5B-Instruct     (Qwen's own chat model: SFT + RLHF on far more data)
#
# Tasks: knowledge (MMLU), reasoning/commonsense (ARC-Easy, ARC-Challenge, HellaSwag, Winogrande)
# and instruction following (IFEval). Chat models are prompted with their chat template.
set -u
OUT=/content/eval
mkdir -p $OUT
pip install -q "lm_eval[ifeval]" 2>&1 | tail -1
MC="mmlu,arc_easy,arc_challenge,hellaswag,winogrande"
run() {  # name path chat
  name=$1; path=$2; chat=$3
  extra=""; [ "$chat" = 1 ] && extra="--apply_chat_template"
  lm_eval --model hf --model_args pretrained=$path,dtype=bfloat16 --tasks $MC --batch_size auto \
          --output_path $OUT/$name --log_samples > $OUT/$name.mc.log 2>&1
  lm_eval --model hf --model_args pretrained=$path,dtype=bfloat16 --tasks ifeval --batch_size 32 \
          $extra --gen_kwargs max_gen_toks=512 --output_path $OUT/$name > $OUT/$name.ifeval.log 2>&1
  echo "$(date -u +%T) evaluated $name"
}
for m in ${ONLY:-base sft dpo reference}; do
  case $m in
    base) run base Qwen/Qwen2.5-0.5B 0 ;;
    sft) run sft /content/out/final 1 ;;
    dpo) run dpo /content/dpo/final 1 ;;
    reference) run reference Qwen/Qwen2.5-0.5B-Instruct 1 ;;
  esac
done
python - <<'EOF'
import glob, json, os
rows = {}
for name in ["base", "sft", "dpo", "reference"]:
    res = {}
    for f in glob.glob(f"/content/eval/{name}/**/results_*.json", recursive=True):
        for task, m in json.load(open(f))["results"].items():
            for k, v in m.items():
                if isinstance(v, float) and "stderr" not in k and (k.startswith("acc") or "strict" in k):
                    res[f"{task}/{k.split(',')[0]}"] = v
    rows[name] = res
json.dump(rows, open("/content/eval/summary.json", "w"), indent=2)
print(json.dumps(rows, indent=2))
EOF
echo "EVAL DONE"
