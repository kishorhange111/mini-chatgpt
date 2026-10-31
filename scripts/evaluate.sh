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
#
# Usage:  bash scripts/evaluate.sh               # all four models
#         ONLY="sft dpo" bash scripts/evaluate.sh  # a subset
set -u                       # fail on undefined variables
OUT=/content/eval
mkdir -p $OUT
pip install -q "lm_eval[ifeval]" 2>&1 | tail -1
# Multiple-choice tasks: scored by comparing log-likelihoods of the answer options (no generation).
MC="mmlu,arc_easy,arc_challenge,hellaswag,winogrande"

# run NAME PATH CHAT - evaluate one model. CHAT=1 wraps prompts in the model's chat template,
# which is how a chat model is actually used; the base model has no template, so CHAT=0.
run() {
  name=$1; path=$2; chat=$3
  extra=""; [ "$chat" = 1 ] && extra="--apply_chat_template"
  lm_eval --model hf --model_args pretrained=$path,dtype=bfloat16 --tasks $MC --batch_size auto \
          --output_path $OUT/$name --log_samples > $OUT/$name.mc.log 2>&1
  # IFEval is generative (the model writes an answer that is checked against rules), so it gets
  # its own call with a fixed batch size and a generation length cap.
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
# Merge every results_*.json into $OUT/summary.json (= results/benchmarks.json).
# The logic lives in src/mini_chatgpt/eval_summary.py so it is unit-tested.
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python -m mini_chatgpt.eval_summary "$OUT"
echo "EVAL DONE"
