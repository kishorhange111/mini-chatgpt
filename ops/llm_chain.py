# (runs on "llm") After SFT finishes: DPO -> benchmark evaluation (base / SFT / DPO / Qwen-Instruct) -> chat samples.
import subprocess

rows = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout.splitlines()[1:]
sft = [r.split()[0] for r in rows if r.rstrip().endswith(" train.py") and "bash" not in r and "/bin/sh" not in r]
wait = f"while kill -0 {sft[0]} 2>/dev/null; do sleep 30; done\n" if sft else ""
open("/content/llm_chain.sh", "w").write(
    wait +
    "cd /content && [ -d /content/out/final ] || { echo 'SFT final model missing'; exit 1; }\n"
    "cd /content && python dpo.py > dpo.log 2>&1\n"
    "cd /content && sed -i 's/\\r//' eval.sh && bash eval.sh > eval.log 2>&1\n")
subprocess.Popen("nohup bash /content/llm_chain.sh > /content/llm_chain.out 2>&1 &", shell=True)
print("llm chain queued" + (f" after SFT pid {sft[0]}" if sft else " (SFT not running!)"))
