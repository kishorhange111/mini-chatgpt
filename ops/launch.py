# Start train.py (and the Drive checkpoint backup loop) in the background on the VM.
# Settings come from /content/run.env.
import subprocess

subprocess.Popen(
    "cd /content && set -a && . ./run.env && set +a && "
    "(nohup bash ckpt_sync.sh > ckpt_sync.log 2>&1 &) && sleep 2 && "
    "nohup python train.py > train.log 2>&1 &",
    shell=True,
)
print("launched")
