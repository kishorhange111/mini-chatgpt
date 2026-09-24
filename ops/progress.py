# Show the tail of the training log and GPU usage.
import subprocess

print(subprocess.run("tail -c 3000 /content/train.log", shell=True, capture_output=True, text=True).stdout)
print(subprocess.run(
    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader",
    shell=True, capture_output=True, text=True,
).stdout)
print("running" if subprocess.run("pgrep -f 'python train.py'", shell=True).returncode == 0 else "NOT running")
