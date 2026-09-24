# (runs on the VM) Package the most valuable artefact for download, in 200 MB pieces:
#   final DPO model  >  final SFT model  >  newest training checkpoint.
# Prints "BACKUP <name> <piece> <piece> ..." or "NONE <reason>".
import glob
import os
import subprocess

STAGE = "/content/backup"
candidates = []
if os.path.isfile("/content/dpo/final/config.json") and os.path.exists("/content/dpo/dpo_summary.json"):
    candidates.append(("final-dpo", "/content/dpo", "final"))
if os.path.isfile("/content/out/final/config.json") and os.path.exists("/content/out/summary.json"):
    candidates.append(("final-sft", "/content/out", "final"))
ckpts = sorted(glob.glob("/content/out/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
if ckpts:
    candidates.append((os.path.basename(ckpts[-1]), "/content/out", os.path.basename(ckpts[-1])))
if not candidates:
    print("NONE nothing-to-back-up-yet")
else:
    name, root, sub = candidates[0]
    done = f"{STAGE}/{name}.ready"
    if not os.path.exists(done):
        subprocess.run(f"rm -rf {STAGE} && mkdir -p {STAGE} && "
                       f"tar -cf - -C {root} --exclude=optimizer.pt {sub} $( [ -f {root}/summary.json ] && echo summary.json ) "
                       f"$( [ -f {root}/dpo_summary.json ] && echo dpo_summary.json ) | split -b 200M - {STAGE}/{name}.tar.part_ && "
                       f"touch {done}", shell=True, check=True)
    parts = sorted(os.path.basename(p) for p in glob.glob(f"{STAGE}/{name}.tar.part_*"))
    print("BACKUP", name, *parts)
