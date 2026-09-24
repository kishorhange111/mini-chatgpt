#!/bin/bash
# (runs locally in WSL) After a lost VM: upload the newest COMPLETE local checkpoint backup to a new
# session, unpack it into /content/out, upload the code and relaunch training (it resumes automatically).
# The Jupyter contents API rejects very large uploads (500 at 200 MB), so the checkpoint is re-split
# into 40 MB pieces first.
# Usage: bash ops/restore.sh <session>
set -e
S=${1:-trainer}
C=~/.local/bin/colab
DEST=/mnt/e/Resumes/small_chatgpt/backup
ckpt=$(ls -d $DEST/checkpoint-* | grep -v partial | while read d; do [ -f "$d/COMPLETE" ] && echo $d; done | sort -t- -k2 -n | tail -1)
name=$(basename $ckpt)
echo "restoring $name to $S"
rm -rf /tmp/restore_$S && mkdir -p /tmp/restore_$S
cat $ckpt/$name.tar.part_* | split -b 40M - /tmp/restore_$S/r.
for part in /tmp/restore_$S/r.*; do
  for try in 1 2 3; do
    timeout 300 $C upload -s $S "$part" "/content/$(basename $part)" >/dev/null 2>&1 && break
    echo "retry $(basename $part) ($try)"; sleep 5
  done
done
echo "uploaded $(ls /tmp/restore_$S | wc -l) pieces"
for f in train.py chat_test.py dpo.py eval.sh; do timeout 300 $C upload -s $S $f /content/$f; done
timeout 300 $C upload -s $S configs/full.env /content/run.env
cat > /tmp/unpack_$S.py <<EOF
import subprocess
print(subprocess.run("mkdir -p /content/out && cat /content/r.* | tar -xf - -C /content/out && rm -f /content/r.* && ls /content/out/$name",
                     shell=True, capture_output=True, text=True).stdout)
subprocess.Popen("cd /content && set -a && . ./run.env && set +a && nohup python train.py > train.log 2>&1 &", shell=True)
print("training relaunched (will resume from $name)")
EOF
timeout 300 $C exec -s $S -f /tmp/unpack_$S.py
