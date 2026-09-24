#!/bin/bash
# Back up the newest training checkpoint to Google Drive every 5 minutes.
#
# Colab VMs are temporary: if the runtime is reclaimed, /content is wiped.
# Once Drive is mounted at /content/drive, this loop mirrors the latest
# checkpoint there (without optimizer.pt, to keep it ~2 GB instead of ~6 GB;
# on resume the Trainer simply re-initialises Adam's moment estimates).
#
# On start-up it also does the reverse: if the VM has no checkpoint but Drive
# has one, it is copied back so train.py resumes from it.
OUT=${OUT:-/content/out}
BACKUP=/content/drive/MyDrive/mini_chatgpt_backup

if [ -d /content/drive/MyDrive ] && [ -d "$BACKUP" ] && ! ls -d $OUT/checkpoint-* >/dev/null 2>&1; then
  mkdir -p $OUT && cp -r $BACKUP/checkpoint-* $OUT/ && echo "restored $(ls $OUT)"
fi

while true; do
  sleep 300
  [ -d /content/drive/MyDrive ] || continue
  latest=$(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [ -n "$latest" ] || continue
  name=$(basename $latest)
  if [ ! -d "$BACKUP/$name" ]; then
    mkdir -p $BACKUP
    rsync -a --exclude optimizer.pt $latest/ $BACKUP/$name.tmp/ && \
      rm -rf $BACKUP/checkpoint-* && mv $BACKUP/$name.tmp $BACKUP/$name && \
      echo "$(date +%T) backed up $name"
  fi
done
