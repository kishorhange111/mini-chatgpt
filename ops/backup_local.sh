#!/bin/bash
# (runs locally in WSL) Every 10 minutes, download the newest training checkpoint
# from the Colab VM to E:\Resumes\small_chatgpt\backup\ so that a reclaimed VM
# does not lose the training progress. Only the newest checkpoint is kept.
#
#   nohup bash ops/backup_local.sh > backup/backup.log 2>&1 &
C=~/.local/bin/colab
S=${1:-llm}                                   # Colab session running the LLM training
DEST=/mnt/e/Resumes/small_chatgpt/backup
mkdir -p $DEST
while true; do
  line=$(timeout 300 $C exec -s $S -f ops/backup_prepare.py 2>/dev/null | grep -E '^(BACKUP|NONE)')
  set -- $line
  if [ "$1" = "BACKUP" ]; then
    name=$2; shift 2
    if [ ! -f "$DEST/$name/COMPLETE" ]; then
      mkdir -p "$DEST/$name.partial"
      ok=1
      for part in "$@"; do
        [ -s "$DEST/$name.partial/$part" ] && continue
        # a hung transfer must not block the loop forever (200 MB normally takes ~1 min)
        timeout 600 $C download -s $S "/content/backup/$part" "$DEST/$name.partial/$part.tmp" >/dev/null 2>&1 \
          && mv "$DEST/$name.partial/$part.tmp" "$DEST/$name.partial/$part" || { ok=0; rm -f "$DEST/$name.partial/$part.tmp"; break; }
      done
      if [ $ok = 1 ]; then
        for old in "$DEST"/checkpoint-*; do                # drop older checkpoints and stale partials
          [ "$old" != "$DEST/$name.partial" ] && rm -rf "$old"   # (final-sft / final-dpo are always kept)
        done
        mv "$DEST/$name.partial" "$DEST/$name" && touch "$DEST/$name/COMPLETE"
        echo "$(date +%T) saved $name ($# pieces, $(du -sh $DEST/$name | cut -f1))"
      else
        echo "$(date +%T) download of $name interrupted; will retry"
      fi
    fi
  else
    echo "$(date +%T) $line"
  fi
  sleep 600
done
