#!/bin/bash
# Evaluate the 50K checkpoint across every free chip on the three NPU hosts.
#
# Why shard by task: eval.py runs a suite's 10 tasks sequentially on one chip, so
# libero_10 alone is ~16 h. The 40 (suite, task) shards over many chips finish in
# ~3 h. Sharding is lossless: episode seeds come from --seed + batch index only,
# never from which tasks a suite contains, and `eval_shard.py merge` rebuilds the
# full-suite numbers from raw per-episode successes (round-trip verified
# bit-identical against the coverageA d=0 result).
#
# Protocol matches the 97.95 reproduction: n_action_steps=10, overlap=0,
# 50 episodes/task, seed=1000.
#
# Usage:  SKIP_WAIT=1 bash run_eval_after_train.sh   # training already done
set -u

REPO=/home/z00633863/flashvla
CKPT_ROOT=/home/data/flashvla/train_50k_prod/checkpoints
LOG_ROOT=/home/data/flashvla/logs/prod50k
EVALLOG=/home/data/flashvla/logs/eval_prod50k.log
TRAINLOG=/home/data/flashvla/logs/train_50k_prod.log
KEY=/root/.ssh/id_ed25519_to8
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=8"
PY=/root/miniconda3/envs/flashvla-npu/bin/python
SKIP_WAIT=${SKIP_WAIT:-0}

# .9 first: it is ours end to end. .10 and .8 can be reclaimed by their owners,
# so they take the tail of the queue rather than the critical path.
HOSTS="192.168.45.9 192.168.45.10 192.168.45.8"
REMOTE_HOSTS="192.168.45.10 192.168.45.8"

say() { echo "[eval-dispatch $(date '+%F %T')] $*" | tee -a "$EVALLOG"; }

mkdir -p "$LOG_ROOT"

# ---------------------------------------------------------------- 1. wait
if [ "$SKIP_WAIT" != "1" ]; then
  say "waiting for training to finish"
  while ! /usr/bin/grep -q "End of FlashVLA training" "$TRAINLOG" 2>/dev/null; do
    sleep 120
  done
  say "training banner seen; waiting for trainer processes to exit"
  while /usr/bin/pgrep -f "flashvla.train" >/dev/null 2>&1; do sleep 30; done
  sleep 90
fi

CKPT="$CKPT_ROOT/050000/pretrained_model"
if [ ! -f "$CKPT/model.safetensors" ]; then
  LAST=$(/usr/bin/readlink -f "$CKPT_ROOT/last" 2>/dev/null)
  if [ -n "$LAST" ] && [ -f "$LAST/pretrained_model/model.safetensors" ]; then
    CKPT="$LAST/pretrained_model"
  else
    say "ERROR: no usable checkpoint under $CKPT_ROOT"; exit 1
  fi
fi
say "checkpoint: $CKPT"

# ---------------------------------------------------------------- 2. free chips
# Idle baseline HBM is ~3.4 GB; anything under 5 GB counts as free.
declare -A CHIPLIST
say "probing free chips"
for h in $HOSTS; do
  if [ "$h" = "192.168.45.9" ]; then
    cs=$($PY - <<'INNER' 2>/dev/null
import re, subprocess
out = subprocess.run(["/usr/local/bin/npu-smi", "info"], capture_output=True, text=True).stdout
free, cur = [], None
for line in out.splitlines():
    m = re.match(r"^\|\s*(\d+)\s+910", line)
    if m:
        cur = int(m.group(1)); continue
    m = re.match(r"^\|\s*\d+\s*\|.*?(\d+)\s*/\s*65536", line)
    if m and cur is not None:
        if int(m.group(1)) < 5000:
            free.append(cur)
        cur = None
print(" ".join(str(c) for c in free))
INNER
)
  else
    cs=$($SSH root@"$h" 'python3 - 5000' <<'INNER' 2>/dev/null
import re, subprocess, sys
thr = int(sys.argv[1])
out = subprocess.run(["/usr/local/bin/npu-smi", "info"], capture_output=True, text=True).stdout
free, cur = [], None
for line in out.splitlines():
    m = re.match(r"^\|\s*(\d+)\s+910", line)
    if m:
        cur = int(m.group(1)); continue
    m = re.match(r"^\|\s*\d+\s*\|.*?(\d+)\s*/\s*65536", line)
    if m and cur is not None:
        if int(m.group(1)) < thr:
            free.append(cur)
        cur = None
print(" ".join(str(c) for c in free))
INNER
)
  fi
  CHIPLIST[$h]="$cs"
  say "  $h free chips: ${cs:-none}"
done

# ---------------------------------------------------------------- 3. plan
cd "$REPO" || exit 1
TASKS=/tmp/eval_shard_plan.txt
$PY sim_eval/libero/eval_shard.py plan --out "$TASKS" >/dev/null

rm -rf "$LOG_ROOT/shard_out"
mkdir -p "$LOG_ROOT/shard_out"

# One sequential queue per (host, chip): a chip runs its shards one at a time.
# Plain sort puts libero_10 first, so the slow suite lands on the earliest slots.
declare -A QUEUE
SLOTS=()
for h in $HOSTS; do
  for c in ${CHIPLIST[$h]}; do SLOTS+=("$h:$c"); done
done
NSLOTS=${#SLOTS[@]}
if [ "$NSLOTS" -eq 0 ]; then
  say "ERROR: no free chips"; exit 1
fi
say "using $NSLOTS slots: ${SLOTS[*]}"

i=0
while read -r suite task; do
  [ -z "${suite:-}" ] && continue
  slot=${SLOTS[$((i % NSLOTS))]}
  QUEUE[$slot]="${QUEUE[$slot]:-}$suite:$task "
  i=$((i+1))
done < <(/usr/bin/sort "$TASKS")
say "queue plan: $i shards over $NSLOTS slots"

# ---------------------------------------------------------------- 4. sync ckpt
for h in $REMOTE_HOSTS; do
  [ -z "${CHIPLIST[$h]}" ] && continue
  say "syncing checkpoint -> $h"
  $SSH root@"$h" "mkdir -p /home/data/flashvla/weights $LOG_ROOT/shard_out" 2>/dev/null
  rsync -a --partial -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    "$CKPT/" root@"$h":/home/data/flashvla/weights/prod50k/ 2>&1 | tail -2
done

# ---------------------------------------------------------------- 5. launch
say "launching per-chip queues"
for slot in "${SLOTS[@]}"; do
  h=${slot%%:*}; c=${slot##*:}
  q=${QUEUE[$slot]:-}
  [ -z "$q" ] && continue

  pol=$CKPT
  [ "$h" != "192.168.45.9" ] && pol=/home/data/flashvla/weights/prod50k

  runner=/tmp/shardq_${c}.sh
  {
    echo '#!/bin/bash'
    for item in $q; do
      s=${item%%:*}; t=${item##*:}
      echo "\"$PY\" \"$REPO/sim_eval/libero/eval_shard.py\" run \\"
      echo "  --suite $s --task $t --policy \"$pol\" \\"
      echo "  --out $LOG_ROOT/shard_out --chip $c \\"
      echo "  >> $LOG_ROOT/shard_${s}_${t}.log 2>&1"
    done
  } > "$runner"
  chmod +x "$runner"

  if [ "$h" = "192.168.45.9" ]; then
    setsid nohup bash "$runner" >> "$LOG_ROOT/shard_q_${c}.log" 2>&1 < /dev/null &
  else
    scp -q -i "$KEY" -o StrictHostKeyChecking=no "$runner" root@"$h":"$runner" 2>/dev/null
    $SSH root@"$h" "chmod +x $runner; cd $REPO && setsid nohup bash $runner >> $LOG_ROOT/shard_q_${c}.log 2>&1 < /dev/null &" 2>/dev/null
  fi
  say "  $h chip=$c <- $q"
  sleep 2
done

# ---------------------------------------------------------------- 6. wait
say "waiting for all shards"
while true; do
  local_n=$(/usr/bin/pgrep -fc 'eval_shard.py run' 2>/dev/null || true)
  local_n=${local_n:-0}
  remote_n=0
  for h in $REMOTE_HOSTS; do
    [ -z "${CHIPLIST[$h]}" ] && continue
    n=$($SSH root@"$h" "/usr/bin/pgrep -fc 'eval_shard.py run'" 2>/dev/null || true)
    remote_n=$((remote_n + ${n:-0}))
  done
  total=$((local_n + remote_n))
  done_cnt=$(/usr/bin/find "$LOG_ROOT/shard_out" -name eval_info.json 2>/dev/null | /usr/bin/wc -l)
  say "  running=$total finished=$done_cnt/40"
  [ "$done_cnt" -ge 40 ] && break
  if [ "$total" -eq 0 ]; then
    say "  no shards running; stopping wait"
    sleep 20
    break
  fi
  sleep 180
done

# ---------------------------------------------------------------- 7. pull + merge
for h in $REMOTE_HOSTS; do
  [ -z "${CHIPLIST[$h]}" ] && continue
  say "pulling results from $h"
  rsync -a --partial -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    root@"$h":"$LOG_ROOT/shard_out/" "$LOG_ROOT/shard_out/" 2>&1 | tail -1
done

say "merging"
$PY sim_eval/libero/eval_shard.py merge \
  --shards "$LOG_ROOT/shard_out" \
  --out "$LOG_ROOT/merged_eval_info.json" 2>&1 | tee -a "$EVALLOG"

say "done"
