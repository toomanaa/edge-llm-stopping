#!/bin/bash
# Orchestrator for the full-scale live controller run. Automates the
# manual "watch for a pause, Ctrl-C the server, restart it, rerun the
# command" cycle that would otherwise need repeating ~10 times over
# ~8-9 hours for the full 1,074-question run.
#
# Usage:
#   nohup bash run_full_orchestrated.sh > out/orchestrator.log 2>&1 &
#   (the "nohup ... &" lets it keep running even if the SSH session drops)
#
# Watch progress with:
#   tail -f out/orchestrator.log
#
# Stop it cleanly with:
#   pkill -f run_full_orchestrated.sh ; pkill llama-server

set -u

MODEL="models/llama-3.2-1b-instruct-q4_k_m.gguf"
DATA="data/qa_v2.jsonl"
OUT="out/live_full_v2.jsonl"
PROBE="probe_weights.npz"
THRESHOLD=0.99
BATCH_SIZE=100
PORT=8080
MAX_RESTARTS=30   # safety valve: never loop forever even if something is stuck

restart_count=0

# Diagnostics: print exactly what directory and files this script sees
# before doing anything else. A silent "done" with no questions
# processed and no output file created (observed once, cause not yet
# confirmed) is consistent with a working-directory or path mismatch
# that these checks are meant to catch immediately instead of guessing.
echo "=== $(date): diagnostic check ==="
echo "working directory: $(pwd)"
echo "DATA=$DATA  -> $([ -f "$DATA" ] && echo "EXISTS ($(wc -l < "$DATA") lines)" || echo "MISSING")"
echo "PROBE=$PROBE  -> $([ -f "$PROBE" ] && echo "EXISTS" || echo "MISSING")"
echo "MODEL=$MODEL  -> $([ -f "$MODEL" ] && echo "EXISTS" || echo "MISSING")"
echo "controller/live_controller.py -> $([ -f "controller/live_controller.py" ] && echo "EXISTS" || echo "MISSING")"
echo "=== end diagnostic check ==="

while [ $restart_count -lt $MAX_RESTARTS ]; do
  echo "=== $(date): starting llama-server (restart #$restart_count) ==="

  # Make sure nothing is already squatting on the port from a previous
  # crashed attempt before starting a fresh one.
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  sleep 2

  ./llama.cpp/build/bin/llama-server -m "$MODEL" -c 2048 --port $PORT --embeddings \
      > out/server_log_$restart_count.txt 2>&1 &
  SERVER_PID=$!

  # Wait for the server to actually be ready, rather than a fixed sleep --
  # poll /health for up to 60 seconds.
  ready=0
  for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"status":"ok"'; then
      ready=1
      break
    fi
    sleep 1
  done

  if [ $ready -eq 0 ]; then
    echo "=== $(date): server did not become ready in time, killing and retrying ==="
    kill -9 $SERVER_PID 2>/dev/null
    restart_count=$((restart_count + 1))
    continue
  fi

  echo "=== $(date): server ready, running controller batch ==="
  python3 controller/live_controller.py --data "$DATA" --out "$OUT" \
      --probe "$PROBE" --threshold $THRESHOLD --batch-size $BATCH_SIZE
  EXIT_CODE=$?

  echo "=== $(date): controller exited with code $EXIT_CODE ==="
  kill $SERVER_PID 2>/dev/null
  sleep 2
  kill -9 $SERVER_PID 2>/dev/null  # in case it didn't die cleanly

  if [ $EXIT_CODE -eq 0 ]; then
    echo "=== $(date): ALL QUESTIONS COMPLETE. Orchestrator finished. ==="
    break
  elif [ $EXIT_CODE -eq 2 ]; then
    echo "=== $(date): batch/memory pause signaled, restarting server and continuing ==="
    restart_count=$((restart_count + 1))
    continue
  else
    echo "=== $(date): unexpected exit code $EXIT_CODE, stopping orchestrator for safety ==="
    echo "=== check out/server_log_$restart_count.txt and the output above for what went wrong ==="
    break
  fi
done

if [ $restart_count -ge $MAX_RESTARTS ]; then
  echo "=== $(date): hit MAX_RESTARTS ($MAX_RESTARTS) safety limit, stopping. ==="
  echo "=== something may be stuck in a loop -- check the logs before resuming manually. ==="
fi

echo "=== orchestrator done, total restarts: $restart_count ==="
