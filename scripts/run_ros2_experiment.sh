#!/usr/bin/env bash
# Runs the full ROS 2 pipeline on a recording and stores per-decision timing.
# Usage: run_ros2_experiment.sh <data.npz> <model.npz> <fs> <task> <speed> <chunk_ms> <out.csv> [timeout_s]
set -e
DATA=$1 MODEL=$2 FS=$3 TASK=$4 SPEED=$5 CHUNK=$6 OUT=$7 TMO=${8:-400}
timeout --signal=INT "$TMO" ros2 launch vep_aep_ros2 vep_aep.launch.py \
  data_file:="$DATA" model_file:="$MODEL" fs:="$FS" task:="$TASK" speed:="$SPEED" chunk_ms:="$CHUNK" out_csv:="$OUT" \
  > "${OUT%.csv}.log" 2>&1 || true
echo "decisions: $(($(wc -l < "$OUT") - 1))"
