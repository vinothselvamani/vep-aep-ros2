#!/usr/bin/env bash
# Reviewer 2 follow-ups: middleware overhead (ROS 2 vs single process) and message-loss robustness.
#   PY=~/vepenv/bin/python ./run_extra.sh        (about 15 minutes)
set -o pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p extra
echo "== 1/4 Building ROS 2 packages =="
mkdir -p ws/src && rm -rf ws/src/vep_aep_ros2 ws/src/vep_aep_msgs && cp -r vep_aep_ros2 vep_aep_msgs ws/src/
(cd ws && colcon build 2>&1 | tail -2)
source ws/install/setup.bash
FS=$(env -u PYTHONPATH $PY -c "import numpy as np; print(float(np.load('data/eeg.npz')['fs']))")
MODEL=$PWD/results/model_modality.npz
echo "== 2/4 ROS 2 reference run (real time, about 5 min) =="
timeout --signal=INT 400 ros2 launch vep_aep_ros2 vep_aep.launch.py data_file:=$PWD/data/eeg.npz model_file:=$MODEL \
  fs:=$FS task:=modality speed:=1.0 chunk_ms:=20.0 out_csv:=$PWD/extra/ros_ref.csv > extra/ros_ref.log 2>&1
echo "== 3/4 Single-process baseline, same timing, no ROS 2 (about 5 min) =="
env -u PYTHONPATH $PY scripts/baseline_single_process.py data/eeg.npz $MODEL extra/single_process.csv --speed 1.0 --chunk_ms 20
echo "== 4/4 Message-loss runs (x5 speed, about 4 min) =="
for P in 0.005 0.01 0.02 0.05; do
  timeout --signal=INT 120 ros2 launch vep_aep_ros2 vep_aep.launch.py data_file:=$PWD/data/eeg.npz model_file:=$MODEL \
    fs:=$FS task:=modality speed:=5.0 chunk_ms:=20.0 drop_prob:=$P out_csv:=$PWD/extra/loss_$P.csv > extra/loss_$P.log 2>&1
  echo "drop $P: $(($(wc -l < extra/loss_$P.csv) - 1)) decisions, $(grep -c 'gap of' extra/loss_$P.log) gaps detected"
done
env -u PYTHONPATH $PY scripts/extra_analysis.py results extra data/eeg.npz | tail -25
tar czf vep_aep_extra_results.tar.gz extra/*.csv extra/extra.json
echo "DONE. Upload $(pwd)/vep_aep_extra_results.tar.gz to the chat."
