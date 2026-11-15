#!/usr/bin/env bash
# Runs every analysis requested by the reviewers and the turtlesim robot demo, then packs the results.
#   PY=~/vepenv/bin/python ./run_revision.sh
# Needs: data/eeg.npz and results/ from run_all.sh; ROS 2 sourced; turtlesim installed.
set -o pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
mkdir -p rev
RAW=$(env -u PYTHONPATH $PY -c "import mne; print(mne.datasets.sample.data_path(download=False))" | tail -1)/MEG/sample
echo "== 0/4 Checking dependencies =="
env -u PYTHONPATH $PY -c "import pyriemann" 2>/dev/null || env -u PYTHONPATH $PY -m pip install -q pyriemann
if ! ros2 pkg prefix turtlesim >/dev/null 2>&1; then
  echo "turtlesim missing: run   sudo apt install -y ros-${ROS_DISTRO}-turtlesim   and start again"; exit 1
fi
echo "== 1/4 Building ROS 2 packages (vep_aep_msgs + vep_aep_ros2) =="
mkdir -p ws/src && rm -rf ws/src/vep_aep_ros2 ws/src/vep_aep_msgs && cp -r vep_aep_ros2 vep_aep_msgs ws/src/
(cd ws && colcon build 2>&1 | tail -3)
source ws/install/setup.bash
echo "== 2/4 Sequence-confound controls (about 3 min) =="
env -u PYTHONPATH $PY scripts/controls.py data/eeg.npz rev | tail -7
echo "== 3/4 Reviewer analyses: EOG, frontal, Riemannian, statistics, fair detector, 1000 permutations (about 8 min) =="
env -u PYTHONPATH $PY scripts/revision.py "$RAW/sample_audvis_filt-0-40_raw.fif" "$RAW/sample_audvis_filt-0-40_raw-eve.fif" rev
echo "== 4/4 Robot demo: EEG -> detector -> turtlesim (real time, about 5 min; a turtle window opens) =="
FS=$(env -u PYTHONPATH $PY -c "import numpy as np; print(float(np.load('data/eeg.npz')['fs']))")
( sleep 25; timeout 10 ros2 topic echo --once /erp/decision_typed > rev/typed_msg_example.txt 2>&1;
  timeout 10 ros2 node list > rev/node_list.txt 2>&1; timeout 15 ros2 topic hz /eeg/raw --window 100 > rev/hz_raw.txt 2>&1 ) &
timeout --signal=INT 400 ros2 launch vep_aep_ros2 robot_demo.launch.py data_file:=$PWD/data/eeg.npz \
  model_file:=$PWD/results/model_4class.npz fs:=$FS task:=4class \
  out_csv:=$PWD/rev/robot_decisions.csv robot_csv:=$PWD/rev/robot.csv > rev/robot_log.txt 2>&1
wait
echo "robot commands logged: $(($(wc -l < rev/robot.csv) - 1))"
lscpu | grep -E "Model name|^CPU\(s\)" > rev/machine.txt; echo "ROS_DISTRO=$ROS_DISTRO" >> rev/machine.txt
tar czf vep_aep_revision_results.tar.gz rev
echo "DONE. Upload $(pwd)/vep_aep_revision_results.tar.gz to the chat."
