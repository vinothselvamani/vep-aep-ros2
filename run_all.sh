#!/usr/bin/env bash
# One command to regenerate every number, figure and table of the paper.
#   ./run_all.sh                         -> downloads the MNE sample dataset automatically (~1.6 GB, once)
#   ./run_all.sh <raw.fif> [<events.fif>] -> use files you already have
# Needs a sourced ROS 2 (Humble/Jazzy) workspace in which vep_aep_ros2 is built (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
ROSPY=${ROSPY:-python3}
mkdir -p data results ros figs

if [ $# -ge 1 ]; then
  RAW=$1; EVE=${2:-}
else
  echo "== Downloading / locating MNE sample dataset (first time only, ~1.6 GB) =="
  SP=$(env -u PYTHONPATH $PY -c "import mne; print(mne.datasets.sample.data_path(verbose=True))" | tail -1)
  RAW="$SP/MEG/sample/sample_audvis_filt-0-40_raw.fif"
  EVE="$SP/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
fi

echo "== 1/5 Preparing EEG data =="
env -u PYTHONPATH $PY scripts/prepare_data.py "$RAW" $EVE data/eeg.npz
FS=$(env -u PYTHONPATH $PY -c "import numpy as np; print(float(np.load('data/eeg.npz')['fs']))")
echo "== 2/5 Offline evaluation (takes a few minutes) =="
env -u PYTHONPATH $PY scripts/evaluate.py data/eeg.npz results --reps 10
echo "== 3/5 ROS 2 real-time runs (6 runs, about 15 minutes) =="
$ROSPY scripts/ros2_suite.py data/eeg.npz results "$FS" ros
echo "== 4/5 Figures and LaTeX numbers =="
env -u PYTHONPATH $PY scripts/figures.py results ros figs
cp figs/*.pdf paper/
env -u PYTHONPATH $PY scripts/make_tex_numbers.py results ros paper
env -u PYTHONPATH $PY scripts/figures_extra.py results ros paper
lscpu | grep -E "Model name|^CPU\(s\)" > ros/machine.txt || true
echo "ROS_DISTRO=${ROS_DISTRO:-unknown}" >> ros/machine.txt
echo "== 5/5 Packing results for upload =="
tar czf vep_aep_results.tar.gz results ros figs paper/numbers*.tex paper/tab_*.tex
if command -v pdflatex >/dev/null; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && \
   pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex | grep "Output written") || true
fi
echo "DONE. Upload $(pwd)/vep_aep_results.tar.gz to the chat."
