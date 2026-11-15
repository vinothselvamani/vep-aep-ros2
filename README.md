# VEP/AEP discrimination with ROS 2

Code, results and paper source for **"A ROS 2 Pipeline for Real-Time Single-Trial Discrimination of Visual and Auditory Evoked Potentials from Multichannel EEG"** (Vinoth S, Department of Biomedical Engineering, KIT Coimbatore).

A four-node ROS 2 graph streams EEG, filters it causally, cuts stimulus-locked epochs and classifies each trial as an auditory (AEP) or visual (VEP) evoked potential with xDAWN spatial filtering and shrinkage LDA. Decisions are published on `/erp/decision` (JSON) and `/erp/decision_typed` (`vep_aep_msgs/ErpDecision`), and can drive a robot (turtlesim demo included).

Key results on the public MNE sample recording (59 EEG channels, 288 trials, ROS 2 Lyrical, Intel i3-9100F):

| Result | Value |
|---|---|
| Single-trial AEP vs VEP | 98.2 % (permutation p = 0.001) |
| 4-class (LA/RA/LV/RV) | 83.9 % |
| Excess latency after epoch end (20 ms chunks) | 15.4 ms, of which 2.3 ms is ROS 2 middleware |
| Streamed vs offline decisions | 100 % identical |
| Robot (turtlesim) correct actions, held-out | 98.6 % |

## Layout
```
vep_aep_ros2/   ROS 2 package: eeg_streamer, erp_preprocessor, erp_detector, erp_evaluator, erp_robot_bridge + launch files
vep_aep_msgs/   ErpDecision.msg (typed decision message)
scripts/        data preparation, evaluation, controls, reviewer analyses, figures, LaTeX numbers
paper/          main.tex (IEEEtran), refs.bib, figures and auto-generated tables
results/ ros/ rev/ extra/   result files used in the paper
run_all.sh, run_revision.sh, run_extra.sh   regenerate every number, table and figure
```

## Reproduce (Ubuntu, ROS 2 Humble/Jazzy/Lyrical)
```bash
sudo apt install -y python3-scipy python3-psutil python3-colcon-common-extensions ros-$ROS_DISTRO-turtlesim
python3 -m venv ~/vepenv && ~/vepenv/bin/pip install mne scikit-learn matplotlib pyriemann
source /opt/ros/$ROS_DISTRO/setup.bash
mkdir -p ws/src && cp -r vep_aep_ros2 vep_aep_msgs ws/src/ && (cd ws && colcon build) && source ws/install/setup.bash
PY=~/vepenv/bin/python ./run_all.sh        # downloads the MNE sample data (~1.6 GB) on first run
PY=~/vepenv/bin/python ./run_revision.sh   # controls, baselines, statistics, robot demo
PY=~/vepenv/bin/python ./run_extra.sh      # middleware overhead and message-loss tests
```

## Live demo
```bash
ros2 launch vep_aep_ros2 robot_demo.launch.py data_file:=$PWD/data/eeg.npz \
     model_file:=$PWD/results/model_4class.npz fs:=150.15 task:=4class
```

## Data
The EEG is the public MNE-Python "sample" dataset (Gramfort et al., Front. Neurosci. 2013); it is downloaded automatically and not included in this repository.

## License
BSD 3-Clause, see `LICENSE`.
