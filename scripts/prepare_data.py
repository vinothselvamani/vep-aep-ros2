"""Convert the MNE sample recording to an EEG-only .npz used by the ROS 2 nodes.

Usage: python prepare_data.py <raw.fif> [<events.fif>] <out.npz>
"""
import sys

import mne
import numpy as np

args = sys.argv[1:]
raw_f, out = args[0], args[-1]
eve_f = args[1] if len(args) == 3 else None

raw = mne.io.read_raw_fif(raw_f, preload=True, verbose=False)
events = mne.read_events(eve_f) if eve_f else mne.find_events(raw, stim_channel="STI 014", verbose=False)
events = events[np.isin(events[:, 2], [1, 2, 3, 4])]
raw.pick("eeg", exclude="bads")
data = raw.get_data().T * 1e6  # (n_samples, n_ch) in microvolts
pos = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
np.savez_compressed(
    out,
    data=data.astype(np.float32),
    fs=raw.info["sfreq"],
    events=np.c_[events[:, 0] - raw.first_samp, events[:, 2]],
    ch_names=np.array(raw.ch_names),
    pos=pos,
)
print(f"{out}: {data.shape[0]/raw.info['sfreq']:.1f} s, {data.shape[1]} EEG ch, fs={raw.info['sfreq']:.2f} Hz,"
      f" events={np.bincount(events[:,2])[1:]}")
