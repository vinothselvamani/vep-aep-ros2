"""Summarise the message-loss and single-process runs.

Usage: python extra_analysis.py <results_dir> <extra_dir> <data.npz>
"""
import json
import os
import sys

import numpy as np

res_dir, ex, npz = sys.argv[1:4]
d = np.load(npz)
fs, ev = float(d["fs"]), d["events"]
off = np.load(os.path.join(res_dir, "offline_pred_modality.npy"))  # onset, code, pred (held-out half)
off_map = {int(o): int(p) for o, _, p in off}
chunk = max(1, int(round(0.020 * fs)))
n_epoch = int(round(0.1 * fs)) + int(round(0.5 * fs))
out = {"chunk_samples": chunk, "epoch_samples": n_epoch}


def load(f):
    return np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)


loss = {}
for f in sorted(os.listdir(ex)):
    if not f.startswith("loss_") or not f.endswith(".csv"):
        continue
    p = float(f[5:-4])
    r = load(os.path.join(ex, f))
    r = np.atleast_1d(r)
    held = [(o, pr) for o, pr in zip(r["onset"], r["pred"]) if int(o) in off_map]
    agree = float(np.mean([off_map[int(o)] == int(pr) for o, pr in held])) if held else float("nan")
    # an epoch spans about ceil(n_epoch/chunk)+1 chunks; it survives only if none of them is lost
    k = int(np.ceil(n_epoch / chunk)) + 1
    loss[p] = {"decisions": int(len(r)), "markers": int(len(ev)), "survival": len(r) / len(ev),
               "expected_survival": (1 - p) ** k, "agree_offline_heldout": agree, "n_heldout": len(held),
               "excess_ms_mean": float(np.mean(r["excess_ms"]))}
out["loss"] = loss
sp = load(os.path.join(ex, "single_process.csv"))
held = [(o, pr) for o, pr in zip(sp["onset"], sp["pred"]) if int(o) in off_map]
out["single_process"] = {"decisions": int(len(sp)), "excess_ms_mean": float(sp["excess_ms"].mean()),
                         "excess_ms_sd": float(sp["excess_ms"].std()), "excess_ms_p95": float(np.percentile(sp["excess_ms"], 95)),
                         "proc_ms": float(sp["proc_us"].mean() / 1e3),
                         "agree_offline_heldout": float(np.mean([off_map[int(o)] == int(pr) for o, pr in held]))}
ros = load(os.path.join(ex, "ros_ref.csv"))
out["ros_ref"] = {"decisions": int(len(ros)), "excess_ms_mean": float(ros["excess_ms"].mean()),
                  "excess_ms_sd": float(ros["excess_ms"].std()), "excess_ms_p95": float(np.percentile(ros["excess_ms"], 95)),
                  "proc_ms": float(ros["proc_us"].mean() / 1e3)}
out["middleware_overhead_ms"] = out["ros_ref"]["excess_ms_mean"] - out["single_process"]["excess_ms_mean"]
json.dump(out, open(os.path.join(ex, "extra.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
