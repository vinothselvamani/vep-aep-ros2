"""Control analyses for the sequence confound (stimuli follow a near-fixed RA->LV->LA->RV cycle).

If the classifier used residual activity from the previous stimulus rather than the response to the
current one, then (i) pre-stimulus windows would already be decodable, and (ii) accuracy would drop on
trials whose predecessor broke the cycle or came after a long gap.

Usage: python controls.py <data.npz> <outdir>
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vep_aep_ros2"))
from vep_aep_ros2.core import MODALITY, CausalBandpass, XdawnLDA, car  # noqa: E402

npz, outdir = sys.argv[1:3]
os.makedirs(outdir, exist_ok=True)
d = np.load(npz)
x, fs, ev = d["data"].astype(float), float(d["fs"]), d["events"]
xf = car(CausalBandpass(fs, x.shape[1], 1.0, 30.0, 4)(x))
ev = ev[np.argsort(ev[:, 0])]
codes = ev[:, 1]
CYCLE = {(2, 3), (3, 1), (1, 4), (4, 2)}
prev = np.r_[-1, codes[:-1]]
gap = np.r_[np.inf, np.diff(ev[:, 0]) / fs]
in_cycle = np.array([(p, c) in CYCLE for p, c in zip(prev, codes)])
regular = in_cycle & (gap < 1.0)
S = lambda t: int(round(t * fs))  # noqa: E731


def epochs(t0, t1, base=None):
    keep = (ev[:, 0] + S(t0) >= 0) & (ev[:, 0] + S(t1) <= len(xf))
    X = np.stack([xf[s + S(t0): s + S(t1)].T for s in ev[keep, 0]])
    if base is not None:
        b0, b1 = S(base[0] - t0), S(base[1] - t0)
        X = X - X[:, :, b0:b1].mean(-1, keepdims=True)
    else:
        X = X - X.mean(-1, keepdims=True)  # demean only (no stimulus-locked baseline)
    return X, keep


def cv_pred(X, y, reps=10, decim=4):
    preds = []
    for r in range(reps):
        p = np.empty_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            p[te] = XdawnLDA(3, decim).fit(X[tr], y[tr]).predict(X[te])
        preds.append(p)
    return np.array(preds)


out = {"n_trials": int(len(ev)), "frac_in_cycle": float(in_cycle.mean()), "n_irregular": int((~regular).sum()),
       "prev_label_predictability": {}}
# how well does the previous stimulus alone predict the current one (the confound's strength)?
nxt = {2: 3, 3: 1, 1: 4, 4: 2}
v = prev > 0
out["prev_label_predictability"]["4class"] = float(np.mean([nxt[p] == c for p, c in zip(prev[v], codes[v])]))
out["prev_label_predictability"]["modality"] = float(np.mean([MODALITY[nxt[p]] == MODALITY[c] for p, c in zip(prev[v], codes[v])]))

windows = {
    "pre -200..0 ms (demeaned)": (-0.2, 0.0, None),
    "pre -100..0 ms (demeaned)": (-0.1, 0.0, None),
    "early 0..50 ms": (-0.1, 0.05, (-0.1, 0.0)),
    "standard -100..500 ms": (-0.1, 0.5, (-0.1, 0.0)),
    "standard, baseline -200..0": (-0.2, 0.5, (-0.2, 0.0)),
}
out["windows"] = {}
for name, (t0, t1, base) in windows.items():
    X, keep = epochs(t0, t1, base)
    if name.startswith("early"):
        X = X[:, :, S(0.1):]  # keep only 0..50 ms
    decim = 1 if X.shape[-1] < 20 else 4
    res = {}
    for task in ["modality", "4class"]:
        y = np.array([MODALITY[c] for c in codes[keep]]) if task == "modality" else codes[keep]
        P = cv_pred(X, y, decim=decim)
        acc = (P == y).mean(1)
        reg = regular[keep]
        res[task] = {"acc": float(acc.mean()), "acc_sd": float(acc.std()),
                     "chance": float(np.bincount(y - y.min()).max() / len(y)),
                     "acc_regular": float((P[:, reg] == y[reg]).mean()),
                     "acc_irregular": float((P[:, ~reg] == y[~reg]).mean()), "n_irregular": int((~reg).sum())}
    out["windows"][name] = res
    print(name, {t: round(v["acc"], 3) for t, v in res.items()},
          "irregular:", {t: (round(v["acc_irregular"], 3), v["n_irregular"]) for t, v in res.items()})
json.dump(out, open(os.path.join(outdir, "controls.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ["frac_in_cycle", "n_irregular", "prev_label_predictability"]}))
