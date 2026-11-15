"""Offline evaluation of VEP/AEP detection using exactly the streaming (causal) chain.

Usage: python evaluate.py <data.npz> <outdir> [--seed 0]
Produces: results.json, model.npz (chronological calibration model for ROS 2), figures.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vep_aep_ros2"))
from vep_aep_ros2.core import (MODALITY, CausalBandpass, ChannelLDA, TemplateMatch, XdawnLDA, car,
                               component_detected, make_epoch, peak_measure)

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("outdir")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--reps", type=int, default=10)
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
rng = np.random.default_rng(args.seed)

d = np.load(args.npz)
x, fs, ev, pos, names = d["data"].astype(float), float(d["fs"]), d["events"], d["pos"], d["ch_names"]
n_ch = x.shape[1]
TMIN, TMAX = -0.1, 0.5
n_pre, n_post = int(round(-TMIN * fs)), int(round(TMAX * fs))
times = np.arange(-n_pre, n_post) / fs
decim = max(1, int(round(fs / 37.5)))  # ~37.5 Hz feature rate

# ---- streaming-equivalent preprocessing (causal) ----
bp = CausalBandpass(fs, n_ch, 1.0, 30.0, 4)
xf = car(bp(x))
ev = ev[(ev[:, 0] - n_pre >= 0) & (ev[:, 0] + n_post <= len(xf))]
X = np.stack([make_epoch(xf[s - n_pre:s + n_post], n_pre) for s, _ in ev])
codes = ev[:, 1]
ymod = np.array([MODALITY[c] for c in codes])

# ---- channel clusters from sensor geometry ----
p = pos - pos.mean(0)
vertex = np.argsort(-p[:, 2])[:1]
fc = np.argsort(np.linalg.norm(p - p[vertex] + np.array([0, -0.02, 0]), axis=1))[:5]  # slightly anterior to vertex
occ = np.argsort(p[:, 1])[:5]  # most posterior
clusters = {"FC": fc.tolist(), "O": occ.tolist()}

res = {"fs": fs, "n_ch": n_ch, "n_trials": int(len(X)), "counts": {int(c): int((codes == c).sum()) for c in [1, 2, 3, 4]},
       "duration_s": len(x) / fs, "clusters": {k: [str(names[i]) for i in v] for k, v in clusters.items()}, "decim": decim}

# ---- grand averages & component measures ----
aep = X[ymod == 0][:, fc].mean(1)  # trials x time
vep = X[ymod == 1][:, occ].mean(1)
comp = {}
for name, w, win, pol in [("AEP_N1", aep.mean(0), (0.07, 0.15), -1), ("AEP_P2", aep.mean(0), (0.15, 0.26), +1),
                          ("VEP_P1", vep.mean(0), (0.07, 0.16), +1), ("VEP_N1", vep.mean(0), (0.12, 0.22), -1)]:
    lat, amp = peak_measure(w, times, win, pol)
    comp[name] = {"lat_ms": lat * 1e3, "amp_uV": amp}
res["components"] = comp
np.savez(os.path.join(args.outdir, "grand.npz"), times=times, aep=aep, vep=vep,
         X_mean_aep=X[ymod == 0].mean(0), X_mean_vep=X[ymod == 1].mean(0), pos=pos, ch_names=names)


# ---- cross-validated classification ----
def methods():
    return {"xDAWN-LDA": lambda: XdawnLDA(3, decim),
            "Cluster-LDA": lambda: ChannelLDA(fc.tolist() + occ.tolist(), decim),
            "Template": lambda: TemplateMatch()}


def cv_eval(Xs, y, reps, k=5):
    out = {}
    min_class = np.bincount(y - y.min()).min()
    k = int(min(k, min_class))
    for mname, mk in methods().items():
        accs, baccs, kap, aucs = [], [], [], []
        cm = np.zeros((len(np.unique(y)),) * 2)
        for r in range(reps):
            skf = StratifiedKFold(k, shuffle=True, random_state=args.seed + r)
            pred = np.empty_like(y)
            score = np.zeros(len(y))
            for tr, te in skf.split(Xs, y):
                m = mk().fit(Xs[tr], y[tr])
                pred[te] = m.predict(Xs[te])
                if len(np.unique(y)) == 2:
                    score[te] = m.decision_function(Xs[te])
            accs.append(accuracy_score(y, pred)); baccs.append(balanced_accuracy_score(y, pred))
            kap.append(cohen_kappa_score(y, pred)); cm += confusion_matrix(y, pred)
            if len(np.unique(y)) == 2:
                aucs.append(roc_auc_score(y, score))
        out[mname] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                      "bacc_mean": float(np.mean(baccs)), "kappa_mean": float(np.mean(kap)),
                      "confusion": (cm / reps).round(2).tolist(), "k_folds": k}
        if aucs:
            out[mname]["auc_mean"] = float(np.mean(aucs)); out[mname]["auc_sd"] = float(np.std(aucs))
    return out


tasks = {"modality (AEP vs VEP)": (X, ymod),
         "4-class (LA/RA/LV/RV)": (X, codes),
         "aud. laterality (L vs R)": (X[ymod == 0], codes[ymod == 0]),
         "vis. laterality (L vs R)": (X[ymod == 1], codes[ymod == 1])}
res["cv"] = {t: cv_eval(Xs, y, args.reps) for t, (Xs, y) in tasks.items()}
res["chance"] = {t: float(np.bincount(y - y.min()).max() / len(y)) for t, (_, y) in tasks.items()}

# ---- permutation test for modality (xDAWN-LDA, 5-fold) ----
def quick_acc(Xs, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for tr, te in skf.split(Xs, y):
        pred[te] = XdawnLDA(3, decim).fit(Xs[tr], y[tr]).predict(Xs[te])
    return accuracy_score(y, pred)

obs = quick_acc(X, ymod, 0)
null = [quick_acc(X, rng.permutation(ymod), i) for i in range(200)]
res["perm_modality"] = {"acc": obs, "p": float((1 + np.sum(np.array(null) >= obs)) / 201), "n_perm": 200}

# ---- learning curve (modality, xDAWN-LDA): accuracy vs # training trials ----
lc = {}
for ntr in [10, 20, 40, 60, 80, 120, 160, 200]:
    if ntr >= len(X) - 20:
        break
    a = []
    for r in range(args.reps * 2):
        perm = rng.permutation(len(X)); tr, te = perm[:ntr], perm[ntr:]
        if len(np.unique(ymod[tr])) < 2:
            continue
        a.append(accuracy_score(ymod[te], XdawnLDA(3, decim).fit(X[tr], ymod[tr]).predict(X[te])))
    lc[ntr] = [float(np.mean(a)), float(np.std(a))]
res["learning_curve"] = lc

# ---- chronological calibration split -> model for ROS 2 ----
n_cal = len(X) // 2
cal, tst = np.arange(n_cal), np.arange(n_cal, len(X))
chrono = {}
for task, y in [("modality", ymod), ("4class", codes)]:
    m = XdawnLDA(3, decim).fit(X[cal], y[cal])
    pr = m.predict(X[tst])
    chrono[task] = {"acc": float(accuracy_score(y[tst], pr)), "n_cal": int(n_cal), "n_test": int(len(tst)),
                    "cal_end_sample": int(ev[n_cal - 1, 0] + n_post)}
    np.savez(os.path.join(args.outdir, f"model_{task}.npz"), W=m.W_, coef=m.lda_.coef_, intercept=m.lda_.intercept_,
             classes=m.lda_.classes_, decim=decim, tmin=TMIN, tmax=TMAX, lo=1.0, hi=30.0, order=4, fs=fs)
    np.save(os.path.join(args.outdir, f"offline_pred_{task}.npy"), np.c_[ev[tst, 0], codes[tst], pr])
res["chronological"] = chrono

# ---- averaging-based component detection (classical clinical criterion) ----
avg = {}
for comp_name, sel, ch, win, pol in [("AEP N1", ymod == 0, fc, (0.07, 0.15), -1),
                                     ("VEP P1", ymod == 1, occ, (0.07, 0.16), +1)]:
    W = X[sel][:, ch].mean(1)
    rows = {}
    for k in [1, 2, 4, 8, 16, 32]:
        if k > len(W):
            break
        det, fa, lats = [], [], []
        for r in range(300):
            idx = rng.choice(len(W), k, replace=False)
            ok, lat, _ = component_detected(W[idx].mean(0), times, win, pol)
            det.append(ok); lats.append(lat)
            signs = rng.permutation(np.r_[np.ones(k // 2 + k % 2), -np.ones(k // 2)])  # +/- reference
            okn, _, _ = component_detected((W[idx] * signs[:, None]).mean(0), times, win, pol)
            fa.append(okn)
        rows[k] = {"det_rate": float(np.mean(det)), "fa_rate": float(np.mean(fa)),
                   "lat_ms_mean": float(np.mean(lats) * 1e3), "lat_ms_sd": float(np.std(lats) * 1e3)}
    avg[comp_name] = rows
res["averaging_detection"] = avg

# ---- classification with k-trial averaging at test time (modality) ----
kavg = {}
for k in [1, 2, 3, 5]:
    a = []
    for r in range(args.reps):
        perm = rng.permutation(len(X)); tr, te = perm[: int(0.6 * len(X))], perm[int(0.6 * len(X)):]
        m = XdawnLDA(3, decim).fit(X[tr], ymod[tr])
        Xa, ya = [], []
        for c in (0, 1):
            idx = te[ymod[te] == c]
            for g in range(len(idx) // k):
                Xa.append(X[idx[g * k:(g + 1) * k]].mean(0)); ya.append(c)
        if len(ya) and len(np.unique(ya)) == 2:
            a.append(accuracy_score(ya, m.predict(np.array(Xa))))
    if a:
        kavg[k] = [float(np.mean(a)), float(np.std(a))]
res["k_avg_classification"] = kavg

with open(os.path.join(args.outdir, "results.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: res[k] for k in ["n_trials", "counts", "components", "chance", "perm_modality", "chronological"]}, indent=1))
for t, r in res["cv"].items():
    print(t, {m: round(v["acc_mean"], 3) for m, v in r.items()})
