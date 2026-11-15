"""Additional analyses requested in review.

  1. Eye movements: decode the classes from the EOG channel alone; regress EOG out of the EEG and re-decode.
  2. Frontal channels: re-decode with the anterior third of the electrodes removed.
  3. xDAWN spatial patterns (forward model, Haufe et al.) for the modality model.
  4. Riemannian baseline (xDAWN covariances + tangent space + logistic regression).
  5. Corrected resampled t-test (Nadeau & Bengio) between xDAWN-LDA and every other method.
  6. Fairer classical detector: noise from the +/- reference over the whole epoch.
  7. 1000-permutation test for modality decoding.
  8. Group delay of the causal band-pass filter.

Usage: python revision.py <raw.fif> [<events.fif>] <outdir>
"""
import json
import os
import sys
import time

import mne
import numpy as np
from scipy import signal, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vep_aep_ros2"))
from vep_aep_ros2.core import (MODALITY, CausalBandpass, ChannelLDA, TemplateMatch,  # noqa: E402
                               XdawnLDA, car, make_epoch, peak_measure)

args = sys.argv[1:]
raw_f, outdir = args[0], args[-1]
eve_f = args[1] if len(args) == 3 else None
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(0)
T0 = time.time()

raw = mne.io.read_raw_fif(raw_f, preload=True, verbose=False)
events = mne.read_events(eve_f) if eve_f else mne.find_events(raw, stim_channel="STI 014", verbose=False)
events = events[np.isin(events[:, 2], [1, 2, 3, 4])]
eeg = raw.copy().pick("eeg", exclude="bads")
eog = raw.copy().pick("eog")
fs = raw.info["sfreq"]
x = eeg.get_data().T * 1e6
v = eog.get_data().T * 1e6
pos = np.array([ch["loc"][:3] for ch in eeg.info["chs"]])
names = eeg.ch_names
n_ch = x.shape[1]

xf = car(CausalBandpass(fs, n_ch, 1.0, 30.0, 4)(x))
vf = CausalBandpass(fs, v.shape[1], 1.0, 30.0, 4)(v)
n_pre, n_post = int(round(0.1 * fs)), int(round(0.5 * fs))
times = np.arange(-n_pre, n_post) / fs
on = events[:, 0] - raw.first_samp
keep = (on - n_pre >= 0) & (on + n_post <= len(xf))
on, codes = on[keep], events[keep, 2]
ymod = np.array([MODALITY[c] for c in codes])
decim = max(1, int(round(fs / 37.5)))


def epoch(sig):
    return np.stack([make_epoch(sig[s - n_pre:s + n_post], n_pre) for s in on])


X = epoch(xf)
V = epoch(vf)
p = pos - pos.mean(0)
vertex = np.argsort(-p[:, 2])[:1]
fc = np.argsort(np.linalg.norm(p - p[vertex] + np.array([0, -0.02, 0]), axis=1))[:5]
occ = np.argsort(p[:, 1])[:5]

TASKS = {"modality": (np.ones(len(X), bool), ymod), "4class": (np.ones(len(X), bool), codes),
         "aud_lat": (ymod == 0, codes), "vis_lat": (ymod == 1, codes)}
out = {"n_trials": int(len(X)), "fs": fs, "eog_channels": eog.ch_names}

# ---------------------------------------------------------------- methods
try:
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    def riemann():
        return make_pipeline(XdawnCovariances(nfilter=3, estimator="lwf", xdawn_estimator="lwf"),
                             TangentSpace(metric="riemann"), LogisticRegression(max_iter=2000))
    HAVE_RIEMANN = True
except Exception as e:  # pragma: no cover
    HAVE_RIEMANN = False
    out["riemann_error"] = repr(e)


class EOGLDA:
    def fit(self, X, y):
        self.m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self._f(X), y); return self

    def _f(self, X):
        return signal.decimate(X, decim, axis=-1, zero_phase=True).reshape(len(X), -1)

    def predict(self, X):
        return self.m.predict(self._f(X))


def methods(picks_fc=fc, picks_o=occ):
    M = {"xDAWN-LDA": lambda: XdawnLDA(3, decim),
         "Cluster-LDA": lambda: ChannelLDA(list(picks_fc) + list(picks_o), decim),
         "Template": lambda: TemplateMatch()}
    if HAVE_RIEMANN:
        M["Riemann-TS"] = riemann
    return M


def cv_folds(Xs, y, makers, reps=10, k=5):
    """Return per-fold accuracies [method][rep*k] using identical splits for every method."""
    acc = {m: [] for m in makers}
    ntr = []
    for r in range(reps):
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=r).split(Xs, y):
            ntr.append((len(tr), len(te)))
            for m, mk in makers.items():
                Xin = Xs.astype(np.float64)
                acc[m].append(accuracy_score(y[te], mk().fit(Xin[tr], y[tr]).predict(Xin[te])))
    return {m: np.array(a) for m, a in acc.items()}, ntr


def corrected_ttest(a, b, ntr):
    """Nadeau & Bengio (2003) corrected resampled t-test over repeated k-fold CV."""
    d = a - b
    J = len(d)
    n_train, n_test = ntr[0]
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    pval = 2 * stats.t.sf(abs(t), J - 1)
    return float(t), float(pval)


# ---------------------------------------------------------------- 4/5: all methods, same folds, stats
out["methods"] = {}
for task, (sel, y) in TASKS.items():
    accs, ntr = cv_folds(X[sel], y[sel], methods())
    entry = {m: {"acc": float(a.mean()), "sd_rep": float(a.reshape(10, -1).mean(1).std())} for m, a in accs.items()}
    for m in accs:
        if m != "xDAWN-LDA":
            t, pv = corrected_ttest(accs["xDAWN-LDA"], accs[m], ntr)
            entry[m]["t_vs_xdawn"], entry[m]["p_vs_xdawn"] = t, pv
    out["methods"][task] = entry
    print(task, {m: round(e["acc"], 3) for m, e in entry.items()}, f"{time.time() - T0:.0f}s", flush=True)

# ---------------------------------------------------------------- 1: eye movements
eog_res = {}
for task, (sel, y) in TASKS.items():
    a, _ = cv_folds(V[sel], y[sel], {"EOG-LDA": lambda: EOGLDA()})
    eog_res[task] = {"acc": float(a["EOG-LDA"].mean()), "chance": float(np.bincount(y[sel] - y[sel].min()).max() / sel.sum())}
ptp = {int(c): float(np.ptp(V[codes == c, 0], axis=-1).mean()) for c in [1, 2, 3, 4]}
eog_res["ptp_uV_by_class"] = ptp
# regress EOG out of continuous EEG (unsupervised, no labels involved)
B, *_ = np.linalg.lstsq(np.c_[vf, np.ones(len(vf))], xf, rcond=None)
xr = car(xf - np.c_[vf, np.ones(len(vf))] @ B)
Xr = epoch(xr)
eog_res["after_regression"] = {}
for task, (sel, y) in TASKS.items():
    a, _ = cv_folds(Xr[sel], y[sel], {"xDAWN-LDA": lambda: XdawnLDA(3, decim)})
    eog_res["after_regression"][task] = float(a["xDAWN-LDA"].mean())
out["eog"] = eog_res
print("EOG", {t: round(eog_res[t]["acc"], 3) for t in TASKS}, "after regression", eog_res["after_regression"], flush=True)

# ---------------------------------------------------------------- 2: frontal exclusion
yq = p[:, 1]
post = np.where(yq < np.percentile(yq, 100 * 2 / 3))[0]
fr = {"n_removed": int(n_ch - len(post)), "removed": [names[i] for i in range(n_ch) if i not in set(post)]}
for task, (sel, y) in TASKS.items():
    a, _ = cv_folds(X[sel][:, post], y[sel], {"xDAWN-LDA": lambda: XdawnLDA(3, decim)})
    fr[task] = float(a["xDAWN-LDA"].mean())
out["frontal_excluded"] = fr
print("frontal excluded", fr, flush=True)

# ---------------------------------------------------------------- 3: xDAWN patterns (forward model)
m = XdawnLDA(3, decim).fit(X, ymod)
Xc = np.concatenate(list(X), axis=1)
Cx = np.cov(Xc)
W = m.W_
A = Cx @ W.T @ np.linalg.pinv(W @ Cx @ W.T)
np.savez(os.path.join(outdir, "xdawn_patterns.npz"), A=A, pos=pos, ch_names=np.array(names),
         first_aep=A[:, 0], first_vep=A[:, 3])

# ---------------------------------------------------------------- 6: fairer classical detector
fair = {}
for comp, sel, ch, win, pol in [("AEP N1", ymod == 0, fc, (0.07, 0.15), -1), ("VEP P1", ymod == 1, occ, (0.07, 0.16), +1)]:
    Wv = X[sel][:, ch].mean(1)
    rows = {}
    for k in [2, 4, 8, 16, 32]:
        if 2 * k > len(Wv):
            break
        det, fa = [], []
        for _ in range(300):
            idx = rng.choice(len(Wv), 2 * k, replace=False)
            a_idx, n_idx = idx[:k], idx[k:]
            signs = rng.permutation(np.r_[np.ones(k // 2 + k % 2), -np.ones(k // 2)])
            noise_sd = (Wv[a_idx] * signs[:, None]).mean(0).std()  # +/- reference, whole epoch
            lat, amp = peak_measure(Wv[a_idx].mean(0), times, win, pol)
            det.append(amp * pol > 3 * noise_sd)
            # false alarm: an independent +/- average tested against the noise estimate of another
            s2 = rng.permutation(signs)
            null_wave = (Wv[n_idx[:k]] * s2[:, None]).mean(0)
            _, amp0 = peak_measure(null_wave, times, win, pol)
            fa.append(amp0 * pol > 3 * noise_sd)
        rows[k] = {"det_rate": float(np.mean(det)), "fa_rate": float(np.mean(fa))}
    fair[comp] = rows
out["fair_detector"] = fair
print("fair detector", fair, flush=True)

# ---------------------------------------------------------------- 7: 1000 permutations
def acc5(y, seed):
    pr = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pr[te] = XdawnLDA(3, decim).fit(X[tr], y[tr]).predict(X[te])
    return accuracy_score(y, pr)


obs = acc5(ymod, 0)
NPERM = int(os.environ.get("NPERM", 1000))
null = np.array([acc5(rng.permutation(ymod), i) for i in range(NPERM)])
out["perm1000"] = {"acc": float(obs), "p": float((1 + (null >= obs).sum()) / (NPERM + 1)), "n_perm": NPERM, "null_max": float(null.max()),
                   "null_mean": float(null.mean())}
print("perm", out["perm1000"], f"{time.time() - T0:.0f}s", flush=True)

# ---------------------------------------------------------------- 8: group delay
sos = signal.butter(4, [1, 30], btype="bandpass", fs=fs, output="sos")
w, h = signal.sosfreqz(sos, worN=4096, fs=fs)
gd = -np.gradient(np.unwrap(np.angle(h)), 2 * np.pi * w / fs) / fs * 1e3  # ms
out["group_delay_ms"] = {str(f): float(np.interp(f, w, gd)) for f in [3, 5, 10, 15, 20]}
# empirical: peak latency with zero-phase vs causal on the averaged AEP at FC cluster
xz = signal.sosfiltfilt(sos, x, axis=0); xz = car(xz)
Xz = epoch(xz)
lat_c, _ = peak_measure(X[ymod == 0][:, fc].mean((0, 1)), times, (0.07, 0.15), -1)
lat_z, _ = peak_measure(Xz[ymod == 0][:, fc].mean((0, 1)), times, (0.05, 0.15), -1)
latv_c, _ = peak_measure(X[ymod == 1][:, occ].mean((0, 1)), times, (0.07, 0.16), +1)
latv_z, _ = peak_measure(Xz[ymod == 1][:, occ].mean((0, 1)), times, (0.05, 0.16), +1)
out["latency_causal_vs_zerophase_ms"] = {"AEP_N1": [lat_c * 1e3, lat_z * 1e3], "VEP_P1": [latv_c * 1e3, latv_z * 1e3]}
print("group delay", out["group_delay_ms"], out["latency_causal_vs_zerophase_ms"], flush=True)

json.dump(out, open(os.path.join(outdir, "revision.json"), "w"), indent=1)
print(f"done in {time.time() - T0:.0f}s")
