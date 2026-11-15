"""Generate paper figures from evaluate.py and ros2_suite.py outputs.

Usage: python figures.py <eval_outdir> <ros_outdir> <fig_outdir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

ev_dir, ros_dir, fig_dir = sys.argv[1:4]
os.makedirs(fig_dir, exist_ok=True)
res = json.load(open(os.path.join(ev_dir, "results.json")))
g = np.load(os.path.join(ev_dir, "grand.npz"))
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#888888"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
W1 = 3.45  # IEEE single column (in)

# ---------------- Fig. 2: grand-average waveforms + topographies ----------------
t = g["times"] * 1e3
fig = plt.figure(figsize=(W1, 2.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.75, wspace=0.3)
ax = fig.add_subplot(gs[0, :])
for arr, col, lab, ls in [(g["aep"], BLUE, "AEP (fronto-central cluster)", "-"), (g["vep"], ORANGE, "VEP (occipital cluster)", "--")]:
    m, se = arr.mean(0), arr.std(0) / np.sqrt(len(arr))
    ax.fill_between(t, m - se, m + se, color=col, alpha=0.2, lw=0)
    ax.plot(t, m, color=col, lw=1.4, ls=ls, label=lab)
ax.axvline(0, color=MUTED, lw=0.6); ax.axhline(0, color=MUTED, lw=0.6)
c = res["components"]
for k, col in [("AEP_N1", BLUE), ("AEP_P2", BLUE), ("VEP_P1", ORANGE), ("VEP_N1", ORANGE)]:
    ax.plot(c[k]["lat_ms"], c[k]["amp_uV"], "o", ms=3.5, color=col, mec="white", mew=0.6)
    lab = k.replace("_", " ")
    off = {"AEP_N1": (-24, -9), "AEP_P2": (6, 1), "VEP_P1": (-24, 4), "VEP_N1": (6, -6)}[k]
    ax.annotate(lab, (c[k]["lat_ms"], c[k]["amp_uV"]), textcoords="offset points",
                xytext=off, fontsize=6.5, color=INK)
ax.set_xlabel("Time from stimulus onset (ms)"); ax.set_ylabel(r"Amplitude ($\mu$V)")
ax.set_xlim(t[0], t[-1]); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
ax.set_title("(a) Grand-average evoked responses (mean $\\pm$ SEM)", fontsize=8, loc="left", pad=16)
ch_names = [str(n) for n in g["ch_names"]]
info = mne.create_info(ch_names, 100.0, "eeg")
info.set_montage(mne.channels.make_dig_montage(dict(zip(ch_names, g["pos"])), coord_frame="head"))
for i, (key, arr, name) in enumerate([("AEP_N1", g["X_mean_aep"], "AEP N1"), ("VEP_P1", g["X_mean_vep"], "VEP P1")]):
    axt = fig.add_subplot(gs[1, i])
    ti = np.argmin(np.abs(t - c[key]["lat_ms"]))
    im, _ = mne.viz.plot_topomap(arr[:, ti], info, axes=axt, show=False, cmap="RdBu_r", contours=4, sensors=True)
    axt.set_title(f"({'bc'[i]}) {name}, {c[key]['lat_ms']:.0f} ms", fontsize=7.5)
    cb = fig.colorbar(im, ax=axt, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label(r"$\mu$V", fontsize=6.5)
fig.savefig(os.path.join(fig_dir, "fig_erp.pdf")); fig.savefig(os.path.join(fig_dir, "fig_erp.png"))
plt.close(fig)

# ---------------- Fig. 3: single-trial classification ----------------
tasks = list(res["cv"].keys())
methods = ["xDAWN-LDA", "Cluster-LDA", "Template"]
cols = [BLUE, ORANGE, AQUA]
hatches = ["", "////", ".."]
fig, ax = plt.subplots(figsize=(W1, 1.6))
xw = 0.26
for j, (m, col, h) in enumerate(zip(methods, cols, hatches)):
    vals = [res["cv"][tk][m]["acc_mean"] * 100 for tk in tasks]
    sds = [res["cv"][tk][m]["acc_sd"] * 100 for tk in tasks]
    ax.bar(np.arange(len(tasks)) + (j - 1) * xw, vals, xw * 0.92, yerr=sds, color=col, hatch=h, edgecolor="white",
           lw=0.3, label=m, error_kw=dict(lw=0.6, capsize=1.5, ecolor=INK))
for i, tk in enumerate(tasks):
    ax.plot([i - 1.5 * xw, i + 1.5 * xw], [res["chance"][tk] * 100] * 2, color=INK, lw=0.8, ls=":")
ax.set_xticks(range(len(tasks)))
short = {"modality (AEP vs VEP)": "AEP vs VEP", "4-class (LA/RA/LV/RV)": "4-class", "aud. laterality (L vs R)": "Aud. L/R", "vis. laterality (L vs R)": "Vis. L/R"}
ax.set_xticklabels([short.get(tk, tk) for tk in tasks], fontsize=7)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6.5)
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(fig_dir, "fig_cls.pdf")); fig.savefig(os.path.join(fig_dir, "fig_cls.png"))
plt.close(fig)

# ---------------- Fig. 4: averaging detection + learning curve ----------------
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.8), gridspec_kw=dict(wspace=0.6))
ax = axs[0]
for comp, col, mk in [("AEP N1", BLUE, "o"), ("VEP P1", ORANGE, "s")]:
    rows = res["averaging_detection"][comp]
    ks = [int(k) for k in rows]
    ax.plot(ks, [rows[str(k)]["det_rate"] * 100 for k in ks], marker=mk, ms=3.5, color=col, lw=1.4, label=f"{comp} hit")
    ax.plot(ks, [rows[str(k)]["fa_rate"] * 100 for k in ks], marker=mk, ms=3.5, color=col, lw=1.0, ls="--", mfc="white",
            label=f"{comp} FA")
ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
ax.minorticks_off(); ax.set_xlabel("Trials averaged, $k$"); ax.set_ylabel("Rate (%)")
ax.set_ylim(-3, 103); ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, handlelength=1.6, columnspacing=0.8)
ax.set_title("(a) 3-SD peak criterion", fontsize=7.5, loc="left")
ax = axs[1]
lc = res["learning_curve"]
n = [int(k) for k in lc]
mu = np.array([lc[str(k)][0] for k in n]) * 100; sd = np.array([lc[str(k)][1] for k in n]) * 100
ax.fill_between(n, mu - sd, mu + sd, color=BLUE, alpha=0.2, lw=0)
ax.plot(n, mu, marker="o", ms=3.5, color=BLUE, lw=1.4)
ax.set_xlabel("Calibration trials"); ax.set_ylabel("Modality acc. (%)")
ax.set_title("(b) Learning curve", fontsize=7.5, loc="left")
fig.savefig(os.path.join(fig_dir, "fig_det.pdf")); fig.savefig(os.path.join(fig_dir, "fig_det.png"))
plt.close(fig)

# ---------------- Fig. 5: ROS 2 latency distributions ----------------
summ = json.load(open(os.path.join(ros_dir, "ros2_summary.json")))
fig, ax = plt.subplots(figsize=(W1, 1.35))
data, labels = [], []
for r in summ:
    f = os.path.join(ros_dir, f"{r['task']}_x{r['speed']:g}_c{r['chunk_ms']}.csv")
    a = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
    data.append(a["excess_ms"]); labels.append(f"{'2-cl' if r['task']=='modality' else '4-cl'}\n{r['chunk_ms']} ms\n×{r['speed']:g}")
bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                flierprops=dict(ms=2, mec=MUTED), medianprops=dict(color=INK, lw=1))
for b in bp["boxes"]:
    b.set_facecolor(BLUE); b.set_alpha(0.55); b.set_edgecolor(BLUE)
ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("Excess latency (ms)"); ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(fig_dir, "fig_lat.pdf")); fig.savefig(os.path.join(fig_dir, "fig_lat.png"))
plt.close(fig)
print("figures written to", fig_dir)
