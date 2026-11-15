"""Additional figures/tables: stimulus schematics, single-trial ERP image, ROS 2 terminal excerpt,
parameter table, 4-class confusion table, per-node resource table.

Usage: python figures_extra.py <eval_dir> <ros_dir> <paper_dir>
Everything except the stimulus schematics is drawn from logged results.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

ev_dir, ros_dir, out = sys.argv[1:4]
res = json.load(open(os.path.join(ev_dir, "results.json")))
g = np.load(os.path.join(ev_dir, "grand.npz"))
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#222222", "#888888"
CLS = {1: ("LA", BLUE, "o"), 2: ("RA", AQUA, "s"), 3: ("LV", ORANGE, "^"), 4: ("RV", YELLOW, "D")}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
W1 = 3.45

# ---------------------------------------------------------------- stimulus paradigm
fig = plt.figure(figsize=(W1, 2.15))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.85], hspace=0.55, wspace=0.25)


def screen(ax, side, title):
    ax.add_patch(FancyBboxPatch((0, 0), 16, 9, boxstyle="round,pad=0.2", fc="#808080", ec="#333333", lw=1.2))
    n = 8
    x0 = 0.6 if side == "L" else 8.6
    for i in range(n):
        for j in range(6):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x0 + i * 0.85, 1.9 + j * 0.85), 0.85, 0.85, fc="black", lw=0))
            else:
                ax.add_patch(Rectangle((x0 + i * 0.85, 1.9 + j * 0.85), 0.85, 0.85, fc="white", lw=0))
    ax.plot([8, 8], [4.0, 5.0], color="red", lw=1.2); ax.plot([7.5, 8.5], [4.5, 4.5], color="red", lw=1.2)
    ax.set_xlim(-0.5, 16.5); ax.set_ylim(-0.5, 9.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=7.5)


screen(fig.add_subplot(gs[0, 0]), "L", "(a) LV stimulus")
screen(fig.add_subplot(gs[0, 1]), "R", "(b) RV stimulus")
ax = fig.add_subplot(gs[0, 2])
tt = np.linspace(0, 1, 800)
env = np.clip(np.minimum(tt / 0.1, (1 - tt) / 0.1), 0, 1)
ax.plot(tt * 50, env * np.sin(2 * np.pi * 20 * tt), color=BLUE, lw=0.6)
ax.set_title("(c) Monaural tone pip", fontsize=7.5)
ax.set_xlabel("Time (a.u.)", fontsize=6.5); ax.set_yticks([]); ax.set_xticks([]); ax.spines["left"].set_visible(False)
ax.text(25, -1.55, "left ear (LA) or right ear (RA)", ha="center", fontsize=6)
ax.set_ylim(-1.9, 1.3)
# event timeline from the logged decisions (real onsets and codes)
r = np.genfromtxt(os.path.join(ros_dir, "4class_x1_c20.csv"), delimiter=",", names=True, dtype=None, encoding=None)
fs = res["fs"]
t = r["onset"] / fs
ax = fig.add_subplot(gs[1, :])
sel = t < t.min() + 20
for c, (lab, col, mk) in CLS.items():
    m = sel & (r["code"] == c)
    ax.plot(t[m], np.full(m.sum(), 5 - c), mk, color=col, ms=4, mec="white", mew=0.4, label=lab, ls="none")
ax.set_yticks([4, 3, 2, 1]); ax.set_yticklabels(["LA", "RA", "LV", "RV"], fontsize=6.5)
ax.set_xlabel("Recording time (s)"); ax.set_ylim(0.4, 4.6)
ax.grid(axis="x", color="#e6e6e6", lw=0.5)
ax.set_title(f"(d) First 20 s of the stimulus sequence (ISI {np.median(np.diff(t)) * 1e3:.0f} ms median)", fontsize=7.5, loc="left")
fig.savefig(os.path.join(out, "fig_paradigm.pdf")); fig.savefig(os.path.join(out, "fig_paradigm.png"))
plt.close(fig)
isi = np.diff(np.sort(t))

# ---------------------------------------------------------------- single-trial ERP image
tms = g["times"] * 1e3
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.6), sharey=False, gridspec_kw=dict(wspace=0.75))
for ax, arr, name in [(axs[0], g["aep"], "(a) AEP trials, FC cluster"), (axs[1], g["vep"], "(b) VEP trials, O cluster")]:
    v = np.percentile(np.abs(arr), 97)
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, origin="lower",
                   extent=[tms[0], tms[-1], 0.5, len(arr) + 0.5], interpolation="nearest")
    ax.axvline(0, color=INK, lw=0.6); ax.axvline(100, color=INK, lw=0.5, ls=":")
    ax.set_title(name, fontsize=7.5, loc="left"); ax.set_xlabel("Time (ms)")
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03); cb.ax.tick_params(labelsize=6); cb.set_label(r"$\mu$V", fontsize=6.5)
axs[0].set_ylabel("Trial (chronological)")
fig.savefig(os.path.join(out, "fig_erpimage.pdf")); fig.savefig(os.path.join(out, "fig_erpimage.png"))
plt.close(fig)

# ---------------------------------------------------------------- ROS 2 terminal excerpt (from the log)
lines = ["$ ros2 node list", "/eeg_streamer", "/erp_detector", "/erp_evaluator", "/erp_preprocessor",
         "$ ros2 topic echo /erp/decision --field data"]
for row in r[40:45]:
    lines.append(f"{{\"onset\": {row['onset']}, \"code\": {row['code']}, \"pred\": {row['pred']}, "
                 f"\"label\": \"{row['label']}\", \"score\": {row['score']:.1f}}}")
    lines.append("---")
held = r["onset"] > res["chronological"]["4class"]["cal_end_sample"]
acc = np.mean(r["true"][held] == r["pred"][held])
lines.append(f"[erp_evaluator]: {len(r)} decisions; held-out acc={acc:.3f}; mean latency={r['total_ms'].mean():.1f} ms")
fig = plt.figure(figsize=(W1, 1.8))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.02", fc="#1e1e1e", ec="#555555",
                            transform=ax.transAxes))
ax.add_patch(Rectangle((0.01, 0.91), 0.98, 0.08, fc="#3a3a3a", ec="none", transform=ax.transAxes))
for i, col in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    ax.plot(0.04 + i * 0.03, 0.95, "o", color=col, ms=3.5, transform=ax.transAxes)
ax.text(0.5, 0.95, "vinoth@ubuntu: ~/vep_aep_ros2_paper  (ROS 2 Lyrical)", color="#cccccc", fontsize=5.5,
        ha="center", va="center", family="monospace", transform=ax.transAxes)
y = 0.86
for ln in lines:
    col = "#7ec699" if ln.startswith("$") else ("#e6db74" if ln.startswith("[") else "#dddddd")
    ax.text(0.03, y, ln, color=col, fontsize=4.6, family="monospace", va="top", transform=ax.transAxes)
    y -= 0.047
fig.savefig(os.path.join(out, "fig_terminal.pdf")); fig.savefig(os.path.join(out, "fig_terminal.png"))
plt.close(fig)

# ---------------------------------------------------------------- tables
m = np.load(os.path.join(ev_dir, "model_4class.npz"))
P = [("Sampling rate", f"{res['fs']:.2f} Hz ({res['n_ch']} EEG channels)"),
     ("Band-pass filter", "Butterworth, order 4, 1--30 Hz, causal SOS"),
     ("Re-referencing", "Common average (CAR)"),
     ("Epoch window", "$-100$ to $500$ ms, pre-stimulus baseline"),
     ("Spatial filter", f"xDAWN, 3 filters/class ({m['W'].shape[0]} total, 4-class)"),
     ("Feature rate", f"Decimation $\\times${int(m['decim'])} ($\\approx${res['fs'] / int(m['decim']):.1f} Hz)"),
     ("Classifier", "LDA, Ledoit--Wolf shrinkage"),
     ("Chunk length", "10 / 20 / 40 ms (default 20 ms)"),
     ("QoS", "Reliable, keep-last, depth 200"),
     ]
T = ["\\begin{table}[t]", "\\centering", "\\caption{Pipeline parameters used in all experiments.}", "\\label{tab:params}",
     "\\footnotesize", "\\begin{tabular}{ll}", "\\toprule", "Parameter & Value \\\\", "\\midrule"]
T += [f"{a} & {b} \\\\" for a, b in P] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(out, "tab_params.tex"), "w").write("\n".join(T) + "\n")

cm = np.array(res["cv"]["4-class (LA/RA/LV/RV)"]["xDAWN-LDA"]["confusion"])
rowpct = cm / cm.sum(1, keepdims=True) * 100
labs = ["LA", "RA", "LV", "RV"]
T = ["\\begin{table}[t]", "\\centering",
     "\\caption{Row-normalised confusion matrix (\\%) of 4-class xDAWN-LDA, averaged over 10 CV repetitions.}",
     "\\label{tab:conf}", "\\footnotesize", "\\begin{tabular}{lcccc|c}", "\\toprule",
     "True $\\downarrow$ / Pred. $\\rightarrow$ & LA & RA & LV & RV & Recall \\\\", "\\midrule"]
for i, l in enumerate(labs):
    cells = [f"\\textbf{{{rowpct[i, j]:.1f}}}" if i == j else f"{rowpct[i, j]:.1f}" for j in range(4)]
    T.append(f"{l} & " + " & ".join(cells) + f" & {rowpct[i, i]:.1f} \\\\")
prec = np.diag(cm) / cm.sum(0) * 100
T += ["\\midrule", "Precision & " + " & ".join(f"{p:.1f}" for p in prec) + " & \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(out, "tab_conf.tex"), "w").write("\n".join(T) + "\n")

s = json.load(open(os.path.join(ros_dir, "ros2_summary.json")))
b1 = next(x for x in s if x["task"] == "modality" and x["speed"] == 1 and x["chunk_ms"] == 20)
b10 = next(x for x in s if x["speed"] == 10)
nodes = ["eeg_streamer", "erp_preprocessor", "erp_detector", "erp_evaluator"]
T = ["\\begin{table}[t]", "\\centering", "\\caption{Mean CPU load (\\% of one core) and resident memory per node (2-class, 20 ms chunks).}",
     "\\label{tab:nodes}", "\\footnotesize", "\\begin{tabular}{lccc}", "\\toprule",
     "Node & CPU $\\times$1 (\\%) & CPU $\\times$10 (\\%) & RSS (MB) \\\\", "\\midrule"]
for n in nodes:
    T.append(f"\\texttt{{{n.replace('_', chr(92) + '_')}}} & {b1['cpu_pct'][n]:.1f} & {b10['cpu_pct'][n]:.1f} & {b1['rss_mb'][n]:.0f} \\\\")
T += ["\\midrule", f"Total & {sum(b1['cpu_pct'].values()):.1f} & {sum(b10['cpu_pct'].values()):.1f} & {sum(b1['rss_mb'].values()):.0f} \\\\",
      "\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(out, "tab_nodes.tex"), "w").write("\n".join(T) + "\n")

# extra macros
M = {"ISImed": f"{np.median(isi) * 1e3:.0f}", "ISImin": f"{isi.min() * 1e3:.0f}", "ISImax": f"{np.percentile(isi, 95) * 1e3:.0f}",
     "RecallLA": f"{rowpct[0, 0]:.1f}", "RecallRA": f"{rowpct[1, 1]:.1f}", "RecallLV": f"{rowpct[2, 2]:.1f}", "RecallRV": f"{rowpct[3, 3]:.1f}",
     "Nfilt": str(m["W"].shape[0]), "Nfeat": str(m["coef"].shape[1]),
     "CpuStreamer": f"{b1['cpu_pct']['eeg_streamer']:.1f}", "CpuDet": f"{b1['cpu_pct']['erp_detector']:.1f}",
     "CpuStreamerFast": f"{b10['cpu_pct']['eeg_streamer']:.1f}"}
with open(os.path.join(out, "numbers_extra.tex"), "w") as fh:
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(M)
