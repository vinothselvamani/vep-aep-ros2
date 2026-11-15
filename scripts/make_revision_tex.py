"""Tables, macros and figures for the revised paper from rev/ (run_revision.sh output).

Usage: python make_revision_tex.py <eval_dir> <rev_dir> <paper_dir>
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ev_dir, rev_dir, out = sys.argv[1:4]
res = json.load(open(os.path.join(ev_dir, "results.json")))
rv = json.load(open(os.path.join(rev_dir, "revision.json")))
ct = json.load(open(os.path.join(rev_dir, "controls.json")))
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#888888"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
W1 = 3.45
f1 = lambda x: f"{x * 100:.1f}"  # noqa: E731
M = {}

# ------------------------------------------------ classification table with Riemannian column and tests
names = {"modality": "AEP vs.\\ VEP", "4class": "4-class", "aud_lat": "Aud.\\ L vs.\\ R", "vis_lat": "Vis.\\ L vs.\\ R"}
key_cv = {"modality": "modality (AEP vs VEP)", "4class": "4-class (LA/RA/LV/RV)", "aud_lat": "aud. laterality (L vs R)",
          "vis_lat": "vis. laterality (L vs R)"}
meth = ["xDAWN-LDA", "Riemann-TS", "Cluster-LDA", "Template"]


def mark(p):
    return "" if p is None else ("$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else ""))


T = ["\\begin{table}[t]", "\\centering",
     "\\caption{Single-trial accuracy (\\%, 10$\\times$5-fold CV; mean $\\pm$ SD over repetitions). $^{*}p<0.05$, $^{**}p<0.01$: worse than xDAWN-LDA (corrected resampled $t$-test).}",
     "\\label{tab:cls}", "\\setlength{\\tabcolsep}{1.6pt}", "\\footnotesize", "\\begin{tabular}{lccccc}", "\\toprule",
     "Task & Chance & xDAWN-LDA & Riem.-TS & Clust.-LDA & Templ. \\\\", "\\midrule"]
for t, nm in names.items():
    cells = []
    best = max(meth, key=lambda m: rv["methods"][t][m]["acc"])
    for m in meth:
        e = rv["methods"][t][m]
        c = f"{f1(e['acc'])}$\\pm${e['sd_rep'] * 100:.1f}{mark(e.get('p_vs_xdawn'))}"
        cells.append(f"\\textbf{{{c}}}" if m == best else c)
    T.append(f"{nm} & {res['chance'][key_cv[t]] * 100:.1f} & " + " & ".join(cells) + " \\\\")
T += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(out, "tab_cls.tex"), "w").write("\n".join(T) + "\n")
for t, tk in [("modality", "Mod"), ("4class", "Four"), ("aud_lat", "AudLat"), ("vis_lat", "VisLat")]:
    M[f"Rie{tk}"] = f1(rv["methods"][t]["Riemann-TS"]["acc"])
    M[f"RieP{tk}"] = f"{rv['methods'][t]['Riemann-TS']['p_vs_xdawn']:.2f}"
M["PClusterMax"] = f"{max(rv['methods'][t]['Cluster-LDA']['p_vs_xdawn'] for t in names):.2f}"

# ------------------------------------------------ control table
W = ct["windows"]
rows = [("Pre-stimulus $-200$--$0$\\,ms", W["pre -200..0 ms (demeaned)"]),
        ("Pre-stimulus $-100$--$0$\\,ms", W["pre -100..0 ms (demeaned)"]),
        ("Early $0$--$50$\\,ms", W["early 0..50 ms"]),
        ("Standard $-100$--$500$\\,ms", W["standard -100..500 ms"])]
T = ["\\begin{table}[t]", "\\centering",
     "\\caption{Control analyses (xDAWN-LDA, 10$\\times$5-fold CV accuracy in \\%). Off-cycle: the "
     + str(ct["n_irregular"]) + " trials that did not follow the fixed stimulus cycle.}",
     "\\label{tab:ctl}", "\\setlength{\\tabcolsep}{3pt}", "\\footnotesize", "\\begin{tabular}{lcccc}", "\\toprule",
     "Input & AEP/VEP & off-cycle & 4-class & Vis.\\ L/R \\\\", "\\midrule",
     f"Chance & {f1(W['standard -100..500 ms']['modality']['chance'])} & & {f1(W['standard -100..500 ms']['4class']['chance'])} & {f1(rv['eog']['vis_lat']['chance'])} \\\\",
     "\\midrule"]
for nm, w in rows:
    vis = f1(rv["methods"]["vis_lat"]["xDAWN-LDA"]["acc"]) if nm.startswith("Standard") else "--"
    T.append(f"{nm} & {f1(w['modality']['acc'])} & {f1(w['modality']['acc_irregular'])} & {f1(w['4class']['acc'])} & {vis} \\\\")
T.append(f"EOG channel only & {f1(rv['eog']['modality']['acc'])} & -- & {f1(rv['eog']['4class']['acc'])} & {f1(rv['eog']['vis_lat']['acc'])} \\\\")
T.append(f"EEG, EOG regressed out & {f1(rv['eog']['after_regression']['modality'])} & -- & {f1(rv['eog']['after_regression']['4class'])} & {f1(rv['eog']['after_regression']['vis_lat'])} \\\\")
fe = rv["frontal_excluded"]
T.append(f"EEG, {fe['n_removed']} frontal ch.\\ removed & {f1(fe['modality'])} & -- & {f1(fe['4class'])} & {f1(fe['vis_lat'])} \\\\")
T += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(os.path.join(out, "tab_ctl.tex"), "w").write("\n".join(T) + "\n")
pre = [W[k]["modality"]["acc"] for k in ["pre -200..0 ms (demeaned)", "pre -100..0 ms (demeaned)", "early 0..50 ms"]]
M["PreMin"], M["PreMax"] = f1(min(pre)), f1(max(pre))
pre4 = [W[k]["4class"]["acc"] for k in ["pre -200..0 ms (demeaned)", "pre -100..0 ms (demeaned)", "early 0..50 ms"]]
M["PreFourMin"], M["PreFourMax"] = f1(min(pre4)), f1(max(pre4))
M["OffCycleN"] = ct["n_irregular"]; M["OffCycleMod"] = f1(W["standard -100..500 ms"]["modality"]["acc_irregular"])
M["OffCycleFour"] = f1(W["standard -100..500 ms"]["4class"]["acc_irregular"])
M["CycleFrac"] = f1(ct["frac_in_cycle"]); M["PrevPred"] = f1(ct["prev_label_predictability"]["modality"])
M["EogMod"], M["EogVis"], M["EogFour"] = f1(rv["eog"]["modality"]["acc"]), f1(rv["eog"]["vis_lat"]["acc"]), f1(rv["eog"]["4class"]["acc"])
M["EogRegMod"], M["EogRegVis"] = f1(rv["eog"]["after_regression"]["modality"]), f1(rv["eog"]["after_regression"]["vis_lat"])
M["FrontMod"], M["FrontVis"], M["FrontAud"] = f1(fe["modality"]), f1(fe["vis_lat"]), f1(fe["aud_lat"])
M["FrontN"] = fe["n_removed"]
M["PermThousandP"] = f"{rv['perm1000']['p']:.3f}"; M["PermNullMax"] = f1(rv["perm1000"]["null_max"])
M["GdTen"] = f"{rv['group_delay_ms']['10']:.0f}"; M["GdFive"] = f"{rv['group_delay_ms']['5']:.0f}"
lc = rv["latency_causal_vs_zerophase_ms"]
M["ShiftN"] = f"{lc['AEP_N1'][0] - lc['AEP_N1'][1]:.1f}"; M["ShiftP"] = f"{lc['VEP_P1'][0] - lc['VEP_P1'][1]:.1f}"
M["ZpN"] = f"{lc['AEP_N1'][1]:.0f}"; M["ZpP"] = f"{lc['VEP_P1'][1]:.0f}"
fd = rv["fair_detector"]
M["FairNThirtyTwo"] = f1(fd["AEP N1"]["32"]["det_rate"]); M["FairNFa"] = f1(max(v["fa_rate"] for v in fd["AEP N1"].values()))
M["FairNSixteen"] = f1(fd["AEP N1"]["16"]["det_rate"])
M["FairPThirtyTwo"] = f1(fd["VEP P1"]["32"]["det_rate"]); M["FairPFa"] = f1(fd["VEP P1"]["32"]["fa_rate"])

# ------------------------------------------------ detection figure: fair detector + learning curve
fig, axs = plt.subplots(1, 2, figsize=(W1, 1.8), gridspec_kw=dict(wspace=0.6))
ax = axs[0]
for comp, col, mk in [("AEP N1", BLUE, "o"), ("VEP P1", ORANGE, "s")]:
    rows_ = fd[comp]; ks = [int(k) for k in rows_]
    ax.plot(ks, [rows_[str(k)]["det_rate"] * 100 for k in ks], marker=mk, ms=3.5, color=col, lw=1.4, label=f"{comp} hit")
    ax.plot(ks, [rows_[str(k)]["fa_rate"] * 100 for k in ks], marker=mk, ms=3.5, color=col, lw=1.0, ls="--", mfc="white", label=f"{comp} FA")
ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
ax.set_xlabel("Trials averaged, $k$"); ax.set_ylabel("Rate (%)"); ax.set_ylim(-3, 103)
ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, handlelength=1.6, columnspacing=0.8)
ax.set_title("(a) 3-SD criterion, $\\pm$-ref. noise", fontsize=7.5, loc="left")
ax = axs[1]
L = res["learning_curve"]; n = [int(k) for k in L]
mu = np.array([L[str(k)][0] for k in n]) * 100; sd = np.array([L[str(k)][1] for k in n]) * 100
ax.fill_between(n, mu - sd, mu + sd, color=BLUE, alpha=0.2, lw=0); ax.plot(n, mu, marker="o", ms=3.5, color=BLUE, lw=1.4)
ax.set_xlabel("Calibration trials"); ax.set_ylabel("Modality acc. (%)"); ax.set_title("(b) Learning curve", fontsize=7.5, loc="left")
fig.savefig(os.path.join(out, "fig_det.pdf")); fig.savefig(os.path.join(out, "fig_det.png")); plt.close(fig)

# ------------------------------------------------ robot figure
rows_r = list(csv.DictReader(open(os.path.join(rev_dir, "robot.csv"))))
cal_end = res["chronological"]["4class"]["cal_end_sample"]
held = [r for r in rows_r if int(r["onset"]) > cal_end]
acts = ["forward", "left", "right"]
cm = np.zeros((3, 3))
for r in held:
    cm[acts.index(r["intended"]), acts.index(r["action"])] += 1
oc = np.array([(int(r["t_cmd_ns"]) - int(r["t_onset_ns"])) / 1e6 for r in rows_r])
dc = np.array([(int(r["t_cmd_ns"]) - int(r["t_decision_ns"])) / 1e6 for r in rows_r])
dec = np.genfromtxt(os.path.join(rev_dir, "robot_decisions.csv"), delimiter=",", names=True, dtype=None, encoding=None)
fig = plt.figure(figsize=(W1, 1.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.55)
ax = fig.add_subplot(gs[0])
last = (round(0.5 * res["fs"]) - 1) / res["fs"] * 1e3
ax.hist(oc, bins=np.arange(490, 524, 1.5), color=BLUE, edgecolor="white", lw=0.4)
ax.axvline(last, color=INK, lw=0.8, ls="--")
ax.text(last + 0.5, ax.get_ylim()[1] * 0.97, "last\nepoch\nsample", fontsize=5.5, va="top", ha="left")
ax.axvline(oc.mean(), color=ORANGE, lw=1.0)
ax.text(oc.mean() + 0.6, ax.get_ylim()[1] * 0.92, f"mean\n{oc.mean():.0f} ms", fontsize=6, va="top", color=INK)
ax.set_xlabel("Stimulus onset $\\rightarrow$ robot command (ms)"); ax.set_ylabel("Trials")
ax.set_title("(a) Command latency (all trials)", fontsize=7.5, loc="left")
ax = fig.add_subplot(gs[1])
pct = cm / cm.sum(1, keepdims=True) * 100
ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7, color="white" if pct[i, j] > 60 else INK)
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(["fwd", "left", "right"], fontsize=6.5)
ax.set_yticklabels(["fwd", "left", "right"], fontsize=6.5); ax.set_xlabel("Executed"); ax.set_ylabel("Intended")
ax.grid(False); ax.set_title("(b) Held-out actions", fontsize=7.5, loc="left")
fig.savefig(os.path.join(out, "fig_robot.pdf")); fig.savefig(os.path.join(out, "fig_robot.png")); plt.close(fig)
M["RobotN"] = len(rows_r); M["RobotHeldN"] = len(held)
M["RobotAcc"] = f1(np.trace(cm) / cm.sum()); M["RobotErr"] = int(cm.sum() - np.trace(cm))
M["RobotCmd"] = f"{dc.mean():.2f}"; M["RobotCmdP"] = f"{np.percentile(dc, 95):.2f}"
M["RobotOnset"] = f"{oc.mean():.0f}"; M["RobotOnsetSd"] = f"{oc.std():.1f}"; M["RobotOnsetMax"] = f"{oc.max():.0f}"
M["LastSample"] = f"{(round(0.5 * res['fs']) - 1) / res['fs'] * 1e3:.0f}"
M["RobotExcess"] = f"{dec['excess_ms'].mean():.1f}"
with open(os.path.join(out, "numbers_rev.tex"), "w") as fh:
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(M)
