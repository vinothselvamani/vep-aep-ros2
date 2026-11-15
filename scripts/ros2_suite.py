"""Runs the ROS 2 pipeline under several configurations and summarises latency/CPU.

Usage (inside a sourced ROS 2 environment):
  python ros2_suite.py <data.npz> <model_dir> <fs> <outdir> [--quick]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import psutil

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("model_dir"); ap.add_argument("fs"); ap.add_argument("outdir")
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
d = np.load(a.npz)
dur = len(d["data"]) / float(d["fs"])
off = {t: np.load(os.path.join(a.model_dir, f"offline_pred_{t}.npy")) for t in ["modality", "4class"]}

configs = [("modality", 1.0, 10), ("modality", 1.0, 20), ("modality", 1.0, 40), ("4class", 1.0, 20),
           ("modality", 5.0, 20), ("modality", 10.0, 20)]
if a.quick:
    configs = configs[:2]
summary = []
for task, speed, chunk in configs:
    tag = f"{task}_x{speed:g}_c{chunk}"
    csv = os.path.join(a.outdir, tag + ".csv")
    cmd = ["ros2", "launch", "vep_aep_ros2", "vep_aep.launch.py", f"data_file:={os.path.abspath(a.npz)}",
           f"model_file:={os.path.abspath(os.path.join(a.model_dir, f'model_{task}.npz'))}", f"fs:={a.fs}",
           f"task:={task}", f"speed:={speed}", f"chunk_ms:={chunk}", f"out_csv:={csv}"]
    log = open(csv.replace(".csv", ".log"), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    lp = psutil.Process(p.pid)
    cpu, rss, procs = {}, {}, {}
    t_end = time.time() + dur / speed + 30
    time.sleep(3.0)
    while p.poll() is None and time.time() < t_end:
        try:
            kids = lp.children(recursive=True)
        except psutil.Error:
            break
        for c in kids:
            try:
                name = next((n for n in ["eeg_streamer", "erp_preprocessor", "erp_detector", "erp_evaluator"]
                             if n in " ".join(c.cmdline())), None)
            except psutil.Error:
                continue
            if not name:
                continue
            if c.pid not in procs:
                procs[c.pid] = c; c.cpu_percent(None); continue
            try:
                pc = procs[c.pid]; cpu.setdefault(name, []).append(pc.cpu_percent(None)); rss.setdefault(name, []).append(pc.memory_info().rss / 2**20)
            except psutil.Error:
                pass
        time.sleep(1.0)
    if p.poll() is None:
        p.send_signal(2); p.wait(20)
    log.close()
    r = np.genfromtxt(csv, delimiter=",", names=True, dtype=None, encoding=None)
    o = off[task]
    sel = np.isin(r["onset"], o[:, 0])
    pred_map = dict(zip(r["onset"], r["pred"]))
    agree = float(np.mean([pred_map.get(s, -99) == pr for s, _, pr in o]))
    ex = r["excess_ms"]
    row = {"task": task, "speed": speed, "chunk_ms": chunk, "n_decisions": int(len(r)), "n_markers": int(len(d["events"])),
           "excess_ms_mean": float(ex.mean()), "excess_ms_sd": float(ex.std()), "excess_ms_p95": float(np.percentile(ex, 95)),
           "excess_ms_max": float(ex.max()), "total_ms_mean": float(r["total_ms"].mean()),
           "proc_us_mean": float(r["proc_us"].mean()), "proc_us_p95": float(np.percentile(r["proc_us"], 95)),
           "test_acc": float(np.mean(r["true"][sel] == r["pred"][sel])), "agree_offline": agree,
           "cpu_pct": {k: float(np.mean(v)) for k, v in cpu.items()}, "rss_mb": {k: float(np.mean(v)) for k, v in rss.items()}}
    summary.append(row)
    print(json.dumps(row))
json.dump(summary, open(os.path.join(a.outdir, "ros2_summary.json"), "w"), indent=1)
