"""Single-process (no ROS 2) baseline with the same replay timing, processing chain and latency definition.

The difference between this excess latency and the ROS 2 excess latency is the middleware overhead
(serialisation, DDS transport across two hops and executor scheduling).

Usage: python baseline_single_process.py <data.npz> <model.npz> <out.csv> [--speed 1] [--chunk_ms 20]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vep_aep_ros2"))
from vep_aep_ros2.core import CausalBandpass, RingBuffer, car, make_epoch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--chunk_ms", type=float, default=20.0)
a = ap.parse_args()
d = np.load(a.npz)
x, fs, ev = d["data"].astype(np.float32), float(d["fs"]), d["events"]
m = np.load(a.model)
W, coef, b, classes, decim = m["W"], m["coef"], m["intercept"], m["classes"], int(m["decim"])
n_pre, n_post = int(round(-float(m["tmin"]) * fs)), int(round(float(m["tmax"]) * fs))
chunk = max(1, int(round(a.chunk_ms * 1e-3 * fs)))
bp = CausalBandpass(fs, x.shape[1], float(m["lo"]), float(m["hi"]), int(m["order"]))
rb = RingBuffer(x.shape[1], 8 * (n_pre + n_post))
pending, ev_i, pos, rows = [], 0, 0, []
t0 = time.monotonic()


def classify(ep):
    Z = W @ ep
    Z = signal.decimate(Z, decim, axis=-1, zero_phase=True) if decim > 1 else Z
    s = coef @ Z.ravel() + b
    return int(classes[int(s[0] > 0)]) if len(classes) == 2 else int(classes[int(np.argmax(s))])


while pos + chunk <= len(x):
    due = int((time.monotonic() - t0) * fs * a.speed)
    if pos + chunk > due:  # same oversampled clock as the ROS 2 streamer
        time.sleep(chunk / fs / a.speed / 4)
        continue
    seg = x[pos:pos + chunk].astype(np.float64)
    t_ns = time.time_ns()  # chunk "acquired"
    while ev_i < len(ev) and ev[ev_i, 0] < pos + chunk:
        s0, c = ev[ev_i]
        pending.append((int(s0), int(c), t_ns - int((pos + chunk - 1 - s0) / (fs * a.speed) * 1e9)))
        ev_i += 1
    rb.push(car(bp(seg)))
    pos += chunk
    keep = []
    for s0, c, t_on in pending:
        ep = rb.get(s0 - n_pre, s0 + n_post)
        if ep is None:
            if s0 + n_post > rb.n_written:
                keep.append((s0, c, t_on))
            continue
        t1 = time.perf_counter()
        pred = classify(make_epoch(ep, n_pre))
        proc = (time.perf_counter() - t1) * 1e6
        total = (time.time_ns() - t_on) / 1e6
        rows.append([s0, c, pred, total, total - (float(m["tmax"]) * 1e3 - 1e3 / fs) / a.speed, proc])
    pending = keep

with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["onset", "code", "pred", "total_ms", "excess_ms", "proc_us"]); w.writerows(rows)
e = np.array([r[4] for r in rows])
print(f"{len(rows)} decisions; excess {e.mean():.2f} +/- {e.std():.2f} ms (p95 {np.percentile(e, 95):.2f}); "
      f"classify {np.mean([r[5] for r in rows]) / 1e3:.2f} ms")
