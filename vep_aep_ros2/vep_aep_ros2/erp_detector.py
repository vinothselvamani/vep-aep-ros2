"""Stimulus-locked epoching and single-trial VEP/AEP classification (xDAWN + shrinkage LDA).

Subscribes /eeg/filtered and /eeg/markers. When the post-stimulus window of a marker is
complete in the ring buffer, the epoch is baseline-corrected, spatially filtered and classified.
Publishes /erp/decision std_msgs/String (JSON):
  {onset, code, pred, label, score, t_onset_ns, t_ready_ns, t_decision_ns, proc_us}
"""
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from scipy import signal
from std_msgs.msg import Float32MultiArray, Int64MultiArray, String

from vep_aep_ros2.core import RingBuffer, make_epoch
from vep_aep_ros2.eeg_streamer import QOS

LABELS = {0: "AEP", 1: "VEP", 1 + 10: "AEP-left", 2 + 10: "AEP-right", 3 + 10: "VEP-left", 4 + 10: "VEP-right"}


class ERPDetector(Node):
    def __init__(self):
        super().__init__("erp_detector")
        self.declare_parameter("model_file", "")
        self.declare_parameter("task", "modality")
        m = np.load(self.get_parameter("model_file").value)
        self.W, self.coef, self.b, self.classes = m["W"], m["coef"], m["intercept"], m["classes"]
        self.decim, fs = int(m["decim"]), float(m["fs"])
        self.n_pre, self.n_post = int(round(-float(m["tmin"]) * fs)), int(round(float(m["tmax"]) * fs))
        self.task = self.get_parameter("task").value
        self.rb = None
        self.pending = []  # [onset, code, t_onset_ns]
        self.gaps = []
        self.pub = self.create_publisher(String, "/erp/decision", QOS)
        try:  # typed decision message, if the vep_aep_msgs package is built
            from vep_aep_msgs.msg import ErpDecision
            self.ErpDecision = ErpDecision
            self.pub_typed = self.create_publisher(ErpDecision, "/erp/decision_typed", QOS)
        except ImportError:
            self.pub_typed = None
        self.create_subscription(Float32MultiArray, "/eeg/filtered", self.on_data, QOS)
        self.create_subscription(Int64MultiArray, "/eeg/markers", self.on_marker, QOS)

    def on_marker(self, msg):
        self.pending.append(list(msg.data))
        self.try_classify()

    def on_data(self, msg):
        n_s, n_ch = msg.layout.dim[0].size, msg.layout.dim[1].size
        if self.rb is None:
            self.rb = RingBuffer(n_ch, 8 * (self.n_pre + self.n_post))
            self.rb.n_written = msg.layout.data_offset
        gap = msg.layout.data_offset - self.rb.n_written
        if gap > 0:  # chunks lost in transport: keep absolute indexing aligned, flag gap
            self.get_logger().warning(f"gap of {gap} samples before {msg.layout.data_offset}")
            self.gaps.append((self.rb.n_written, msg.layout.data_offset))
            self.rb.push(np.full((gap, n_ch), np.nan))
        self.rb.push(np.frombuffer(msg.data, dtype=np.float32).astype(np.float64).reshape(n_s, n_ch))
        self.try_classify()

    def classify(self, ep):
        Z = self.W @ ep
        if self.decim > 1:
            Z = signal.decimate(Z, self.decim, axis=-1, zero_phase=True)
        s = self.coef @ Z.ravel() + self.b
        if len(self.classes) == 2:
            return int(self.classes[int(s[0] > 0)]), float(s[0])
        return int(self.classes[int(np.argmax(s))]), float(np.max(s))

    def try_classify(self):
        if self.rb is None:
            return
        keep = []
        for onset, code, t_on in self.pending:
            seg = self.rb.get(onset - self.n_pre, onset + self.n_post)
            if seg is None:
                if onset + self.n_post > self.rb.n_written:
                    keep.append([onset, code, t_on])  # not complete yet
                continue
            if np.isnan(seg).any():
                self.get_logger().warning(f"epoch at {onset} overlaps a data gap; skipped")
                continue
            t_ready = time.time_ns()
            t0 = time.perf_counter()
            pred, score = self.classify(make_epoch(seg, self.n_pre))
            proc_us = (time.perf_counter() - t0) * 1e6
            label = LABELS[pred] if self.task == "modality" else LABELS[pred + 10]
            self.pub.publish(String(data=json.dumps({
                "onset": int(onset), "code": int(code), "pred": pred, "label": label, "score": score,
                "t_onset_ns": int(t_on), "t_ready_ns": t_ready, "t_decision_ns": time.time_ns(), "proc_us": proc_us})))
            if self.pub_typed is not None:
                m = self.ErpDecision()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = "eeg"
                m.onset_sample, m.true_code, m.predicted = int(onset), int(code), int(pred)
                m.label, m.score, m.classify_us = label, float(score), float(proc_us)
                m.onset_stamp.sec, m.onset_stamp.nanosec = int(t_on // 10**9), int(t_on % 10**9)
                self.pub_typed.publish(m)
        self.pending = keep


def main():
    rclpy.init()
    n = ERPDetector()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
