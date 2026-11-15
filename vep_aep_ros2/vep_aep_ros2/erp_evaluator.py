"""Logs decisions and timing to CSV and prints running accuracy.

Latency definitions (all wall-clock, same host):
  total_ms   = t_decision - t_onset           (includes the unavoidable post-stimulus window)
  excess_ms  = t_decision - t(last epoch sample acquired)  (chunk buffering + transport + compute)
  proc_us    = classifier compute time inside erp_detector
"""
import csv
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from vep_aep_ros2.core import MODALITY
from vep_aep_ros2.eeg_streamer import QOS


class ERPEvaluator(Node):
    def __init__(self):
        super().__init__("erp_evaluator")
        self.declare_parameter("out_csv", "decisions.csv")
        self.declare_parameter("tmax", 0.5)
        self.declare_parameter("task", "modality")
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("fs", 150.15)
        self.fs = float(self.get_parameter("fs").value)
        self.speed = float(self.get_parameter("speed").value)
        self.tmax = float(self.get_parameter("tmax").value)
        self.task = self.get_parameter("task").value
        self.f = open(self.get_parameter("out_csv").value, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["onset", "code", "true", "pred", "label", "score", "total_ms", "excess_ms", "proc_us", "t_recv_ns"])
        self.n = self.ok = 0
        self.create_subscription(String, "/erp/decision", self.cb, QOS)

    def cb(self, msg):
        d = json.loads(msg.data)
        true = MODALITY[d["code"]] if self.task == "modality" else d["code"]
        total = (d["t_decision_ns"] - d["t_onset_ns"]) / 1e6
        self.n += 1
        self.ok += int(true == d["pred"])
        self.w.writerow([d["onset"], d["code"], true, d["pred"], d["label"], d["score"], total,
                         total - (self.tmax * 1e3 - 1e3 / self.fs) / self.speed, d["proc_us"], time.time_ns()])
        self.f.flush()
        if self.n % 10 == 0:
            self.get_logger().info(f"{self.n} decisions, acc={self.ok / self.n:.3f}, last latency={total:.1f} ms")


def main():
    rclpy.init()
    n = ERPEvaluator()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.f.close()
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
