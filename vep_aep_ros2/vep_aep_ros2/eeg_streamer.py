"""Replays a recorded EEG file as a real-time stream (amplifier emulator).

Publishes
  /eeg/raw      std_msgs/Float32MultiArray  chunk (n_samples x n_ch); layout.data_offset = first sample index
  /eeg/markers  std_msgs/Int64MultiArray    [onset_sample, event_code, t_onset_ns]
t_onset_ns is the wall-clock time at which the onset sample was acquired (chunk publish time minus
the remaining samples of that chunk),
so downstream nodes can measure end-to-end latency.
"""
import time
from array import array

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Int64MultiArray, MultiArrayDimension

QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=200)


class EEGStreamer(Node):
    def __init__(self):
        super().__init__("eeg_streamer")
        self.declare_parameter("data_file", "")
        self.declare_parameter("chunk_ms", 20.0)
        self.declare_parameter("speed", 1.0)
        self.declare_parameter("start_sample", 0)
        d = np.load(self.get_parameter("data_file").value)
        self.x, self.fs, self.ev = d["data"].astype(np.float32), float(d["fs"]), d["events"]
        self.chunk = max(1, int(round(self.get_parameter("chunk_ms").value * 1e-3 * self.fs)))
        self.speed = float(self.get_parameter("speed").value)
        self.pos = int(self.get_parameter("start_sample").value)
        self.ev = self.ev[self.ev[:, 0] >= self.pos]
        self.ev_i = 0
        self.pub = self.create_publisher(Float32MultiArray, "/eeg/raw", QOS)
        self.pub_m = self.create_publisher(Int64MultiArray, "/eeg/markers", QOS)
        self.t0 = None
        self.timer = self.create_timer(self.chunk / self.fs / self.speed / 4, self.tick)  # oversampled clock
        self.get_logger().info(f"streaming {self.x.shape[1]} ch @ {self.fs:.2f} Hz, chunk={self.chunk} samples, speed x{self.speed}")

    def tick(self):
        if self.t0 is None:
            # wait until the preprocessor (raw) and detector (markers) have matched, then settle
            if self.pub.get_subscription_count() < 1 or self.pub_m.get_subscription_count() < 1:
                return
            time.sleep(2.0)
            self.t0, self.s0 = time.monotonic(), self.pos
        due = self.s0 + int((time.monotonic() - self.t0) * self.fs * self.speed)
        while self.pos + self.chunk <= min(due, len(self.x)):
            seg = self.x[self.pos:self.pos + self.chunk]
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension(label="samples", size=seg.shape[0], stride=seg.size),
                              MultiArrayDimension(label="channels", size=seg.shape[1], stride=seg.shape[1])]
            msg.layout.data_offset = self.pos
            buf = array("f"); buf.frombytes(np.ascontiguousarray(seg, dtype=np.float32).tobytes())
            msg.data = buf
            self.pub.publish(msg)
            t_ns = time.time_ns()
            while self.ev_i < len(self.ev) and self.ev[self.ev_i, 0] < self.pos + self.chunk:
                s, c = self.ev[self.ev_i]
                # wall time at which the onset sample was 'acquired' (last sample of chunk = t_ns)
                t_on = t_ns - int((self.pos + self.chunk - 1 - s) / (self.fs * self.speed) * 1e9)
                self.pub_m.publish(Int64MultiArray(data=[int(s), int(c), t_on]))
                self.ev_i += 1
            self.pos += self.chunk
        if self.pos + self.chunk > len(self.x):
            self.get_logger().info("end of recording")
            self.timer.cancel()
            raise SystemExit


def main():
    rclpy.init()
    n = EEGStreamer()
    try:
        rclpy.spin(n)
    except (SystemExit, KeyboardInterrupt):
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
