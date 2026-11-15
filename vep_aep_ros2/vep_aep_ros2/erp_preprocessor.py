"""Causal band-pass (Butterworth, SOS with persistent state) + common average reference.

Subscribes /eeg/raw, publishes /eeg/filtered (same layout, same data_offset).
"""
from array import array

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from vep_aep_ros2.core import CausalBandpass, car
from vep_aep_ros2.eeg_streamer import QOS


class ERPPreprocessor(Node):
    def __init__(self):
        super().__init__("erp_preprocessor")
        self.declare_parameter("fs", 150.15)
        self.declare_parameter("lo", 1.0)
        self.declare_parameter("hi", 30.0)
        self.declare_parameter("order", 4)
        self.declare_parameter("drop_prob", 0.0)  # fault injection: drop filtered chunks with this probability
        self.declare_parameter("seed", 0)
        self.drop = float(self.get_parameter("drop_prob").value)
        self.rng = np.random.default_rng(int(self.get_parameter("seed").value))
        self.n_dropped = 0
        self.bp = None
        self.pub = self.create_publisher(Float32MultiArray, "/eeg/filtered", QOS)
        self.create_subscription(Float32MultiArray, "/eeg/raw", self.cb, QOS)

    def cb(self, msg):
        n_s, n_ch = msg.layout.dim[0].size, msg.layout.dim[1].size
        x = np.frombuffer(msg.data, dtype=np.float32).astype(np.float64).reshape(n_s, n_ch)
        if self.bp is None:
            g = self.get_parameter
            self.bp = CausalBandpass(g("fs").value, n_ch, g("lo").value, g("hi").value, g("order").value)
        y = car(self.bp(x)).astype(np.float32)
        if self.drop > 0 and self.rng.random() < self.drop:
            self.n_dropped += 1  # filter state is kept; only the message is lost
            return
        out = Float32MultiArray(layout=msg.layout)
        buf = array("f"); buf.frombytes(y.tobytes())
        out.data = buf
        self.pub.publish(out)


def main():
    rclpy.init()
    n = ERPPreprocessor()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
