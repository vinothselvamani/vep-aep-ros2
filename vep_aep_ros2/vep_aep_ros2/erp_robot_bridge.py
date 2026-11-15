"""Turns evoked-potential decisions into robot commands (demonstrated with turtlesim).

Subscribes /erp/decision (JSON String) and publishes geometry_msgs/Twist on /turtle1/cmd_vel:
  AEP (either ear)  -> move forward         (modality is decoded reliably)
  VEP-left / -right -> turn left / right    (visual laterality is decoded reliably)
  modality task: AEP -> forward, VEP -> turn left
It also subscribes /turtle1/pose and logs the actuation latency: time from publishing the command to
the first pose message whose velocity reflects it.
"""
import csv
import json
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from vep_aep_ros2.eeg_streamer import QOS

try:
    from turtlesim.msg import Pose
except ImportError:  # turtlesim not installed: commands are still published and logged
    Pose = None

ACTION = {"AEP": "forward", "VEP": "left", "AEP-left": "forward", "AEP-right": "forward",
          "VEP-left": "left", "VEP-right": "right"}
INTENDED = {"modality": {1: "forward", 2: "forward", 3: "left", 4: "left"},
            "4class": {1: "forward", 2: "forward", 3: "left", 4: "right"}}


class ERPRobotBridge(Node):
    def __init__(self):
        super().__init__("erp_robot_bridge")
        self.declare_parameter("out_csv", "robot.csv")
        self.declare_parameter("task", "4class")
        self.declare_parameter("linear", 0.6)
        self.declare_parameter("angular", 1.2)
        self.task = self.get_parameter("task").value
        self.lin, self.ang = self.get_parameter("linear").value, self.get_parameter("angular").value
        self.pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.create_subscription(String, "/erp/decision", self.on_decision, QOS)
        if Pose is not None:
            self.create_subscription(Pose, "/turtle1/pose", self.on_pose, 50)
        self.pending = None
        self.pose = (float("nan"),) * 3
        self.f = open(self.get_parameter("out_csv").value, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["onset", "code", "label", "action", "intended", "t_onset_ns", "t_decision_ns", "t_recv_ns",
                         "t_cmd_ns", "t_pose_ns", "x", "y", "theta"])
        self.n = self.ok = 0

    def on_decision(self, msg):
        d = json.loads(msg.data)
        t_recv = time.time_ns()
        action = ACTION[d["label"]]
        tw = Twist()
        if action == "forward":
            tw.linear.x = self.lin
        else:
            tw.angular.z = self.ang if action == "left" else -self.ang
        self.flush_pending()
        self.pub.publish(tw)
        t_cmd = time.time_ns()
        intended = INTENDED[self.task][d["code"]]
        self.pending = [d["onset"], d["code"], d["label"], action, intended, d["t_onset_ns"], d["t_decision_ns"],
                        t_recv, t_cmd, None, tw]
        self.n += 1
        self.ok += int(action == intended)
        if self.n % 20 == 0:
            self.get_logger().info(f"{self.n} commands, correct action {self.ok / self.n:.3f}")
        if Pose is None:
            self.flush_pending()

    def on_pose(self, p):
        self.pose = (p.x, p.y, p.theta)
        if self.pending is not None and self.pending[9] is None:
            tw = self.pending[10]
            moving = abs(p.linear_velocity - tw.linear.x) < 1e-3 and abs(p.angular_velocity - tw.angular.z) < 1e-3
            if moving:
                self.pending[9] = time.time_ns()
                self.flush_pending()

    def flush_pending(self):
        if self.pending is None:
            return
        row = self.pending[:9] + [self.pending[9] if self.pending[9] else ""] + list(self.pose)
        self.w.writerow(row)
        self.f.flush()
        self.pending = None


def main():
    rclpy.init()
    n = ERPRobotBridge()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.flush_pending()
    n.f.close()
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
