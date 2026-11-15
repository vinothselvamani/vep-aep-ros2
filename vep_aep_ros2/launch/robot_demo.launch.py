"""Closed demo: EEG replay -> VEP/AEP detector -> turtlesim robot.

ros2 launch vep_aep_ros2 robot_demo.launch.py data_file:=... model_file:=... fs:=150.15 task:=4class
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue as PV


def generate_launch_description():
    args = [DeclareLaunchArgument(n, default_value=v) for n, v in [
        ("data_file", ""), ("model_file", ""), ("fs", "150.15"), ("task", "4class"), ("speed", "1.0"),
        ("chunk_ms", "20.0"), ("out_csv", "decisions_robot.csv"), ("robot_csv", "robot.csv")]]
    streamer = Node(package="vep_aep_ros2", executable="eeg_streamer", parameters=[{
        "data_file": LC("data_file"), "speed": PV(LC("speed"), value_type=float),
        "chunk_ms": PV(LC("chunk_ms"), value_type=float)}])
    return LaunchDescription(args + [
        Node(package="turtlesim", executable="turtlesim_node", name="turtlesim"),
        Node(package="vep_aep_ros2", executable="erp_robot_bridge", parameters=[{
            "out_csv": LC("robot_csv"), "task": LC("task")}]),
        Node(package="vep_aep_ros2", executable="erp_preprocessor",
             parameters=[{"fs": PV(LC("fs"), value_type=float)}]),
        Node(package="vep_aep_ros2", executable="erp_detector", parameters=[{
            "model_file": LC("model_file"), "task": LC("task")}]),
        Node(package="vep_aep_ros2", executable="erp_evaluator", parameters=[{
            "out_csv": LC("out_csv"), "task": LC("task"), "speed": PV(LC("speed"), value_type=float),
            "fs": PV(LC("fs"), value_type=float)}]),
        TimerAction(period=2.0, actions=[streamer]),
        RegisterEventHandler(OnProcessExit(target_action=streamer, on_exit=[
            TimerAction(period=5.0, actions=[EmitEvent(event=Shutdown(reason="recording finished"))])])),
    ])
