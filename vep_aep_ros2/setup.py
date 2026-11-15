from glob import glob

from setuptools import setup

package_name = "vep_aep_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vinoth S",
    description="ROS 2 VEP/AEP detection pipeline",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "eeg_streamer = vep_aep_ros2.eeg_streamer:main",
            "erp_preprocessor = vep_aep_ros2.erp_preprocessor:main",
            "erp_detector = vep_aep_ros2.erp_detector:main",
            "erp_evaluator = vep_aep_ros2.erp_evaluator:main",
            "erp_robot_bridge = vep_aep_ros2.erp_robot_bridge:main",
        ],
    },
)
