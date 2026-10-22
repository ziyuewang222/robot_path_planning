from setuptools import setup

package_name = "bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/exploration_bringup.launch.py",
                "launch/complex_building_explore.launch.py",
                "launch/exploration_demo.launch.py",
                "launch/real_bringup.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="wzy",
    maintainer_email="wzy@todo.todo",
    description="系统级一键联调入口：仿真 + 建图 + 探索 + 路径跟随",
    license="MIT",
    entry_points={
        "console_scripts": [
            "exploration_viz = bringup.exploration_viz:main",
            "real_diagnostics = bringup.real_diagnostics:main",
            "cmd_vel_test = bringup.cmd_vel_test:main",
            "odom_tf_broadcaster = bringup.odom_tf_broadcaster:main",
            "twist_to_ackermann = bringup.twist_to_ackermann:main",
        ],
    },
)
