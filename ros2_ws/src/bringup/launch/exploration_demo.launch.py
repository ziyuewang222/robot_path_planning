#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""“一条命令看全套” demo 入口：Gazebo 里看车 + matplotlib 里看建图/选点。

本 launch 在已有的 Step 4 一键联调入口（bringup/launch/exploration_bringup.launch.py：
Gazebo 仿真 + Cartographer 建图 + 前沿探索 + 路径跟随）之上，只多加一个可视化节点
bringup/exploration_viz，从而做到**一条命令同时看到**：
  (1) gzclient 窗口里小车的真实运动（Gazebo 看车）；
  (2) matplotlib 窗口里实时栅格地图 + 机器人位姿/轨迹 + 目标点 + 路径（看建图/选点）。

组成：
  - IncludeLaunchDescription(bringup 自己的 exploration_bringup.launch.py)，并转传
    algorithm_root / map_save_prefix / headless 三个参数；
  - Node(bringup/exploration_viz)：受 IfCondition(viz) 控制，参数 use_sim_time=True、
    refresh_hz 由 launch argument 决定。

参数：
  - algorithm_root   | 环境变量 REGIONAL_LEGACY_ROOT，为空回落 /home/wzy/robot_path_planning/regional_legacy
  - map_save_prefix  | os.path.join(os.getcwd(), "maps", "exploration_map")
  - headless         | "false"（默认开 GUI，才能看到 Gazebo 里的小车；无显示时用 headless:=true）
  - viz              | "true"（是否启动 matplotlib 可视化节点）
  - refresh_hz       | "5.0"（可视化重绘频率）

⚠️ 若窗口没显示（远程 SSH / 无显示 / 沙箱内无 X/GLX 环境）：
   改用 ``headless:=true viz:=false`` —— 只跑仿真与算法闭环，不弹任何窗口；
   若只想静默出一张 2D 图做自检，可另行用
   ``MPLBACKEND=Agg python3 -m bringup.exploration_viz --ros-args -p save_once:=/tmp/viz.png``。

示例：
  有显示：ros2 launch bringup exploration_demo.launch.py
  无显示：ros2 launch bringup exploration_demo.launch.py headless:=true viz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # bringup 自己的联调入口（被本文件 include）。
    bringup_launch_path = os.path.join(
        get_package_share_directory("bringup"), "launch", "exploration_bringup.launch.py"
    )

    # ---- 默认值（对齐 exploration_bringup.launch.py，均可被 launch argument 覆盖）----
    # algorithm_root：优先环境变量 REGIONAL_LEGACY_ROOT，为空则回落到固定路径。
    default_algorithm_root = os.environ.get(
        "REGIONAL_LEGACY_ROOT"
    ) or "/home/wzy/robot_path_planning/regional_legacy"
    # map_save_prefix：探索完成时写图的绝对路径前缀（基于当前工作目录）。
    default_map_save_prefix = os.path.join(os.getcwd(), "maps", "exploration_map")

    declared_arguments = [
        DeclareLaunchArgument(
            "algorithm_root",
            default_value=default_algorithm_root,
            description="上游探索器所需的 regional_legacy 算法目录",
        ),
        DeclareLaunchArgument(
            "map_save_prefix",
            default_value=default_map_save_prefix,
            description="探索完成时的存图前缀（绝对路径）",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="true 时只起 gzserver、不起 gzclient（无显示环境用，看不到 Gazebo 窗口）",
        ),
        DeclareLaunchArgument(
            "software_gl",
            default_value="false",
            description="true 时对 Gazebo 进程设 LIBGL_ALWAYS_SOFTWARE=1（走软件 OpenGL，绕开显卡/GLX 驱动问题）",
        ),
        DeclareLaunchArgument(
            "viz",
            default_value="true",
            description="是否启动 matplotlib 可视化节点（无显示环境可置 false）",
        ),
        DeclareLaunchArgument(
            "refresh_hz",
            default_value="5.0",
            description="可视化节点重绘频率（Hz）",
        ),
    ]

    # 1) 一键联调闭环：仿真 + 建图 + 探索 + 跟随（转传 algorithm_root / map_save_prefix / headless / software_gl）。
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bringup_launch_path),
        launch_arguments={
            "algorithm_root": LaunchConfiguration("algorithm_root"),
            "map_save_prefix": LaunchConfiguration("map_save_prefix"),
            "headless": LaunchConfiguration("headless"),
            "software_gl": LaunchConfiguration("software_gl"),
        }.items(),
    )

    # 2) 实时可视化节点：matplotlib 看建图/选点（受 IfCondition(viz) 控制）。
    viz = Node(
        package="bringup",
        executable="exploration_viz",
        name="exploration_viz",
        output="screen",
        condition=IfCondition(LaunchConfiguration("viz")),
        parameters=[
            {
                "use_sim_time": True,
                "refresh_hz": LaunchConfiguration("refresh_hz"),
            }
        ],
    )

    return LaunchDescription(declared_arguments + [bringup, viz])
