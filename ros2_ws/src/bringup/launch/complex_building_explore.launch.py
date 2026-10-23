#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复杂建筑专用探索入口（窄通道 / 多房间 / 3d.world）。

与房间场景入口 exploration_bringup.launch.py 的区别（本文件只覆盖以下默认值）：
  1) world = 3d.world：加载复杂建筑场景；已删除原 3d.world 里的口字型封闭外墙，
     建筑外壳实际到 x≈±18.7、y≈-18.3~14.8，车可以进入建筑大半区域；
  2) inflation_radius_cells = 3.0：窄通道下把障碍膨胀从 5 格（0.25m）收到 3 格
     （0.15m），否则门洞/走廊会被膨胀封死；阿克曼轨迹的 safety_cells 复用同一值；
  3) treat_unknown_as_obstacle = True：规划时未知格按硬障碍处理，避免车往未观测区域
     硬闯；候选前沿的可达性本来只在已知自由格上计算，因此不会误杀合法前沿；
  4) use_eight_neighbors = True：8 邻接减少阶梯折线，转弯更少、更贴通道走向；
  5) path_block_check_m = 1.0：缓存路径前 1m 出现新观测障碍就立即重规划、跳过迟滞；
  6) max_exploration_time_sec = 3000s：复杂建筑面积大，放宽总时长；
  7) stall_detect_sec = 20s：窄通道里脱困/三点掉头耗时较长，避免过早放弃目标；
  8) max_goal_count = 800、goal_count_min_move_m = 2.0：复杂建筑（3d.world）尺度
     远大于单房间，实测 200 个目标预算在 sim 1274s 就耗尽并提前报"达到最大目标数"，
     而此时覆盖率仅 31.49%、东侧 x>5 仍 100% 未知。因此把目标预算放大到 800；
     同时把"里程碑计数"的最小位移从 1.0m 提到 2.0m——建筑尺度下前沿小幅挪动
     （<2m）不代表真正的探索推进，过滤得更狠才不会把预算浪费在局部抖动上；
  9) exploration_bounds = "-19.0,19.5,-19.0,15.5"：把候选前沿限制在建筑外壳内，
     避免 16% 的时间耗在建筑外西侧空地上（实测最后目标 (-20.49,-2.09) 在建筑外）。

其余参数（目标迟滞、黑名单、路径跟随安全状态机等）全部透传 exploration_bringup
的默认值，可用命令行 ``名字:=值`` 覆盖。禁止与 nav_slam/launch/2dpoints.launch.py
同时运行（两者都会发布 /map 与坐标变换）。

建图可视化：本文件默认启动 bringup/exploration_viz（matplotlib 窗口），实时显示
栅格地图（灰=未知 / 白=自由 / 黑=障碍）、机器人位姿与轨迹、当前目标点、规划路径。
无显示环境请加 ``headless:=true viz:=false``；只想静默出一张图可另行用
``MPLBACKEND=Agg python3 -m bringup.exploration_viz --ros-args -p save_once:=/tmp/viz.png``。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_launch_path = os.path.join(
        get_package_share_directory("bringup"), "launch", "exploration_bringup.launch.py"
    )

    declared_arguments = [
        # ---- 复杂建筑场景覆盖项 ----
        DeclareLaunchArgument(
            "world",
            default_value="3d.world",
            description="要加载的世界文件名（gazebo_modele/world/ 下）",
        ),
        # headless/software_gl 透传（无显示环境验证用 headless:=true；
        # 有显示环境观察 Gazebo 用默认 false）。
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="true 时只起 gzserver、不起 gzclient（无显示环境用）",
        ),
        DeclareLaunchArgument(
            "software_gl",
            default_value="false",
            description="true 时对 Gazebo 进程设 LIBGL_ALWAYS_SOFTWARE=1（软件 OpenGL）",
        ),
        DeclareLaunchArgument(
            "inflation_radius_cells",
            default_value="3.0",
            description="障碍硬膨胀半径（格）：窄通道取 3 格（0.15m），同时作为轨迹 safety_cells",
        ),
        DeclareLaunchArgument(
            "treat_unknown_as_obstacle",
            default_value="True",
            description="规划时未知格(-1)当硬障碍，避免往未观测区域硬闯",
        ),
        DeclareLaunchArgument(
            "use_eight_neighbors",
            default_value="True",
            description="A* 使用 8 邻接（对角代价 sqrt(2)，禁止穿角）",
        ),
        DeclareLaunchArgument(
            "path_block_check_m",
            default_value="1.0",
            description="缓存路径前该距离内出现新障碍即立即重规划（跳过迟滞）",
        ),
        DeclareLaunchArgument(
            "max_exploration_time_sec",
            default_value="4500.0",
            description="复杂建筑总面积大，放宽探索总时长（s）；距离场上限化后决策吞吐提升约 3 倍，4500s 足以走到收敛",
        ),
        DeclareLaunchArgument(
            "stall_detect_sec",
            default_value="20.0",
            description="窄通道脱困耗时较长，放宽目标推进检测窗口（s）",
        ),
        DeclareLaunchArgument(
            "max_goal_count",
            default_value="800",
            description="最大目标数：复杂建筑尺度大，200 会在覆盖率仅 31% 时提前耗尽，放宽到 800",
        ),
        DeclareLaunchArgument(
            "goal_count_min_move_m",
            default_value="2.0",
            description="目标计数的最小有效位移（m）：建筑尺度下 <2m 的前沿挪动不算里程碑，避免浪费预算",
        ),
        # 修复 C：把探索限制在建筑外壳范围内，避免 16% 的时间耗在建筑外西侧空地上
        # （实测最后目标 (-20.49,-2.09) 已在建筑外）。传入空字符串可关闭过滤。
        DeclareLaunchArgument(
            "exploration_bounds",
            default_value="-19.0,19.5,-19.0,15.5",
            description=(
                "建筑外壳边界 \"xmin,xmax,ymin,ymax\"（米，世界坐标）；"
                "非空时只考虑边界内的前沿，避免在建筑外围空地上空转"
            ),
        ),
        # ---- 阿克曼轨迹参数（显式透传，默认值与路径跟随节点一致）----
        DeclareLaunchArgument(
            "enable_ackermann_traj",
            default_value="True",
            description="True 时发布阿克曼可行轨迹；False 退回原始 A* 折线（A/B 对比）",
        ),
        DeclareLaunchArgument("wheelbase_m", default_value="0.144"),
        DeclareLaunchArgument("max_steer_rad", default_value="0.489"),
        DeclareLaunchArgument("min_turn_radius_m", default_value="0.35"),
        DeclareLaunchArgument("traj_ds_m", default_value="0.05"),
        DeclareLaunchArgument("traj_v_max", default_value="0.30"),
        DeclareLaunchArgument("traj_v_min", default_value="0.08"),
        DeclareLaunchArgument("traj_a_lat_max", default_value="0.60"),
        DeclareLaunchArgument("traj_a_long_max", default_value="0.25"),
        # ---- 建图可视化（matplotlib 实时窗口）----
        # 与 exploration_demo.launch.py 同口径：栅格地图 + 机器人位姿/轨迹 + 目标点 + 路径。
        # 无显示环境（headless:=true）请同时置 viz:=false，否则 matplotlib 会因无 DISPLAY 报错。
        DeclareLaunchArgument(
            "viz",
            default_value="true",
            description="是否启动 matplotlib 建图可视化窗口（实时看地图/位姿/目标/路径）",
        ),
        DeclareLaunchArgument(
            "refresh_hz",
            default_value="5.0",
            description="可视化重绘频率（Hz）",
        ),
    ]

    exploration = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bringup_launch_path),
        launch_arguments={
            "world": LaunchConfiguration("world"),
            "headless": LaunchConfiguration("headless"),
            "software_gl": LaunchConfiguration("software_gl"),
            "inflation_radius_cells": LaunchConfiguration("inflation_radius_cells"),
            "treat_unknown_as_obstacle": LaunchConfiguration(
                "treat_unknown_as_obstacle"
            ),
            "use_eight_neighbors": LaunchConfiguration("use_eight_neighbors"),
            "path_block_check_m": LaunchConfiguration("path_block_check_m"),
            "max_exploration_time_sec": LaunchConfiguration(
                "max_exploration_time_sec"
            ),
            "stall_detect_sec": LaunchConfiguration("stall_detect_sec"),
            "max_goal_count": LaunchConfiguration("max_goal_count"),
            "goal_count_min_move_m": LaunchConfiguration("goal_count_min_move_m"),
            # 修复 C：把建筑外壳边界透传给 explorer。
            "exploration_bounds": LaunchConfiguration("exploration_bounds"),
            "enable_ackermann_traj": LaunchConfiguration("enable_ackermann_traj"),
            "wheelbase_m": LaunchConfiguration("wheelbase_m"),
            "max_steer_rad": LaunchConfiguration("max_steer_rad"),
            "min_turn_radius_m": LaunchConfiguration("min_turn_radius_m"),
            "traj_ds_m": LaunchConfiguration("traj_ds_m"),
            "traj_v_max": LaunchConfiguration("traj_v_max"),
            "traj_v_min": LaunchConfiguration("traj_v_min"),
            "traj_a_lat_max": LaunchConfiguration("traj_a_lat_max"),
            "traj_a_long_max": LaunchConfiguration("traj_a_long_max"),
        }.items(),
    )

    # 建图可视化节点：matplotlib 窗口实时显示
    #   栅格地图（灰=未知 / 白=自由 / 黑=障碍）+ 机器人位姿与轨迹 + 目标点 + 规划路径。
    # 与 exploration_demo.launch.py 完全同口径（use_sim_time=True），
    # 只是把入口挂到复杂建筑 launch 上，避免再单独起一套。
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

    return LaunchDescription(declared_arguments + [exploration, viz])
