#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 4 一键联调入口：仿真 + 建图 + 区域级探索 + 路径跟随。

本 launch 把整套系统串成一条命令，包含四个部件：
  1) gazebo_modele 的 launch/gazebo.launch.py —— 启动 Gazebo Classic 加载
     world/3d.world，spawn_entity 生成 fishbot 车体，并起 robot_state_publisher、
     joint_state_publisher_gui 与两条静态 TF（odom->base_footprint、
     base_footprint->base_link）。该文件内部已给 rsp / gui 设了 use_sim_time=True。
  2) nav_slam 的 launch/gmapping.launch.py —— 启动 Cartographer 建图
     （cartographer_node + occupancy_grid_node），内部同样设 use_sim_time=True。
  3) regional_explorer 的 frontier_explorer_node —— 订阅 /map 与 TF，周期输出前沿
     目标（exploration_goal）与到目标的初步 A* 路径（exploration_path）。
  4) path_follower 的 path_follower_node —— 订阅目标/路径，经前视点 + 航向比例控制
     生成 cmd_vel 驱动底盘行驶。

本 launch 面向仿真联调，全链路统一仿真时间（写死 use_sim_time=True），故不暴露该参数
（被 include 的 gazebo / cartographer launch 内部同样硬编码 True，避免出现半仿真不一致）。

所有参数都通过 DeclareLaunchArgument 暴露，可用命令行 ``名字:=值`` 覆盖；默认值对齐
真机基准（决策调度 1 Hz、车体安全包络 0.25 m = 5 格、最大线速度 0.3 m/s）。路径跟随
节点只覆盖 max_linear_vel / goal_tolerance，其余保持节点自身默认值。

启动顺序：先仿真/建图，后算法/控制。

警告：本文件与 nav_slam/launch/2dpoints.launch.py 互斥，禁止同时运行 —— 两者都会
发布 /map 与坐标变换，同跑会发生冲突。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- 默认值（对齐真机基准，均可被 launch argument 覆盖）----
    # algorithm_root：优先环境变量 REGIONAL_LEGACY_ROOT，为空则回落到固定路径。
    default_algorithm_root = os.environ.get(
        "REGIONAL_LEGACY_ROOT"
    ) or "/home/wzy/robot_path_planning/regional_legacy"
    # map_save_prefix：探索完成时写图的绝对路径前缀（基于当前工作目录）。
    default_map_save_prefix = os.path.join(os.getcwd(), "maps", "exploration_map")

    # 被包含的两个既有 launch（内部自设 use_sim_time=True）。
    gazebo_launch_path = os.path.join(
        get_package_share_directory("gazebo_modele"), "launch", "gazebo.launch.py"
    )
    gmapping_launch_path = os.path.join(
        get_package_share_directory("nav_slam"), "launch", "gmapping.launch.py"
    )

    declared_arguments = [
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="true 时只起 gzserver、不起 gzclient（无显示环境用）",
        ),
        DeclareLaunchArgument(
            "world",
            default_value="room.world",
            description="要加载的世界文件名（gazebo_modele/world/ 下），默认封闭矩形房间 room.world",
        ),
        DeclareLaunchArgument(
            "software_gl",
            default_value="false",
            description="true 时对 Gazebo 进程设 LIBGL_ALWAYS_SOFTWARE=1（走软件 OpenGL，绕开显卡/GLX 驱动问题）",
        ),
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
            "replan_rate_hz",
            default_value="1.0",
            description="探索器决策调度频率（Hz，真机基准 1 Hz）",
        ),
        DeclareLaunchArgument(
            "obstacle_safe_distance",
            default_value="5.0",
            description="车体安全包络半径（栅格，真机基准 0.25 m = 5 格）",
        ),
        DeclareLaunchArgument(
            "inflation_radius_cells",
            default_value="5.0",
            description="前向安全的障碍硬膨胀半径（格），真机安全包络 0.25m = 5 格",
        ),
        DeclareLaunchArgument(
            "min_frontier_distance",
            default_value="10.0",
            description="前沿到机器人的最小可达距离（格），避免选中贴在车身上的退化前沿",
        ),
        DeclareLaunchArgument(
            "max_linear_vel",
            default_value="0.3",
            description="控制节点最大线速度（m/s）",
        ),
        DeclareLaunchArgument(
            "goal_tolerance",
            default_value="0.25",
            description="到达判定半径（m）",
        ),
        DeclareLaunchArgument(
            "no_frontier_confirm_count",
            default_value="10",
            description="连续多少轮无目标才判定探索完成",
        ),
        DeclareLaunchArgument(
            "max_exploration_time_sec",
            default_value="1800.0",
            description="探索超时（s）",
        ),
        DeclareLaunchArgument(
            "target_hysteresis_sec",
            default_value="10.0",
            description="目标迟滞时长（s）：期内当前目标仍可达则沿用，不重新选点",
        ),
        DeclareLaunchArgument(
            "target_keep_verify",
            default_value="True",
            description="沿用当前目标前是否校验其仍是前沿候选",
        ),
        DeclareLaunchArgument(
            "max_goal_count",
            default_value="200",
            description="最大目标数（只统计相对上一个目标位移 ≥ goal_count_min_move_m 的里程碑式推进）",
        ),
        DeclareLaunchArgument(
            "goal_count_min_move_m",
            default_value="1.0",
            description="目标计数的最小有效位移（m）：小于该值的目标不计数，避免前沿小幅挪动耗光预算",
        ),
        DeclareLaunchArgument(
            "stall_detect_sec",
            default_value="45.0",
            description="机器人在该时长内位移小于阈值即视为目标推进不了（s）",
        ),
        DeclareLaunchArgument(
            "stall_move_eps",
            default_value="0.30",
            description="目标推进检测的位移阈值（m）",
        ),
        DeclareLaunchArgument(
            "target_blacklist_sec",
            default_value="180.0",
            description="被放弃目标的临时黑名单时长（s）",
        ),
        DeclareLaunchArgument(
            "target_blacklist_radius",
            default_value="1.0",
            description="黑名单半径（m）：该点周围这个半径内的前沿一并排除",
        ),
        DeclareLaunchArgument(
            "traj_fail_blacklist_count",
            default_value="2",
            description="同一目标连续多少次无法生成阿克曼轨迹后拉黑换目标（修复 B）",
        ),
        DeclareLaunchArgument(
            "exploration_bounds",
            default_value="",
            description=(
                "建筑外壳边界 \"xmin,xmax,ymin,ymax\"（米，世界坐标）；"
                "非空时过滤边界外前沿，空字符串表示不启用（修复 C）"
            ),
        ),
        # ---- 阿克曼轨迹生成 ----
        DeclareLaunchArgument(
            "enable_ackermann_traj",
            default_value="True",
            description="True 时把 A* 折线转成阿克曼可行轨迹再发布；False 退回原始折线（A/B 对比）",
        ),
        DeclareLaunchArgument(
            "wheelbase_m",
            default_value="0.144",
            description="车辆轴距（m），传入 AckermannTrajConfig",
        ),
        DeclareLaunchArgument(
            "max_steer_rad",
            default_value="0.489",
            description="前轮最大转角（rad），传入 AckermannTrajConfig",
        ),
        DeclareLaunchArgument(
            "min_turn_radius_m",
            default_value="0.35",
            description="最小转弯半径（m）；0.0 表示用 L/tan(max_steer)",
        ),
        DeclareLaunchArgument(
            "traj_ds_m",
            default_value="0.05",
            description="轨迹等弧长重采样步长（m）",
        ),
        DeclareLaunchArgument(
            "traj_v_max",
            default_value="0.30",
            description="轨迹/曲率限速上限（m/s）",
        ),
        DeclareLaunchArgument(
            "traj_v_min",
            default_value="0.08",
            description="轨迹/曲率限速下限（m/s）",
        ),
        DeclareLaunchArgument(
            "traj_a_lat_max",
            default_value="0.60",
            description="横向加速度上限（m/s²），决定弯道限速",
        ),
        DeclareLaunchArgument(
            "traj_a_long_max",
            default_value="0.25",
            description="纵向加速度上限（m/s²），速度剖面扫描",
        ),
        # ---- A* 搜索策略 ----
        DeclareLaunchArgument(
            "use_eight_neighbors",
            default_value="True",
            description="A* 是否用 8 邻接（对角代价 sqrt(2)，禁穿角）",
        ),
        DeclareLaunchArgument(
            "treat_unknown_as_obstacle",
            default_value="True",
            description="A* 是否把未知格(-1)当硬障碍；候选前沿可达性只在已知自由格上算，不会误杀",
        ),
        # ---- 路径受阻检查 ----
        DeclareLaunchArgument(
            "path_block_check_m",
            default_value="1.0",
            description="缓存路径前该距离内出现新障碍即立即重规划（跳过目标迟滞）",
        ),
    ]

    # 1) 仿真：gazebo + 车 + rsp + 静态 TF（include，转传 world / headless / software_gl）。
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_path),
        launch_arguments={
            "world": LaunchConfiguration("world"),
            "headless": LaunchConfiguration("headless"),
            "software_gl": LaunchConfiguration("software_gl"),
        }.items(),
    )

    # 2) 建图：Cartographer（include，不覆盖其内部参数）。
    gmapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gmapping_launch_path),
    )

    # 3) 区域级前沿探索节点（直接起节点，以便传参数）。
    frontier_explorer = Node(
        package="regional_explorer",
        executable="frontier_explorer_node",
        name="frontier_explorer",
        output="screen",
        parameters=[
            {
                # 仿真实时联调：全链路统一仿真时间，写死 True。
                "use_sim_time": True,
                "algorithm_root": LaunchConfiguration("algorithm_root"),
                "replan_rate_hz": LaunchConfiguration("replan_rate_hz"),
                "obstacle_safe_distance": LaunchConfiguration("obstacle_safe_distance"),
                "inflation_radius_cells": LaunchConfiguration("inflation_radius_cells"),
                "min_frontier_distance": LaunchConfiguration("min_frontier_distance"),
                "map_save_prefix": LaunchConfiguration("map_save_prefix"),
                "no_frontier_confirm_count": LaunchConfiguration("no_frontier_confirm_count"),
                "max_exploration_time_sec": LaunchConfiguration("max_exploration_time_sec"),
                "max_goal_count": LaunchConfiguration("max_goal_count"),
                "goal_count_min_move_m": LaunchConfiguration("goal_count_min_move_m"),
                "target_hysteresis_sec": LaunchConfiguration("target_hysteresis_sec"),
                "target_keep_verify": LaunchConfiguration("target_keep_verify"),
                "stall_detect_sec": LaunchConfiguration("stall_detect_sec"),
                "stall_move_eps": LaunchConfiguration("stall_move_eps"),
                "target_blacklist_sec": LaunchConfiguration("target_blacklist_sec"),
                "target_blacklist_radius": LaunchConfiguration("target_blacklist_radius"),
                # 修复 B：连续轨迹失败拉黑阈值。
                "traj_fail_blacklist_count": LaunchConfiguration(
                    "traj_fail_blacklist_count"
                ),
                # 修复 C：建筑外壳边界过滤。
                "exploration_bounds": LaunchConfiguration("exploration_bounds"),
                # 阿克曼轨迹生成。
                "enable_ackermann_traj": LaunchConfiguration("enable_ackermann_traj"),
                "wheelbase_m": LaunchConfiguration("wheelbase_m"),
                "max_steer_rad": LaunchConfiguration("max_steer_rad"),
                "min_turn_radius_m": LaunchConfiguration("min_turn_radius_m"),
                "traj_ds_m": LaunchConfiguration("traj_ds_m"),
                "traj_v_max": LaunchConfiguration("traj_v_max"),
                "traj_v_min": LaunchConfiguration("traj_v_min"),
                "traj_a_lat_max": LaunchConfiguration("traj_a_lat_max"),
                "traj_a_long_max": LaunchConfiguration("traj_a_long_max"),
                # A* 搜索策略。
                "use_eight_neighbors": LaunchConfiguration("use_eight_neighbors"),
                "treat_unknown_as_obstacle": LaunchConfiguration(
                    "treat_unknown_as_obstacle"
                ),
                # 路径受阻检查。
                "path_block_check_m": LaunchConfiguration("path_block_check_m"),
            }
        ],
    )

    # 4) 路径跟随控制节点（直接起节点，以便传参数）。
    path_follower = Node(
        package="path_follower",
        executable="path_follower_node",
        name="path_follower",
        output="screen",
        parameters=[
            {
                # 仿真实时联调：全链路统一仿真时间，写死 True。
                "use_sim_time": True,
                "max_linear_vel": LaunchConfiguration("max_linear_vel"),
                "goal_tolerance": LaunchConfiguration("goal_tolerance"),
                # 曲率限速：与上游轨迹参数保持一致。
                "traj_v_max": LaunchConfiguration("traj_v_max"),
                "traj_v_min": LaunchConfiguration("traj_v_min"),
                "a_lat_max": LaunchConfiguration("traj_a_lat_max"),
                # 可达曲率钳制：必须与 explorer/阿克曼轨迹用同一个最小转弯半径，
                # 否则上游生成的可行轨迹会被下层钳得更紧（或更松）。
                "min_turn_radius_m": LaunchConfiguration("min_turn_radius_m"),
            }
        ],
    )

    # 启动顺序：先仿真/建图，后算法/控制（不加延时）。
    return LaunchDescription(
        declared_arguments + [gazebo, gmapping, frontier_explorer, path_follower]
    )
