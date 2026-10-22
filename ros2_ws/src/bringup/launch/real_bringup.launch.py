#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真机（RDK-X5 小车，ROS2 Humble）一键启动入口：驱动 + 建图 + 探索 + 控制。

与仿真入口 exploration_bringup.launch.py 的关键区别：
  * **绝不 include Gazebo**：真机不跑仿真；所有节点 use_sim_time=False（真机没有 /clock，
    全链路必须是真实时间）。
  * 建图分辨率仍由 cartographer 的 occupancy grid 节点决定（0.05 m/格），本文件对整个
    /map 不做任何缩放 / 重采样。
  * 通过 start_driver 可选地把厂商驱动 launch include 进来；其余部件（Cartographer、
    前沿探索、路径跟随）本文件直接起节点，全部以 launch argument 形式接线。
  * 若驱动只发 /odom 话题、不发 odom->base_link 的 TF，把 publish_odom_tf 置 true，
    由本文件附带起 bringup/odom_tf_broadcaster 补上这段 TF。
  * 若底盘接受 AckermannDriveStamped 而非 Twis，把 use_ackermann_interface 置 true：
    path_follower 把 Twist 发到 twist_topic，再由 twist_to_ackermann 转成 ackermann_topic。

真机启动步骤（安全第一，务必按顺序）：
  1) **架空轮子**（或垫起驱动轮），先只通电 + 接底盘，确认物理急停可用；
  2) 启动驱动与本 launch（建议先只 start_driver:=true，确认 /scan /imu /odom 有数据）；
  3) **先手推建图**（不给电机使能）：确认 map->base_link 跟手推轨迹一致、建图能闭合，
     这一步能排除大部分 TF / 外参问题；
  4) 核对外参：laser_x/y/z 与 laser_roll/pitch/yaw、imu_x/y/z 与 imu_roll/pitch/yaw
     必须与实测安装一致（默认全 0.0 只是占位）；
  5) 低速自动跑：默认 max_linear_vel=0.15 m/s，人必须站在急停旁边，随时可断电；
  6) 逐步放开限速与探索参数。

安全提示：
  * 默认 max_linear_vel=0.15、use_ackermann_interface=false；第一次务必架空轮子。
  * 本 launch 只发 cmd_vel（或 ackermann），不代替物理急停；软件兜底（前向刹车、卡死自救）
    不能替代物理急停。
  * 若真机无 IMU 或 IMU 与 imu_link 不共位，请把 mapping_config 换成
    cartographer_real_noimu.lua，否则 Cartographer 会因 "IMU frame 必须与 tracking frame
    共位" 的 CHECK 崩溃。

启动顺序：driver（可选）→ TF/桥/建图 → 探索器 → 控制器 → viz。

本文件用 OpaqueFunction 构建：因为"是否启动 twist_to_ackermann / 用哪个 cmd_vel 话题"
这类分支需要在 launch 解析期读到确定值之后再决定。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _as_bool(value) -> bool:
    """把 launch 配置（字符串）读成 bool，兼容 "true"/"True"/"1" 等写法。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _setup(context):
    """在 launch 解析期根据已确定的参数值构建节点列表并返回。"""
    lc = context.launch_configurations

    def cfg(name):
        return lc.get(name)

    nav_slam_config_dir = os.path.join(
        get_package_share_directory("nav_slam"), "config"
    )

    actions = []

    # ------------------------------------------------------------------ 1) 驱动（可选）
    # start_driver=true 时用 PythonLaunchDescriptionSource +
    # PathJoinSubstitution([FindPackageShare(driver_package),"launch",driver_launch_file])
    # include 厂商驱动 launch。
    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare(LaunchConfiguration("driver_package")),
                    "launch",
                    LaunchConfiguration("driver_launch_file"),
                ]
            )
        ),
        condition=IfCondition(LaunchConfiguration("start_driver")),
    )
    actions.append(driver)

    # ------------------------------------------------------------------ 2) 建图（Cartographer）
    cartographer = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
        arguments=[
            "-configuration_directory", nav_slam_config_dir,
            "-configuration_basename", LaunchConfiguration("mapping_config"),
        ],
        remappings=[
            ("scan", cfg("scan_topic")),
            ("imu", cfg("imu_topic")),
            ("odom", cfg("odom_topic")),
        ],
    )
    actions.append(cartographer)

    # occupancy grid 节点：Foxy 上可执行名是 occupancy_grid_node，
    # Humble 上是 cartographer_occupancy_grid_node —— 用参数覆盖。
    occupancy_remaps = []
    if cfg("map_topic") != "map":
        occupancy_remaps.append(("map", cfg("map_topic")))
    occupancy = Node(
        package="cartographer_ros",
        executable=LaunchConfiguration("occupancy_grid_executable"),
        name="cartographer_occupancy_grid_node",
        output="screen",
        parameters=[{"use_sim_time": False}],
        remappings=occupancy_remaps,
    )
    actions.append(occupancy)

    # ------------------------------------------------------------------ 3) 静态 TF / 位姿桥
    # 注意：static_transform_publisher 的参数顺序为 x y z yaw pitch roll parent child。
    def _static_tf(name, x, y, z, roll, pitch, yaw, parent, child, cond):
        return Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=name,
            output="screen",
            arguments=[x, y, z, yaw, pitch, roll, parent, child],
            condition=IfCondition(cond),
        )

    actions.append(
        _static_tf(
            "static_laser_tf",
            LaunchConfiguration("laser_x"),
            LaunchConfiguration("laser_y"),
            LaunchConfiguration("laser_z"),
            LaunchConfiguration("laser_roll"),
            LaunchConfiguration("laser_pitch"),
            LaunchConfiguration("laser_yaw"),
            cfg("base_frame"),
            cfg("laser_frame"),
            LaunchConfiguration("publish_laser_tf"),
        )
    )
    actions.append(
        _static_tf(
            "static_imu_tf",
            LaunchConfiguration("imu_x"),
            LaunchConfiguration("imu_y"),
            LaunchConfiguration("imu_z"),
            LaunchConfiguration("imu_roll"),
            LaunchConfiguration("imu_pitch"),
            LaunchConfiguration("imu_yaw"),
            cfg("base_frame"),
            cfg("imu_frame"),
            LaunchConfiguration("publish_imu_tf"),
        )
    )
    actions.append(
        _static_tf(
            "static_base_footprint_tf",
            "0", "0", LaunchConfiguration("base_z"),
            "0", "0", "0",
            "base_footprint",
            cfg("base_frame"),
            LaunchConfiguration("publish_base_footprint_tf"),
        )
    )

    # 位姿桥：驱动只发 /odom 话题、不发 odom->base_link TF 时启用。
    odom_tf = Node(
        package="bringup",
        executable="odom_tf_broadcaster",
        name="odom_tf_broadcaster",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "odom_topic": cfg("odom_topic"),
                "odom_frame": cfg("odom_frame"),
                "base_frame": cfg("base_frame"),
            }
        ],
        condition=IfCondition(LaunchConfiguration("publish_odom_tf")),
    )
    actions.append(odom_tf)

    # ------------------------------------------------------------------ 4) 探索器
    frontier_explorer = Node(
        package="regional_explorer",
        executable="frontier_explorer_node",
        name="frontier_explorer",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "algorithm_root": cfg("algorithm_root"),
                "map_topic": cfg("map_topic"),
                "robot_frame": cfg("base_frame"),
                "global_frame": "map",
                "replan_rate_hz": cfg("replan_rate_hz"),
                "obstacle_safe_distance": cfg("obstacle_safe_distance"),
                "inflation_radius_cells": cfg("inflation_radius_cells"),
                "min_frontier_distance": cfg("min_frontier_distance"),
                "map_save_prefix": cfg("map_save_prefix"),
                "no_frontier_confirm_count": cfg("no_frontier_confirm_count"),
                "max_exploration_time_sec": cfg("max_exploration_time_sec"),
                "max_goal_count": cfg("max_goal_count"),
                "goal_tolerance": cfg("explorer_goal_tolerance"),
                "stall_detect_sec": cfg("stall_detect_sec"),
                "stall_move_eps": cfg("stall_move_eps"),
                "target_blacklist_sec": cfg("target_blacklist_sec"),
                "target_blacklist_radius": cfg("target_blacklist_radius"),
                "target_hysteresis_sec": cfg("target_hysteresis_sec"),
            }
        ],
    )
    actions.append(frontier_explorer)

    # ------------------------------------------------------------------ 5) 控制器
    use_ackermann = _as_bool(cfg("use_ackermann_interface"))
    # 差速/直发 Twist：follower 直接发到 cmd_vel_topic；
    # 阿克曼链路：follower 发到 twist_topic，再经 twist_to_ackermann 转成 ackermann_topic。
    follower_cmd_vel_topic = cfg("twist_topic") if use_ackermann else cfg("cmd_vel_topic")
    path_follower = Node(
        package="path_follower",
        executable="path_follower_node",
        name="path_follower",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "scan_topic": cfg("scan_topic"),
                "robot_frame": cfg("base_frame"),
                "global_frame": "map",
                "cmd_vel_topic": follower_cmd_vel_topic,
                "max_linear_vel": cfg("max_linear_vel"),
                "max_angular_vel": cfg("max_angular_vel"),
                "lookahead_dist": cfg("lookahead_dist"),
                "slowdown_dist": cfg("slowdown_dist"),
                "min_linear_vel": cfg("min_linear_vel"),
                "goal_tolerance": cfg("goal_tolerance"),
            }
        ],
    )
    actions.append(path_follower)

    # 阿克曼转换节点：仅当 use_ackermann_interface=true 时启动。
    twist_to_ackermann = Node(
        package="bringup",
        executable="twist_to_ackermann",
        name="twist_to_ackermann",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "twist_topic": cfg("twist_topic"),
                "ackermann_topic": cfg("ackermann_topic"),
                "wheelbase": cfg("wheelbase"),
                "max_steer": cfg("max_steer"),
                "frame_id": cfg("base_frame"),
            }
        ],
        condition=IfCondition(LaunchConfiguration("use_ackermann_interface")),
    )
    actions.append(twist_to_ackermann)

    # ------------------------------------------------------------------ 6) 可视化（可选）
    viz = Node(
        package="bringup",
        executable="exploration_viz",
        name="exploration_viz",
        output="screen",
        parameters=[{"use_sim_time": False, "refresh_hz": cfg("refresh_hz")}],
        condition=IfCondition(LaunchConfiguration("viz")),
    )
    actions.append(viz)

    return actions


def generate_launch_description():
    # ---- 默认值（真机路径必须 os.path.expanduser 展开 ~）----
    default_algorithm_root = (
        os.environ.get("REGIONAL_LEGACY_ROOT") or os.path.expanduser("~/regional_legacy")
    )
    default_map_save_prefix = os.path.expanduser("~/.ros/maps/exploration_map")

    declared_arguments = [
        # ---------- 驱动接入 ----------
        DeclareLaunchArgument(
            "start_driver", default_value="false",
            description="true 时 include 厂商驱动 launch（driver_package/driver_launch_file）",
        ),
        DeclareLaunchArgument(
            "driver_package", default_value="",
            description="厂商驱动所在 ROS2 包名（start_driver=true 时必填）",
        ),
        DeclareLaunchArgument(
            "driver_launch_file", default_value="",
            description="厂商驱动 launch 文件名（start_driver=true 时必填）",
        ),
        # ---------- 话题 ----------
        DeclareLaunchArgument("scan_topic", default_value="scan", description="雷达 LaserScan 话题"),
        DeclareLaunchArgument("imu_topic", default_value="imu", description="IMU 话题"),
        DeclareLaunchArgument("odom_topic", default_value="odom", description="底盘里程计话题"),
        DeclareLaunchArgument("map_topic", default_value="map", description="栅格地图话题"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="cmd_vel", description="底盘速度指令话题（差速）"),
        DeclareLaunchArgument("twist_topic", default_value="cmd_vel_raw", description="阿克曼链路里 follower 发出的 Twist 话题"),
        DeclareLaunchArgument("ackermann_topic", default_value="ackermann_cmd", description="阿克曼指令话题"),
        # ---------- 坐标系 ----------
        DeclareLaunchArgument("base_frame", default_value="base_link", description="机器人本体坐标系"),
        DeclareLaunchArgument("odom_frame", default_value="odom", description="里程计坐标系"),
        DeclareLaunchArgument("laser_frame", default_value="laser", description="雷达坐标系"),
        DeclareLaunchArgument("imu_frame", default_value="imu_link", description="IMU 坐标系"),
        # ---------- 静态外参（默认占位 0.0，必须按实测回填）----------
        DeclareLaunchArgument("laser_x", default_value="0.0", description="雷达相对 base_frame 的 x（m）"),
        DeclareLaunchArgument("laser_y", default_value="0.0", description="雷达相对 base_frame 的 y（m）"),
        DeclareLaunchArgument("laser_z", default_value="0.0", description="雷达相对 base_frame 的 z（m）"),
        DeclareLaunchArgument("laser_roll", default_value="0.0", description="雷达安装 roll（rad）"),
        DeclareLaunchArgument("laser_pitch", default_value="0.0", description="雷达安装 pitch（rad）"),
        DeclareLaunchArgument("laser_yaw", default_value="0.0", description="雷达安装 yaw（rad）"),
        DeclareLaunchArgument("imu_x", default_value="0.0", description="IMU 相对 base_frame 的 x（m）"),
        DeclareLaunchArgument("imu_y", default_value="0.0", description="IMU 相对 base_frame 的 y（m）"),
        DeclareLaunchArgument("imu_z", default_value="0.0", description="IMU 相对 base_frame 的 z（m）"),
        DeclareLaunchArgument("imu_roll", default_value="0.0", description="IMU 安装 roll（rad）"),
        DeclareLaunchArgument("imu_pitch", default_value="0.0", description="IMU 安装 pitch（rad）"),
        DeclareLaunchArgument("imu_yaw", default_value="0.0", description="IMU 安装 yaw（rad）"),
        DeclareLaunchArgument("publish_laser_tf", default_value="true", description="是否发布 base_frame->laser_frame 静态 TF"),
        DeclareLaunchArgument("publish_imu_tf", default_value="true", description="是否发布 base_frame->imu_frame 静态 TF"),
        DeclareLaunchArgument("publish_base_footprint_tf", default_value="false", description="是否发布 base_footprint->base_frame 静态 TF"),
        DeclareLaunchArgument("base_z", default_value="0.0", description="base_footprint->base_link 的 z 高度（m）"),
        # ---------- 位姿桥 ----------
        DeclareLaunchArgument(
            "publish_odom_tf", default_value="false",
            description="true 时启动 odom_tf_broadcaster（驱动不发 odom->base_link TF 时置 true）",
        ),
        # ---------- 控制接口 ----------
        DeclareLaunchArgument("use_ackermann_interface", default_value="false", description="true 时经 twist_to_ackermann 输出阿克曼指令"),
        DeclareLaunchArgument("wheelbase", default_value="0.16", description="轴距（m），阿克曼转换用"),
        DeclareLaunchArgument("max_steer", default_value="0.6", description="前轮最大转角（rad），阿克曼转换用"),
        # ---------- 建图 ----------
        DeclareLaunchArgument("mapping_config", default_value="cartographer_real.lua", description="cartographer lua 配置文件名（nav_slam/config 下）"),
        DeclareLaunchArgument(
            "occupancy_grid_executable", default_value="cartographer_occupancy_grid_node",
            description="occupancy grid 可执行名（Humble=cartographer_occupancy_grid_node，Foxy=occupancy_grid_node）",
        ),
        # ---------- 算法 / 控制 ----------
        DeclareLaunchArgument("algorithm_root", default_value=default_algorithm_root, description="regional_legacy 算法目录"),
        DeclareLaunchArgument("map_save_prefix", default_value=default_map_save_prefix, description="探索完成时的存图前缀（绝对路径）"),
        DeclareLaunchArgument("replan_rate_hz", default_value="1.0", description="探索器决策调度频率（Hz）"),
        DeclareLaunchArgument("obstacle_safe_distance", default_value="5.0", description="车体安全包络半径（格）"),
        DeclareLaunchArgument("inflation_radius_cells", default_value="5.0", description="A* 障碍硬膨胀半径（格）"),
        DeclareLaunchArgument("min_frontier_distance", default_value="10.0", description="前沿到机器人的最小可达距离（格）"),
        DeclareLaunchArgument("no_frontier_confirm_count", default_value="10", description="连续多少轮无目标才判定完成"),
        DeclareLaunchArgument("max_exploration_time_sec", default_value="1800.0", description="探索超时（s）"),
        DeclareLaunchArgument("max_goal_count", default_value="200", description="最大目标数"),
        DeclareLaunchArgument("goal_tolerance", default_value="0.25", description="path_follower 到达判定半径（m）"),
        DeclareLaunchArgument("explorer_goal_tolerance", default_value="0.35", description="frontier_explorer 到达判定半径（m）"),
        DeclareLaunchArgument("max_linear_vel", default_value="0.15", description="最大线速度（m/s）"),
        DeclareLaunchArgument("max_angular_vel", default_value="0.6", description="最大角速度（rad/s）"),
        DeclareLaunchArgument("lookahead_dist", default_value="0.4", description="前视点距离（m）"),
        DeclareLaunchArgument("slowdown_dist", default_value="0.6", description="减速段距离（m）"),
        DeclareLaunchArgument("min_linear_vel", default_value="0.06", description="最小线速度（m/s）"),
        DeclareLaunchArgument("stall_detect_sec", default_value="45.0", description="目标推进检测时长（s）"),
        DeclareLaunchArgument("stall_move_eps", default_value="0.30", description="目标推进位移阈值（m）"),
        DeclareLaunchArgument("target_blacklist_sec", default_value="180.0", description="临时黑名单时长（s）"),
        DeclareLaunchArgument("target_blacklist_radius", default_value="1.5", description="黑名单半径（m）"),
        DeclareLaunchArgument("target_hysteresis_sec", default_value="10.0", description="目标迟滞时长（s）"),
        # ---------- 可视化 ----------
        DeclareLaunchArgument("viz", default_value="false", description="是否启动 matplotlib 可视化节点"),
        DeclareLaunchArgument("refresh_hz", default_value="5.0", description="可视化重绘频率（Hz）"),
    ]

    return LaunchDescription(declared_arguments + [OpaqueFunction(function=_setup)])
