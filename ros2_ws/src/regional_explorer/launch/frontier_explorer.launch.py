import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 默认指向本仓库 ros2_ws 同级的 regional_legacy 目录，可用 REGIONAL_LEGACY_ROOT 覆盖。
    default_algorithm_root = os.environ.get(
        "REGIONAL_LEGACY_ROOT",
        os.path.abspath(
            os.path.join(
                get_package_share_directory("regional_explorer"),
                "..",
                "..",
                "..",
                "..",
                "..",
                "regional_legacy",
            )
        ),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("algorithm_root", default_value=default_algorithm_root),
            DeclareLaunchArgument("map_topic", default_value="map"),
            DeclareLaunchArgument("robot_frame", default_value="base_link"),
            DeclareLaunchArgument("replan_rate_hz", default_value="1.0"),
            DeclareLaunchArgument("bfs_weight", default_value="8.0"),
            DeclareLaunchArgument("turn_weight", default_value="0.35"),
            DeclareLaunchArgument("obstacle_weight", default_value="1.0"),
            DeclareLaunchArgument("obstacle_safe_distance", default_value="5.0"),
            DeclareLaunchArgument("min_frontier_distance", default_value="10.0"),
            DeclareLaunchArgument("inflation_radius_cells", default_value="5.0"),
            DeclareLaunchArgument("target_hysteresis_sec", default_value="10.0"),
            DeclareLaunchArgument("target_keep_verify", default_value="True"),
            DeclareLaunchArgument("goal_tolerance", default_value="0.35"),
            DeclareLaunchArgument("max_exploration_time_sec", default_value="1800.0"),
            DeclareLaunchArgument("stall_detect_sec", default_value="45.0"),
            DeclareLaunchArgument("stall_move_eps", default_value="0.30"),
            DeclareLaunchArgument("target_blacklist_sec", default_value="180.0"),
            DeclareLaunchArgument("target_blacklist_radius", default_value="1.5"),
            # 同一目标连续多少次无法生成阿克曼轨迹就拉黑换目标（修复 B）。
            DeclareLaunchArgument("traj_fail_blacklist_count", default_value="2"),
            # 建筑外壳边界 "xmin,xmax,ymin,ymax"（米，世界坐标）；空字符串表示不启用（修复 C）。
            DeclareLaunchArgument("exploration_bounds", default_value=""),
            # ---- 阿克曼轨迹生成 ----
            DeclareLaunchArgument(
                "enable_ackermann_traj",
                default_value="True",
                description="True 时把 A* 折线转成阿克曼可行轨迹再发布；False 退回原始折线",
            ),
            DeclareLaunchArgument("wheelbase_m", default_value="0.144"),
            DeclareLaunchArgument("max_steer_rad", default_value="0.489"),
            DeclareLaunchArgument("min_turn_radius_m", default_value="0.35"),
            DeclareLaunchArgument("traj_ds_m", default_value="0.05"),
            DeclareLaunchArgument("traj_v_max", default_value="0.30"),
            DeclareLaunchArgument("traj_v_min", default_value="0.08"),
            DeclareLaunchArgument("traj_a_lat_max", default_value="0.60"),
            DeclareLaunchArgument("traj_a_long_max", default_value="0.25"),
            # ---- A* 搜索策略（部署侧默认打开）----
            DeclareLaunchArgument("use_eight_neighbors", default_value="True"),
            DeclareLaunchArgument("treat_unknown_as_obstacle", default_value="True"),
            # ---- 路径受阻检查 ----
            DeclareLaunchArgument("path_block_check_m", default_value="1.0"),
            Node(
                package="regional_explorer",
                executable="frontier_explorer_node",
                name="frontier_explorer",
                output="screen",
                parameters=[
                    {
                        "algorithm_root": LaunchConfiguration("algorithm_root"),
                        "map_topic": LaunchConfiguration("map_topic"),
                        "robot_frame": LaunchConfiguration("robot_frame"),
                        "replan_rate_hz": LaunchConfiguration("replan_rate_hz"),
                        "bfs_weight": LaunchConfiguration("bfs_weight"),
                        "turn_weight": LaunchConfiguration("turn_weight"),
                        "obstacle_weight": LaunchConfiguration("obstacle_weight"),
                        "obstacle_safe_distance": LaunchConfiguration("obstacle_safe_distance"),
                        "min_frontier_distance": LaunchConfiguration("min_frontier_distance"),
                        "inflation_radius_cells": LaunchConfiguration("inflation_radius_cells"),
                        "target_hysteresis_sec": LaunchConfiguration("target_hysteresis_sec"),
                        "target_keep_verify": LaunchConfiguration("target_keep_verify"),
                        "goal_tolerance": LaunchConfiguration("goal_tolerance"),
                        "max_exploration_time_sec": LaunchConfiguration(
                            "max_exploration_time_sec"
                        ),
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
                        "enable_ackermann_traj": LaunchConfiguration(
                            "enable_ackermann_traj"
                        ),
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
            ),
        ]
    )
