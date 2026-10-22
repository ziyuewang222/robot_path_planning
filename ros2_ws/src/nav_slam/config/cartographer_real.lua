include "map_builder.lua"
include "trajectory_builder.lua"

-- ============================================================================
-- 真机（RDK-X5 小车，ROS2 Humble）Cartographer 2D 配置 —— **有 IMU** 版本
-- ----------------------------------------------------------------------------
-- 与仿真用 cartographer.lua 的差异（真机基准见 AGENTS.md 附十 / 附十四）：
--   * TRAJECTORY_BUILDER_2D.min_range = 0.15、max_range = 8.0、
--     missing_data_ray_length = 8.0（真机雷达 LD14P 有效量程 0.15~8.0 m；
--     仿真里是 0.12 / 12.0）。
--   * use_imu_data = true，tracking_frame = "imu_link"：
--     Cartographer 强制要求 IMU 与 tracking frame 共位（平移 < 1e-5），否则启动即
--     `Check failed: The IMU frame must be colocated with the tracking frame` 崩溃。
--     本文件假设真机 IMU 就装在 imu_link 上、且 imu_link 与 base_link 分离。
--   * published_frame = "odom"、provide_odom_frame = false、
--     publish_frame_projected_to_2d = true（与仿真一致的 TF 约定：map->odom 由
--     Cartographer 唯一权威发布，避免 TF 双权威）。
--   * use_odometry = false：真机里程计可信度待实测。
--
-- 三种情形的处置（请按真机实测选择）：
--   1) 真机**没有 IMU**，或 IMU 与 imu_link 不共位 —— 改用
--      cartographer_real_noimu.lua（use_imu_data = false、tracking_frame = "base_link"）。
--   2) 实测里程计**可靠**（直线误差 < 5% 且与 TF 同步）—— 可复制本文件为真机专用副本，
--      把 use_odometry 改为 true 后使用，以抑制长直走廊的平移漂移。
--   3) IMU 与 base_link 共位（origin 平移为 0）—— 可把 tracking_frame 改回 base_link。
--
-- 本文件保留 Foxy 兼容写法（末尾 return options，且不含 Foxy 不支持的键），
-- 以便在本机 Foxy 上也能被 cartographer_node 加载自检；真机 Humble 同样适用。
-- ============================================================================

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  -- Cartographer 要求 IMU 与 tracking frame 必须共位（平移 < 1e-5）。
  -- 本文件假定真机 IMU 就装在 imu_link 上，故取 tracking_frame = "imu_link"。
  tracking_frame = "imu_link",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
  -- 2D 场景标准做法：把发布的位姿投影到 2D 平面，避免 map->odom 携带 roll/pitch。
  publish_frame_projected_to_2d = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-2,
  trajectory_publish_period_sec = 30e-2,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- 真机雷达 LD14P：有效量程 0.15 ~ 8.0 m（仿真为 0.12 ~ 12.0）。
TRAJECTORY_BUILDER_2D.min_range = 0.15
TRAJECTORY_BUILDER_2D.max_range = 8.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 8.0
-- 打开 IMU：IMU 主要提供 yaw 约束与重力对齐，抑制纯激光匹配 + 常速外推带来的位姿漂移。
-- 若实测 yaw 漂移较大，可调整下面的 imu_gravity_time_constant（值越大越"信任"重力方向、
-- 越平滑，默认 10）。
TRAJECTORY_BUILDER_2D.use_imu_data = true
TRAJECTORY_BUILDER_2D.imu_gravity_time_constant = 10.
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 5.
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.2
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = 0.004

-- 以下 pose graph 参数沿用现有仿真配置（真机基准未要求更改）。
POSE_GRAPH.optimization_problem.huber_scale = 1e2
POSE_GRAPH.optimize_every_n_nodes = 90
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 10
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.6
POSE_GRAPH.constraint_builder.loop_closure_translation_weight = 1.1e4
POSE_GRAPH.constraint_builder.loop_closure_rotation_weight = 1e5
POSE_GRAPH.max_num_final_iterations = 200
POSE_GRAPH.global_sampling_ratio = 0.003
POSE_GRAPH.log_residual_histograms = true

return options
