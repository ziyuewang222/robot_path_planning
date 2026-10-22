include "map_builder.lua"
include "trajectory_builder.lua"

-- ============================================================================
-- 真机（RDK-X5 小车，ROS2 Humble）Cartographer 2D 配置 —— **无 IMU** 版本
-- ----------------------------------------------------------------------------
-- 与 cartographer_real.lua（有 IMU 版本）唯一的差异：
--   * use_imu_data = false（不使用 IMU 数据）；
--   * tracking_frame = "base_link"（无 IMU 时不要求 IMU 共位约束）。
-- 其余雷达量程、TF 约定、pose graph 参数与 cartographer_real.lua 完全一致。
--
-- 适用场景：真机根本没装 IMU，或 IMU 不与 imu_link 共位（无法满足 Cartographer
-- "IMU 必须与 tracking frame 共位" 的硬性要求）。
--
-- 保留 imu_gravity_time_constant 是为了复用同一份骨架（该键在 Cartographer 里恒被读取
-- 一次，不会触发 Foxy 的"键使用次数"校验）；use_imu_data=false 时它不生效。
--
-- 本文件保留 Foxy 兼容写法（末尾 return options，且不含 Foxy 不支持的键）。
-- ============================================================================

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  -- 无 IMU：tracking frame 取 base_link（无需与 IMU 共位）。
  tracking_frame = "base_link",
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
-- 无 IMU：关闭 IMU 数据。
TRAJECTORY_BUILDER_2D.use_imu_data = false
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
