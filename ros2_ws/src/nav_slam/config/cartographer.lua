include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  -- Cartographer 要求 IMU 与 tracking frame 必须共位（平移 < 1e-5），否则会触发
  -- "The IMU frame must be colocated with the tracking frame" 的 CHECK 并崩溃退出。
  -- 本车 IMU 物理上装在 base_link 上方 0.10m（URDF imu_joint 的 origin 为 0 0 0.10），
  -- 与 base_link 不共位，故把 tracking frame 取为 imu_link（二者平移为 0，满足共位要求）。
  -- 若以后把 IMU 改成与 base_link 共位（origin 平移为 0），可改回 base_link。
  tracking_frame = "imu_link",
  published_frame = "odom",
  odom_frame = "odom",
  provide_odom_frame = false,
  -- 2D 场景标准做法：把发布的位姿投影到 2D 平面，避免 map->odom 携带 roll/pitch。
  publish_frame_projected_to_2d = true,
  -- 启用轮式里程计作为运动先验（真机同样有 /odom；仿真由 ackermann 插件发布）。
  -- 原因：纯激光匹配在"长直墙 + 开阔空地"下，沿墙方向的平移在几何上不可观测，
  -- 实测跑到后段出现 17.13m 纯平移滑移（yaw 误差仅 1.4°），地图膨胀、柱子重影。
  use_odometry = true,
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
  -- 保持 1.：IMU 全量参与优化（真机 IMU 已对齐 20Hz，无需降采样）
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- 以下三点对齐真机 LD14 0.15~8.0m：
-- min_range 取 0.15（过滤雷达近距盲区/自车遮挡），
-- max_range 取 8.0（真机物理有效量程，超出的读数不参与匹配），
-- missing_data_ray_length 取 4.0（小于 max_range 的一半，符合 Cartographer 官方建议，
-- 用于把"该有回波却无回波"的方向标记为自由空间）。
TRAJECTORY_BUILDER_2D.min_range = 0.15
TRAJECTORY_BUILDER_2D.max_range = 8.
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 4.
-- 打开 IMU：IMU 主要提供 yaw 约束与重力对齐，抑制纯激光匹配 + 常速外推带来的位姿漂移。
-- 若实测 yaw 漂移较大，可调整下面的 imu_gravity_time_constant（值越大越"信任"重力方向、越平滑，默认 10）。
TRAJECTORY_BUILDER_2D.use_imu_data = true
TRAJECTORY_BUILDER_2D.imu_gravity_time_constant = 10.
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 5.
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.2
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = 0.004

POSE_GRAPH.optimization_problem.huber_scale = 1e2
POSE_GRAPH.optimize_every_n_nodes = 90
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 10

-- ==================== 回环约束收紧（治"米级平移跳变"） ====================
-- 现象：sim 1204->1208 位姿从 0.063m 突跳到 1.629m（yaw 只差 1.43°），1306 到 5.901m，
--       1458 又"回正"到 0.96m；同期 /odom 对真值 RMSE 仅 0.0076m。
-- 诊断：局部匹配的 real_time_correlative_scan_matcher.linear_search_window 只有 0.1m，
--       不可能一次跳 1.6m；能产生米级突变的只有**位姿图的回环约束**。而
--       constraint_builder.fast_correlative_scan_matcher.linear_search_window 默认是 7.（±7m），
--       min_score 仅 0.55 —— 在 28x20 的长直墙对称房间里，"沿墙平移 1.6~7m"也能匹配成功，
--       于是错误约束把整段轨迹拉歪，之后又被别的约束拉回来（来回跳）。
-- 对策：把回环搜索窗口收到 2m、阈值提到 0.75，并加大里程计在优化中的权重。
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.linear_search_window = 2.
POSE_GRAPH.constraint_builder.fast_correlative_scan_matcher.angular_search_window = math.rad(15.)
POSE_GRAPH.constraint_builder.min_score = 0.75
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.8
POSE_GRAPH.constraint_builder.max_constraint_distance = 8.
POSE_GRAPH.constraint_builder.loop_closure_translation_weight = 1.1e4
POSE_GRAPH.constraint_builder.loop_closure_rotation_weight = 1e5

-- 提高里程计(轮式)在全局优化中的权重：本链路 /odom 实测对真值 RMSE 0.0076m、yaw 0.613°，
-- 是比"弱激光回环"更可信的约束。默认两者都是 1e5，这里各提高一个量级。
POSE_GRAPH.optimization_problem.odometry_translation_weight = 1e6
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 1e6

-- 每个子图容纳更多帧（默认 90）：子图内特征更多、更"独特"，降低对称环境下的误匹配。
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 180

POSE_GRAPH.max_num_final_iterations = 200
POSE_GRAPH.global_sampling_ratio = 0.003
POSE_GRAPH.log_residual_histograms = true

return options