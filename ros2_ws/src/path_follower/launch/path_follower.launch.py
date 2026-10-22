from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """启动路径跟随节点，使用仿真时间。

    新增的安全参数（前向脱困状态机 + 卡死检测/倒车自救）全部通过 DeclareLaunchArgument
    暴露，可用命令行 ``名字:=值`` 覆盖；默认值与节点自身默认值一致。
    """
    declared_arguments = [
        DeclareLaunchArgument(
            "stale_timeout_sec",
            default_value="15.0",
            description="上游 path/goal 失效阈值（s），需大于上游单轮决策耗时",
        ),
        DeclareLaunchArgument(
            "max_angular_vel",
            default_value="0.6",
            description="最大角速度（rad/s）；0.6 已收进真机可达范围，再配合曲率钳制双重保护",
        ),
        DeclareLaunchArgument(
            "traj_v_max",
            default_value="0.30",
            description="曲率限速上限（m/s），与上游阿克曼轨迹 traj_v_max 一致",
        ),
        DeclareLaunchArgument(
            "traj_v_min",
            default_value="0.08",
            description="曲率限速下限（m/s），保证弯道仍有转向能力",
        ),
        DeclareLaunchArgument(
            "a_lat_max",
            default_value="0.60",
            description="横向加速度上限（m/s²），v_curv=sqrt(a_lat_max/κ)",
        ),
        DeclareLaunchArgument(
            "min_turn_radius_m",
            default_value="0.35",
            description=(
                "可达曲率钳制的最小转弯半径（m），kappa_max=1/R_min；"
                "必须与上游阿克曼轨迹的 min_turn_radius_m 一致。<=0 回退到 tan(max_steer)/L"
            ),
        ),
        DeclareLaunchArgument(
            "wheelbase_m",
            default_value="0.144",
            description="真机轴距（m）；仅用于 min_turn_radius_m<=0 的旧行为回退",
        ),
        DeclareLaunchArgument(
            "max_steer_rad",
            default_value="0.489",
            description="真机前轮最大转角（rad，约 28°）；仅用于 min_turn_radius_m<=0 的回退",
        ),
        DeclareLaunchArgument(
            "curvature_guard_enable",
            default_value="True",
            description="true 时把角速度钳到当前线速度物理可达的曲率内，避免满舵绕圈",
        ),
        DeclareLaunchArgument(
            "lookahead_adaptive",
            default_value="True",
            description="true 时前视距离按到目标距离自适应（远处大、近处小）",
        ),
        DeclareLaunchArgument(
            "lookahead_min",
            default_value="0.4",
            description="自适应前视距离下限（m）",
        ),
        DeclareLaunchArgument(
            "lookahead_max",
            default_value="1.0",
            description="自适应前视距离上限（m）；实际还受'到路径末点距离'限制",
        ),
        DeclareLaunchArgument(
            "scan_topic",
            default_value="scan",
            description="前向脱困状态机所用的激光雷达话题",
        ),
        DeclareLaunchArgument(
            "enable_safety_brake",
            default_value="True",
            description="true 时启用前向脱困状态机（正前方过近则倒车+转向脱困）",
        ),
        DeclareLaunchArgument(
            "brake_distance",
            default_value="0.30",
            description="脱困触发距离（m），正前方小于该值即进入脱困",
        ),
        DeclareLaunchArgument(
            "brake_clear_distance",
            default_value="0.55",
            description="脱困退出距离（m），正前方大于该值才算脱离（与触发距离构成回差）",
        ),
        DeclareLaunchArgument(
            "brake_half_angle_deg",
            default_value="35.0",
            description="前向扇区半角（度）",
        ),
        DeclareLaunchArgument(
            "escape_speed",
            default_value="0.15",
            description="脱困倒车速度（m/s）",
        ),
        DeclareLaunchArgument(
            "escape_forward_speed",
            default_value="0.10",
            description="三点掉头前进相速度（m/s）",
        ),
        DeclareLaunchArgument(
            "shuttle_sec",
            default_value="0.25",
            description="三点掉头单个相位（倒车/前进）持续时间（s）。取 0.25：转向 P=2.0 后阶跃反向到 90% 约 0.095s，2×0.095≈0.19s，留余量到 0.25s",
        ),
        DeclareLaunchArgument(
            "escape_angular",
            default_value="0.35",
            description="脱困转向角速度（rad/s）；取 0.35 使反解转角 steer≈0.44 不超 max_steer=0.6",
        ),
        DeclareLaunchArgument(
            "escape_timeout_sec",
            default_value="10.0",
            description="单次脱困最长时间（s）",
        ),
        DeclareLaunchArgument(
            "scan_timeout_sec",
            default_value="1.0",
            description="/scan 失效阈值（s）：超过该秒数未收到新帧即视为没有 scan 数据、不脱困",
        ),
        DeclareLaunchArgument(
            "stuck_detect_sec",
            default_value="3.0",
            description="卡死检测时间窗口（s）",
        ),
        DeclareLaunchArgument(
            "stuck_move_eps",
            default_value="0.05",
            description="卡死位移阈值（m），窗口内位移小于该值即视为没动",
        ),
        DeclareLaunchArgument(
            "reverse_speed",
            default_value="0.12",
            description="倒车自救速度（m/s）",
        ),
        DeclareLaunchArgument(
            "reverse_sec",
            default_value="1.0",
            description="倒车自救持续时间（s）",
        ),
        DeclareLaunchArgument(
            "reverse_half_angle_deg",
            default_value="30.0",
            description="正后方锥形半角（度）：只统计 |angle|>180°-该值 的扇区",
        ),
        DeclareLaunchArgument(
            "reverse_brake_distance",
            default_value="0.25",
            description="倒车后向安全距离（m）：正后方锥形最小测距小于该值则取消本次倒车",
        ),
        DeclareLaunchArgument(
            "reverse_angular",
            default_value="0.4",
            description="倒车时的角速度（rad/s），让倒车顺便改变朝向",
        ),
        DeclareLaunchArgument(
            "enable_stuck_recovery",
            default_value="True",
            description="true 时启用卡死检测 + 倒车自救",
        ),
        DeclareLaunchArgument(
            "stuck_recover_cooldown_sec",
            default_value="5.0",
            description="两次倒车自救之间的最小间隔（s）",
        ),
    ]

    path_follower = Node(
        package="path_follower",
        executable="path_follower_node",
        name="path_follower",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "stale_timeout_sec": LaunchConfiguration("stale_timeout_sec"),
                "max_angular_vel": LaunchConfiguration("max_angular_vel"),
                "traj_v_max": LaunchConfiguration("traj_v_max"),
                "traj_v_min": LaunchConfiguration("traj_v_min"),
                "a_lat_max": LaunchConfiguration("a_lat_max"),
                "min_turn_radius_m": LaunchConfiguration("min_turn_radius_m"),
                "wheelbase_m": LaunchConfiguration("wheelbase_m"),
                "max_steer_rad": LaunchConfiguration("max_steer_rad"),
                "curvature_guard_enable": LaunchConfiguration("curvature_guard_enable"),
                "lookahead_adaptive": LaunchConfiguration("lookahead_adaptive"),
                "lookahead_min": LaunchConfiguration("lookahead_min"),
                "lookahead_max": LaunchConfiguration("lookahead_max"),
                "scan_topic": LaunchConfiguration("scan_topic"),
                "enable_safety_brake": LaunchConfiguration("enable_safety_brake"),
                "brake_distance": LaunchConfiguration("brake_distance"),
                "brake_clear_distance": LaunchConfiguration("brake_clear_distance"),
                "brake_half_angle_deg": LaunchConfiguration("brake_half_angle_deg"),
                "escape_speed": LaunchConfiguration("escape_speed"),
                "escape_forward_speed": LaunchConfiguration("escape_forward_speed"),
                "shuttle_sec": LaunchConfiguration("shuttle_sec"),
                "escape_angular": LaunchConfiguration("escape_angular"),
                "escape_timeout_sec": LaunchConfiguration("escape_timeout_sec"),
                "scan_timeout_sec": LaunchConfiguration("scan_timeout_sec"),
                "stuck_detect_sec": LaunchConfiguration("stuck_detect_sec"),
                "stuck_move_eps": LaunchConfiguration("stuck_move_eps"),
                "reverse_speed": LaunchConfiguration("reverse_speed"),
                "reverse_sec": LaunchConfiguration("reverse_sec"),
                "reverse_half_angle_deg": LaunchConfiguration("reverse_half_angle_deg"),
                "reverse_brake_distance": LaunchConfiguration("reverse_brake_distance"),
                "reverse_angular": LaunchConfiguration("reverse_angular"),
                "enable_stuck_recovery": LaunchConfiguration("enable_stuck_recovery"),
                "stuck_recover_cooldown_sec": LaunchConfiguration("stuck_recover_cooldown_sec"),
            }
        ],
    )

    return LaunchDescription(declared_arguments + [path_follower])
