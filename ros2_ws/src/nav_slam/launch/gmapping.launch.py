import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    nav_slam_pkg = get_package_share_directory('nav_slam')
    config_file = os.path.join(nav_slam_pkg, 'config', 'cartographer.lua')

    # 无回波预处理（2026-10-22 决策）：
    #   Gazebo/真机在量程内无物体时发 inf，Cartographer 的 LaserScan 转换会丢弃 inf，
    #   missing_data_ray_length 不会生效。这里先把 inf/0/超量程改写成 sentinel=8.5m
    #   （> max_range=8.0），Cartographer 才会把它投影到 missing_data_ray_length=7.0m
    #   并写成 free 栅格。原始 /scan 保持不变，供 path_follower 避障使用。
    #   注意：missing_range 必须与 cartographer.lua 中的 missing_data_ray_length 一致。
    scan_filter = Node(
        package='nav_slam',
        executable='scan_no_return_filter',
        name='scan_no_return_filter',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/scan',
            'output_topic': '/scan_for_slam',
            'slam_max_range': 8.0,
            'missing_range': 7.0,
            'sentinel_range': 8.5,
        }],
    )

    return LaunchDescription([
        scan_filter,
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
            }],
            arguments=[
                '-configuration_directory', os.path.dirname(config_file),
                '-configuration_basename', 'cartographer.lua',
            ],
            remappings=[
                ('scan', '/scan_for_slam'),
                ('imu', '/imu'),
                ('odom', '/odom'),
            ],
        ),
        Node(
            package='cartographer_ros',
            executable='occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
            }],
        ),
    ])
