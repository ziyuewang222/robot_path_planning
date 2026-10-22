import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    nav_slam_pkg = get_package_share_directory('nav_slam')
    config_file = os.path.join(nav_slam_pkg, 'config', 'cartographer.lua')
    
    return LaunchDescription([
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
                ('scan', '/scan'),
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