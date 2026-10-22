
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 <Ming2zun:https://github.com/Ming2zun/Pure-tracking-slam-automatic-navigation-system>
#                <喵了个水蓝蓝:https://www.bilibili.com/video/BV1kzEwzuEFw?spm_id_from=333.788.videopod.sections&vd_source=134c12873ff478ea447a06d652426f8f>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import GroupAction
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare




def generate_launch_description():
    robot_name_in_model = 'fishbot'
    package_name = 'gazebo_modele' 

    urdf_name = "model.urdf" 
   
    ld = LaunchDescription()
    pkg_share = FindPackageShare(package=package_name).find(package_name) 
    urdf_model_path = os.path.join(pkg_share, f'urdf/{urdf_name}')
    
    # 世界文件可切换：默认封闭矩形房间 room.world（适合"从零跑通完整建图"）；
    # 传 world:=3d.world 可切回原来的复杂建筑场景。
    gazebo_world_path = PathJoinSubstitution(
        [pkg_share, 'world', LaunchConfiguration('world')]
    )

    # world=... 选择要加载的世界文件名（world/ 目录下的文件名）。
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="room.world",
        description="要加载的世界文件名（gazebo_modele/world/ 下），默认封闭矩形房间 room.world",
    )

    # headless=true 时不启动 gzclient（无显示 / GLX 不可用的环境）。
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="true 时只起 gzserver、不起 gzclient（无显示环境用）",
    )
    # software_gl=true 时对 Gazebo 进程设 LIBGL_ALWAYS_SOFTWARE=1（走软件 OpenGL，
    # 绕开显卡/GLX 驱动问题；可救回 gzclient 建 GL/GLX 失败的情形）。
    software_gl_arg = DeclareLaunchArgument(
        "software_gl",
        default_value="false",
        description="true 时对 Gazebo 进程设 LIBGL_ALWAYS_SOFTWARE=1（走软件 OpenGL，绕开显卡/GLX 驱动问题）",
    )

    # headless 时清空 DISPLAY，避免 gzserver 仍去创建 X/GLX 上下文
    #（本环境沙箱会拦截 /dev/nvidiactl，导致 GLX 失败、世界加载前卡死）。
    headless_display_env = GroupAction(
        actions=[SetEnvironmentVariable("DISPLAY", "")],
        condition=IfCondition(LaunchConfiguration("headless")),
        scoped=False,
    )

    # software_gl 时对 Gazebo 进程设软件渲染。须放在 gzserver / gzclient 两个进程动作之前生效。
    software_gl_env = GroupAction(
        actions=[SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1")],
        condition=IfCondition(LaunchConfiguration("software_gl")),
        scoped=False,
    )

    # gzserver 无条件启动：仿真与建图的核心，绝不随 headless 变化。
    # 注意：不再用 `gazebo` 包装器 —— 包装器会同时拉起 gzserver+gzclient，
    # gzclient 建 GL/GLX 失败会让包装器整体退出 255，并把 gzserver 一起带走。
    start_gazebo_cmd = ExecuteProcess(
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', gazebo_world_path],
        output='screen')

    # gzclient 单独作为独立进程：它崩掉不会影响 gzserver，仿真与建图继续跑。
    start_gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration("headless")),
    )
        
    # Launch the robot
    spawn_entity_cmd = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=['-entity', robot_name_in_model,  '-file', urdf_model_path ], output='screen')
	
    # Start Robot State publisher
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        arguments=[urdf_model_path],
        parameters=[{'use_sim_time': True}]
    )
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        arguments=[urdf_model_path],
        parameters=[{'use_sim_time': True}],
        condition=UnlessCondition(LaunchConfiguration("headless"))
        )
    # 注意：odom->base_footprint 现在由 ackermann 插件动态发布（URDF 里
    # publish_odom_tf=true），所以这里**不再**放"静态恒等 odom->base_footprint"，
    # 否则同一条 TF 有两个发布者，Cartographer 会把里程计先验和静态恒等混在一起。
    fake_basel_cmd5 = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=['0', '0', '0.072', '0', '0', '0', 'base_footprint', 'base_link'])
    
    
    

    ld.add_action(world_arg)
    ld.add_action(headless_arg)
    ld.add_action(software_gl_arg)
    # 环境变量条件动作须放在 Gazebo 两个进程动作之前。
    ld.add_action(software_gl_env)
    ld.add_action(headless_display_env)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(start_gzclient_cmd)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(joint_state_publisher_node)

   
    ld.add_action(fake_basel_cmd5)



    return ld