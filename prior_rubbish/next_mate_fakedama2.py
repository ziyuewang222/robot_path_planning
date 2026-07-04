#本文件展示轻量化路径后给下游的数据结构

# from nav_msgs.msg import Path
# from geometry_msgs.msg import PoseStampe

# # 1. 创建一列空火车
# global_path_msg = Path()
# global_path_msg.header.frame_id = "map"
# global_path_msg.header.stamp = self.get_clock().now().to_msg() # ROS2 的写法

# # 2. 遍历 A* 算法生成的原始点集，装进车厢
# # 算法的输入是目标点，输出是二元数组列表：
# raw_path = [(10, 15), (11, 15), (12, 16), (13, 17), ..., (45, 50)]
# for point in raw_path:
#     pose = PoseStamped()
#     # 注意：这里需要把栅格坐标转换为真实物理世界坐标（米）
#     pose.pose.position.x = point[0] * resolution + origin_x 
#     pose.pose.position.y = point[1] * resolution + origin_y
#     pose.pose.position.z = 0.0
    
#     # 路径点通常不需要朝向，默认即可
#     pose.pose.orientation.w = 1.0 
    
#     # 把这个坐标点挂到 Path 消息里
#     global_path_msg.poses.append(pose)

# # 3. 发布给下游
# path_publisher.publish(global_path_msg)