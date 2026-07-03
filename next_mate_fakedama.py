#此文件展示生成目标点的数据结构，以及逻辑

#from geometry_msgs.msg import PoseStamped

# 1. 实例化消息对象
#goal_msg = PoseStamped()

# 2. 填充坐标系和时间戳 (以 ROS 2 为例)
#goal_msg.header.frame_id = "map"
#goal_msg.header.stamp = self.get_clock().now().to_msg() 

# 3. 填充目标点坐标 (x, y)
#goal_msg.pose.position.x = 3.5  # 策略层
#goal_msg.pose.position.y = 4.2  
#goal_msg.pose.position.z = 0.0

# 4. 填充默认朝向 (重要，不填可能会报错)
#goal_msg.pose.orientation.x = 0.0
#goal_msg.pose.orientation.y = 0.0
#goal_msg.pose.orientation.z = 0.0
#goal_msg.pose.orientation.w = 1.0

# 5. 发布给下游
#publisher.publish(goal_msg)