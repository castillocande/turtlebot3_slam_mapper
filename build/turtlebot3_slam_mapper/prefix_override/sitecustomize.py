import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cande/turtlebot3_ws/src/turtlebot3_slam_mapper/install/turtlebot3_slam_mapper'
