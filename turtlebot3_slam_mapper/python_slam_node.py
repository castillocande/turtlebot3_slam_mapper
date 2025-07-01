#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from scipy.spatial.transform import Rotation as R
import random
from copy import deepcopy


class Particle:
    """
    Represents a single particle in the particle filter for SLAM.

    Each particle holds a potential robot pose (x, y, theta), a weight
    representing its probability, and a local counter map and locked mask
    used for updating its associated occupancy grid.
    """
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.counter_map = np.zeros(map_shape, dtype=np.int32)
        self.locked_mask = np.zeros(map_shape, dtype=bool)

    def pose(self):
        """
        Returns the pose of the particle as a NumPy array.
        """
        return np.array([self.x, self.y, self.theta])


class PythonSlamNode(Node):
    """
    A ROS2 node for performing FastSLAM using a particle filter.

    This node subscribes to odometry and laser scan data, maintains a set
    of particles to estimate the robot's pose, and builds an occupancy grid map
    of the environment. It also publishes the estimated map and the transform
    between the map and odometry frames.
    """
    def __init__(self):
        super().__init__('python_slam_node')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_meters', 5.0)
        self.declare_parameter('map_height_meters', 5.0)
        self.declare_parameter('num_particles', 10)

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        self.occupancy_threshold = 10

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.best_particle = self.particles[0]

        self.last_odom = None

        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        """
        Callback to store the most recent odometry message.
        """
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        """
        Callback function for laser scan messages.
        This is the main loop for the particle filter. It updates particle poses
        based on odometry, computes weights based on scan data, resamples particles,
        updates the occupancy grid map for each particle, and broadcasts the
        map-to-odom transform.
        """
        if self.last_odom is None:
            return
        
        x = self.last_odom.pose.pose.position.x
        y = self.last_odom.pose.pose.position.y
        q = self.last_odom.pose.pose.orientation
        theta = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

        for p in self.particles:
            p.x = float(x + np.random.normal(loc=0, scale=0.0015))
            p.y = float(y + np.random.normal(loc=0, scale=0.0015))
            p.theta = float(theta + np.random.normal(loc=0, scale=0.005))

        weights = []
        s = 0
        for p in self.particles:
            w = self.compute_weight(p, msg)
            weights.append(w)
            s += w

        for i, p in enumerate(self.particles):
            p.weight = weights[i]/s

        self.particles = self.resample_particles(self.particles)
        self.best_particle = max(self.particles, key=lambda p: p.weight)

        for p in self.particles:
            self.update_map(p, msg)

        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        """
        Computes the weight of a particle based on how well its predicted scan
        matches the actual laser scan data against its internal map.
        """
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist):
                continue
            angle_scan = scan_msg.angle_min + i * scan_msg.angle_increment
            x_scan = range_dist * math.cos(angle_scan)
            y_scan = range_dist * math.sin(angle_scan)

            x_hit = robot_x + x_scan * math.cos(robot_theta) - y_scan * math.sin(robot_theta)
            y_hit = robot_y + x_scan * math.sin(robot_theta) + y_scan * math.cos(robot_theta)

            x_cell = int((x_hit - self.map_origin_x) / self.resolution)
            y_cell = int((y_hit - self.map_origin_y) / self.resolution)

            if 0 <= x_cell < self.map_width_cells and 0 <= y_cell < self.map_height_cells:
                if particle.counter_map[y_cell, x_cell] >= self.occupancy_threshold:
                    score += 1.0
                else:
                    score += 0.1

        return score + 1e-6

    def resample_particles(self, particles):
        """
        Resamples the particles using the low variance resampling algorithm.
        """
        N = len(particles)
        cdf_sum = 0
        p_cdf = []
        for p in particles:
            cdf_sum += p.weight
            p_cdf.append(cdf_sum)
        step = 1.0 / N
        seed = random.uniform(0, step)
        new_particles = []
        last_index = 0
        for _ in range(N):
            while seed > p_cdf[last_index]:
                last_index += 1
            new_particles.append(deepcopy(particles[last_index]))
            seed += step
        return new_particles

    def update_map(self, particle, scan_msg):
        """
        Updates the occupancy grid map associated with a given particle
        based on the latest laser scan data. This involves marking cells 
        along the laser beam as free and the hit cell as occupied.
        """
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            if math.isinf(range_dist) or math.isnan(range_dist) or range_dist < scan_msg.range_min:
                continue

            current_range = min(range_dist, scan_msg.range_max)
            angle_scan = scan_msg.angle_min + i * scan_msg.angle_increment
            x_scan = current_range * math.cos(angle_scan)
            y_scan = current_range * math.sin(angle_scan)

            x_hit = robot_x + x_scan * math.cos(robot_theta) - y_scan * math.sin(robot_theta)
            y_hit = robot_y + x_scan * math.sin(robot_theta) + y_scan * math.cos(robot_theta)

            x0 = int((robot_x - self.map_origin_x) / self.resolution)
            y0 = int((robot_y - self.map_origin_y) / self.resolution)
            x1 = int((x_hit - self.map_origin_x) / self.resolution)
            y1 = int((y_hit - self.map_origin_y) / self.resolution)

            self.bresenham_line(particle, x0, y0, x1, y1)

            if 0 <= x1 < self.map_width_cells and 0 <= y1 < self.map_height_cells:
                if not particle.locked_mask[y1, x1]:
                    particle.counter_map[y1, x1] += 1
                    if particle.counter_map[y1, x1] >= self.occupancy_threshold:
                        particle.locked_mask[y1, x1] = True

    def bresenham_line(self, particle, x0, y0, x1, y1):
        """
        Implements Bresenham's line algorithm to mark cells along a line as free.
        Used to update the 'free' cells in the particle's counter map based on
        the path of a laser beam.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.locked_mask[y0, x0] = False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        """
        Publishes the occupancy grid map based on the best particle's map.
        Cells with a counter value above a threshold are marked as occupied (100),
        and others as free (0). Unknown cells are -1.
        """
        map_msg = OccupancyGrid()
        log_occ = self.best_particle.counter_map >= self.occupancy_threshold
        log_free = (self.best_particle.counter_map < self.occupancy_threshold) & (~np.isnan(self.best_particle.counter_map))

        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.map_frame
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.orientation.w = 1.0

        grid = np.full(self.best_particle.counter_map.shape, -1, dtype=np.int8)
        grid[log_free] = 0
        grid[log_occ] = 100
        map_msg.data = grid.flatten().tolist()
        self.map_publisher.publish(map_msg)

    def broadcast_map_to_odom(self):
        """
        Broadcasts the transform from the map frame to the odometry frame.
        This transform aligns the odometry frame (which drifts over time)
        with the globally consistent map frame, based on the best particle's pose.
        """
        if self.last_odom is None:
            return
        x_odom = self.last_odom.pose.pose.position.x
        y_odom = self.last_odom.pose.pose.position.y
        q = self.last_odom.pose.pose.orientation
        theta_odom = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

        dx = self.best_particle.x - x_odom
        dy = self.best_particle.y - y_odom
        dtheta = self.angle_diff(self.best_particle.theta, theta_odom)
        rot = R.from_euler('z', -theta_odom)
        trans = rot.apply([dx, dy, 0.0])

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = 0.0
        q_rot = R.from_euler('z', dtheta).as_quat()
        t.transform.rotation.x = float(q_rot[0])
        t.transform.rotation.y = float(q_rot[1])
        t.transform.rotation.z = float(q_rot[2])
        t.transform.rotation.w = float(q_rot[3])
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def angle_diff(a, b):
        """
        Calculates the shortest angular difference between two angles.
        """
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d


def main(args=None):
    rclpy.init(args=args)
    node = PythonSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()