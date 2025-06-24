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
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)

    def pose(self):
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node): # hereda de Node
    def __init__(self):
        super().__init__('python_slam_node') # llama al constructor de Node con el nombre como argumento

        # Parameters
        self.declare_parameter('odom_topic', '/odom') # declare/get_parameter son métodos de Node
        self.declare_parameter('scan_topic', '/scan') # los valores en el yaml con el mismo nombre de parámetro, los sobreescriben
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        # TODO: define map resolution, width, height, and number of particles

        #10 a 20 particulas
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_meters', 5.0)
        self.declare_parameter('map_height_meters', 5.0)
        self.declare_parameter('num_particles', 10)
        

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value #esta en config/slam_toolbox_params.yaml --> modificamos el config?
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        # TODO: define the log-odds criteria for free and occupied cells 
        prob_occ = 0.7
        prob_free = 1 - prob_occ
        self.log_odds_occ = np.log(prob_occ/prob_free)
        self.log_odds_free = np.log(prob_free/prob_occ) # sumamos logs para no multiplicar probabilidades

        self.log_odds_max = 5.0
        self.log_odds_min = -5.0

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        # inicializa las partículas en posición 0,0 con ángulo 0 y con el mismo peso, 1/N
        # cada partícula es un belief del robot; cada partícula tiene su mapa (mapa log-odds)
        self.best_map = np.zeros((self.map_height_cells, self.map_width_cells), dtype=np.float32) # el nodo se va a ir guardando el mejor mapa
        self.last_odom = None

        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile( # políticas para el manejo de mensajes entre nodos en la red
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile) # el que quiere escuchar el mapa, se puede suscribir a /map
        self.tf_broadcaster = TransformBroadcaster(self) # para publicar la transformación del map frame al odom frame; publica 
        # a critical problem is that the robot's local estimate of its position 
        # (from odometry, often called the odom frame) tends to drift over time 
        # due to accumulating errors. The map, however, is built in a globally 
        # consistent, drift-free coordinate system (the map frame).

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
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            return

        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y

        # Extract quaternion (w, x, y, z) format
        q_w = odom.pose.pose.orientation.w
        q_x = odom.pose.pose.orientation.x
        q_y = odom.pose.pose.orientation.y
        q_z = odom.pose.pose.orientation.z

        # Convert quaternion to rotation object
        current_rotation = R.from_quat([q_x, q_y, q_z, q_w])
        theta = current_rotation.as_euler('xyz', degrees=False)[2] 

        # TODO: Model the particles around the current pose
        for p in self.particles:
            # Add noise to simulate motion uncertainty
            dx = x - p.x
            dy = y - p.y
            dtheta = theta - p.theta

            p.x = x + np.random.normal(loc=dx, scale=0.1, size=1)
            p.y = y + np.random.normal(loc=dy, scale=0.1, size=1)
            p.theta = theta + np.random.normal(loc=dtheta, scale=0.1, size=1)



        # TODO: 2. Measurement update (weight particles)
        weights = []
        s = 0
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            weights.append(weight)
            s += weight
            # Save, append

        # Normalize weights
        for i, p in enumerate(self.particles):
            p.weight = weights[i]/s # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        # best_particle = max(self.particles, key=lambda p: p.weight)
        # self.best_map = best_particle.log_odds_map
        #self.best_pose = [best_particle.x, best_particle.y, best_particle.theta]

        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        # Simple likelihood: count how many endpoints match occupied cells
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist):
                continue
            # TODO: Compute the map coordinates of the endpoint: transform the scan into the map frame
            angle_scan = scan_msg.angle_min + i * scan_msg.angle_increment
            x_scan = range_dist * math.cos(angle_scan)
            y_scan = range_dist * math.sin(angle_scan)

            x_hit = robot_x + x_scan * math.cos(robot_theta) - y_scan * math.sin(robot_theta)
            y_hit = robot_y + x_scan * math.sin(robot_theta) + y_scan * math.cos(robot_theta)

            x_cell = int((x_hit - self.map_origin_x) / self.resolution)
            y_cell = int((y_hit - self.map_origin_y) / self.resolution)

            # TODO: Use particle.log_odds_map for scoring
            if 0 <= x_cell < self.map_width_cells and 0 <= y_cell < self.map_height_cells:
                if particle.log_odds_map[y_cell, x_cell] > 0:  # cell likely occupied
                    score += 1.0
                else:
                    score += 0.1  # soft penalty for mismatch

        return score + 1e-6

    def resample_particles(self, particles):
        # TODO: Resample particles
        N = len(particles)
        cdf_sum=0
        p_cdf=[]

        for p in particles:
            cdf_sum = cdf_sum+p.weight
            p_cdf.append(cdf_sum)

        # Calculate the step for random sampling
        step = 1.0/N

        # Sample a value in between [0,step)
        seed = random.uniform(0, step)

        new_particles = []
        last_index = 0
        for h in range(len(particles)):
            while seed > p_cdf[last_index]:
                last_index+=1
            new_particles.append(deepcopy(particles[last_index]))
            seed = seed+step

        return new_particles

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue
            # TODO: Update map: transform the scan into the map frame
            angle_scan = scan_msg.angle_min + i * scan_msg.angle_increment
            x_scan = current_range * math.cos(angle_scan)
            y_scan = current_range * math.sin(angle_scan)

            # Transformar a coordenadas del mapa (pose global del rayo)
            x_hit = robot_x + x_scan * math.cos(robot_theta) - y_scan * math.sin(robot_theta)
            y_hit = robot_y + x_scan * math.sin(robot_theta) + y_scan * math.cos(robot_theta)

            # Coordenadas de inicio (robot) en celdas
            x0 = int((robot_x - self.map_origin_x) / self.resolution)
            y0 = int((robot_y - self.map_origin_y) / self.resolution)

            # Coordenadas de fin (impacto del rayo) en celdas
            x1 = int((x_hit - self.map_origin_x) / self.resolution)
            y1 = int((y_hit - self.map_origin_y) / self.resolution)

            # TODO: Use self.bresenham_line for free cells
            self.bresenham_line(particle, x0, y0, x1, y1)

            # TODO: Update particle.log_odds_map accordingly
            if is_hit:
                if 0 <= x1 < self.map_width_cells and 0 <= y1 < self.map_height_cells:
                    particle.log_odds_map[y1, x1] += self.log_odds_occ
                    particle.log_odds_map[y1, x1] = np.clip(particle.log_odds_map[y1, x1], self.log_odds_min, self.log_odds_max)
            
    def bresenham_line(self, particle, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(particle.log_odds_map[y0, x0], self.log_odds_min, self.log_odds_max)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        # TODO: Fill in map_msg fields and publish one map
        map_msg = OccupancyGrid()
        print("se esta ejecutando")

        best_particle = max(self.particles, key=lambda p: p.weight)
        log_odds = best_particle.log_odds_map #self.best_map 
 
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = "map"
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.orientation.w = 1.0

        occupancy_grid = np.full(log_odds.shape, -1, dtype=np.int8)
        occupancy_grid[log_odds < self.log_odds_free] = 0
        occupancy_grid[log_odds > self.log_odds_occ] = 100

        map_msg.data = occupancy_grid.flatten().tolist()

        self.map_publisher.publish(map_msg)
        print("se termino de ejecutar")
        self.get_logger().debug("Map published.")

    # def broadcast_map_to_odom(self): # FIJARME CUÁL ES LA MATRIZ DE TRANSFORMACIÓN
    #     # TODO: Broadcast map->odom transform
    #     t = TransformStamped()
        
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = "map"
    #     t.child_frame_id = "odom"
    #     t.transform.translation.x = 0.0
    #     t.transform.translation.y = 0.0
    #     t.transform.translation.z = 0.0

    #     t.transform.rotation.x = 0.0
    #     t.transform.rotation.y = 0.0
    #     t.transform.rotation.z = 0.0
    #     t.transform.rotation.w = 1.0
        
    #     self.tf_broadcaster.sendTransform(t)

    # @staticmethod
    # def angle_diff(a, b):
    #     d = a - b
    #     while d > np.pi:
    #         d -= 2 * np.pi
    #     while d < -np.pi:
    #         d += 2 * np.pi
    #     return d

    def broadcast_map_to_odom(self):
        # Obtener la partícula de mayor peso (mejor estimación global)
        best_particle = max(self.particles, key=lambda p: p.weight)
        x_map, y_map, theta_map = best_particle.x, best_particle.y, best_particle.theta

        # Pose odométrica (estimación local)
        if self.last_odom is None:
            return

        x_odom = self.last_odom.pose.pose.position.x
        y_odom = self.last_odom.pose.pose.position.y

        q = self.last_odom.pose.pose.orientation
        rot_odom = R.from_quat([q.x, q.y, q.z, q.w])
        theta_odom = rot_odom.as_euler('xyz')[2]  # yaw

        # Diferencia: T_map_odom = T_map_base * inv(T_odom_base)
        dx = x_map - x_odom
        dy = y_map - y_odom
        dtheta = self.angle_diff(theta_map, theta_odom)

        # Transformación a aplicar
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        # Rotar la diferencia según la orientación del mapa
        # Esto asegura que el frame odom esté bien ubicado en el mapa
        t.transform.translation.x = dx * math.cos(-theta_odom) - dy * math.sin(-theta_odom)
        t.transform.translation.y = dx * math.sin(-theta_odom) + dy * math.cos(-theta_odom)
        t.transform.translation.z = 0.0

        # Convertir dtheta a cuaternion
        q_rot = R.from_euler('z', dtheta).as_quat()
        t.transform.rotation.x = q_rot[0]
        t.transform.rotation.y = q_rot[1]
        t.transform.rotation.z = q_rot[2]
        t.transform.rotation.w = q_rot[3]

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args) # inicializa ROS2 para poder crear nodos
    node = PythonSlamNode() # en base a Node de ROS2, creamos un nodo custom para SLAM para definir suscripciones, publicaciones y lógica
    try:
        rclpy.spin(node) # es un loop de eventos. ROS escucha constántemente odometría o mediciones
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()