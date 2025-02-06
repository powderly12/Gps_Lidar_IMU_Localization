import carla
import numpy as np
import open3d as o3d
import time

class CarlaTester:
    def __init__(self):
        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Set up the vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("model3")[0]  # Tesla Model 3
        spawn_point = self.world.get_map().get_spawn_points()[141]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Set up LiDAR sensor
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "100.0")
        lidar_bp.set_attribute("rotation_frequency", "10")  # 10 Hz
        lidar_transform = carla.Transform(carla.Location(z=2.5))  # Mount on top
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)

        # Set up GPS sensor
        gps_bp = blueprint_library.find("sensor.other.gnss")
        gps_transform = carla.Transform(carla.Location(z=1.5))  # Near vehicle body
        self.gps = self.world.spawn_actor(gps_bp, gps_transform, attach_to=self.vehicle)

        # Get ground truth pose
        self.trajectory = []  # Store ground truth

        # Set up LiDAR callback
        self.lidar.listen(lambda data: self.process_lidar_data(data))

        # Set up GPS callback
        self.gps.listen(lambda data: self.process_gps_data(data))

    def process_lidar_data(self, lidar_data):
        """
        Convert raw CARLA LiDAR data to Open3D PointCloud
        """
        points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Compute ICP-based pose estimation
        transformation_matrix = self.ekf.compute_icp_pose_change(pcd)
        self.ekf.predict_with_icp(transformation_matrix)

    def process_gps_data(self, gps_data):
        """
        Process GPS sensor data
        """
        gps_x, gps_y = gps_data.latitude, gps_data.longitude  # Convert to UTM if needed
        self.ekf.update_with_gps(gps_x, gps_y)

    def get_ground_truth(self):
        """
        Retrieve the vehicle's ground truth pose from CARLA
        """
        transform = self.vehicle.get_transform()
        x_gt, y_gt = transform.location.x, transform.location.y
        yaw_gt = transform.rotation.yaw
        self.trajectory.append((x_gt, y_gt, yaw_gt))
        return x_gt, y_gt, yaw_gt

    def compute_error(self):
        """
        Compute localization error compared to ground truth.
        """
        x_est, y_est, _ = self.ekf.get_state()
        x_gt, y_gt, _ = self.get_ground_truth()
        error = np.sqrt((x_est - x_gt) ** 2 + (y_est - y_gt) ** 2)
        return error
