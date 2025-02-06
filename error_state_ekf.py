import numpy as np
import open3d as o3d

class ErrorStateEKF:
    def __init__(self, dt=0.1):
        """
        Initialize the Error-State EKF for localization using LiDAR and GPS.
        :param dt: Time step for prediction (seconds)
        """
        self.dt = dt  # Time step
        
        # Nominal state [x, y, theta]
        self.x_nominal = np.zeros(3)  # [x, y, heading]
        
        # Error state covariance
        self.P = np.eye(3) * 0.01  # Small initial uncertainty
        
        # Process noise covariance (LiDAR ICP uncertainty)
        self.Q = np.diag([0.01, 0.01, 0.005])
        
        # Measurement noise covariance (GPS uncertainty)
        self.R_GPS = np.diag([0.5, 0.5])  # GPS noise in x and y
        
        # State transition matrix (linearized motion model)
        self.F = np.eye(3)
        
        # Measurement matrix for GPS
        self.H_GPS = np.array([[1, 0, 0], [0, 1, 0]])  # GPS measures x, y
        
        self.prev_pcd = None  # Store previous LiDAR point cloud

    def preprocess_pcd(self, pcd):
        """
        Preprocess LiDAR point cloud:
        - Downsample to reduce processing time
        """
        return pcd.voxel_down_sample(voxel_size=0.2)

    def compute_icp_pose_change(self, current_pcd):
        """
        Perform ICP registration to estimate pose change.
        :param current_pcd: Current LiDAR point cloud (Open3D PointCloud)
        :return: Transform matrix (4x4)
        """
        if self.prev_pcd is None:
            self.prev_pcd = current_pcd
            return np.eye(4)  # No motion if it's the first frame

        # Preprocess point clouds
        current_pcd = self.preprocess_pcd(current_pcd)
        self.prev_pcd = self.preprocess_pcd(self.prev_pcd)

        # Perform ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            current_pcd, self.prev_pcd, max_correspondence_distance=1.0,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        # Update previous point cloud
        self.prev_pcd = current_pcd

        return reg_p2p.transformation  # 4x4 Transformation Matrix

    def extract_pose_changes(self, transformation_matrix):
        """
        Extract x, y translation and orientation changes from a 4x4 transformation matrix.
        """
        dx = transformation_matrix[0, 3]
        dy = transformation_matrix[1, 3]
        yaw = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
        return dx, dy, yaw

    def predict_with_icp(self, transformation_matrix):
        """
        Predict step using motion estimated from LiDAR ICP.
        :param transformation_matrix: 4x4 transformation matrix from ICP
        """
        dx, dy, dtheta = self.extract_pose_changes(transformation_matrix)
        
        # Predict nominal state
        self.x_nominal[0] += dx  # x position
        self.x_nominal[1] += dy  # y position
        self.x_nominal[2] += dtheta  # Heading change

        # Compute Jacobian F (linearized motion model)
        self.F = np.array([
            [1, 0, -dy],
            [0, 1,  dx],
            [0, 0,  1]
        ])
        
        # Propagate error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_with_gps(self, gps_x, gps_y):
        """
        Update step using GPS measurements.
        :param gps_x: Measured x position from GPS
        :param gps_y: Measured y position from GPS
        """
        z = np.array([gps_x, gps_y])  # GPS measurement

        # Compute Kalman gain
        S = self.H_GPS @ self.P @ self.H_GPS.T + self.R_GPS
        K = self.P @ self.H_GPS.T @ np.linalg.inv(S)
        
        # Compute innovation (measurement residual)
        error_state = K @ (z - self.x_nominal[:2])
        
        # Apply error state correction
        self.x_nominal[:2] += error_state[:2]
        
        # Update error covariance
        self.P = (np.eye(3) - K @ self.H_GPS) @ self.P

    def get_state(self):
        """
        Return the current state estimate.
        """
        return self.x_nominal
