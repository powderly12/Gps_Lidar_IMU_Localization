import time
import numpy as np
import matplotlib.pyplot as plt
from carla_tester import CarlaTester
from error_state_ekf import ErrorStateEKF  # Import your EKF class

# Initialize CARLA and EKF
tester = CarlaTester()
tester.ekf = ErrorStateEKF()

# Logging for evaluation
estimated_path = []
ground_truth_path = []
errors = []

# Run Simulation Loop
for _ in range(500):  # Run for 50 seconds at 10 Hz
    # Retrieve EKF estimated state
    est_state = tester.ekf.get_state()
    estimated_path.append(est_state[:2])

    # Retrieve ground truth pose from CARLA
    gt_state = tester.get_ground_truth()
    ground_truth_path.append(gt_state[:2])

    # Compute localization error
    error = tester.compute_error()
    errors.append(error)

    print(f"Step {_}: Error = {error:.3f} m")

    time.sleep(0.1)  # Sync with CARLA sensors

# Convert lists to NumPy arrays for plotting
estimated_path = np.array(estimated_path)
ground_truth_path = np.array(ground_truth_path)

# Plot estimated vs. ground truth trajectory
plt.figure(figsize=(10, 6))
plt.plot(ground_truth_path[:, 0], ground_truth_path[:, 1], label="Ground Truth", linestyle="dashed")
plt.plot(estimated_path[:, 0], estimated_path[:, 1], label="EKF Estimate", linestyle="solid")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Localization Performance: EKF vs Ground Truth")
plt.legend()
plt.grid()
plt.show()

# Plot localization error over time
plt.figure(figsize=(8, 5))
plt.plot(errors, label="Localization Error (m)")
plt.xlabel("Time step")
plt.ylabel("Error (m)")
plt.title("Localization Error Over Time")
plt.legend()
plt.grid()
plt.show()
