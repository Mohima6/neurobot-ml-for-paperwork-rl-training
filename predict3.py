import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Simulated data for predicted vs actual movement (e.g., in 2D space)
actual_x = np.sin(np.linspace(0, 10, 100))
actual_y = np.cos(np.linspace(0, 10, 100))
predicted_x = actual_x + np.random.normal(0, 0.1, 100)  # Simulated prediction error
predicted_y = actual_y + np.random.normal(0, 0.1, 100)

# Calculate error
error_x = predicted_x - actual_x
error_y = predicted_y - actual_y
error = np.sqrt(error_x**2 + error_y**2)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(actual_x, actual_y, label='Actual Trajectory')
plt.plot(predicted_x, predicted_y, label='Predicted Trajectory', linestyle='dashed')
plt.scatter(predicted_x, predicted_y, c=error, cmap='viridis', label='Error Magnitude', alpha=0.6)
plt.colorbar(label='Error Magnitude')
plt.title('Predicted vs Actual Robotic Movement')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.show()
