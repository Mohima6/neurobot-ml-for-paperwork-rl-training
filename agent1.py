import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from mpl_toolkits.mplot3d import Axes3D


class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions: [move_forward, turn_left, turn_right, stop]
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0], dtype=np.float32),
                                                high=np.array([100, 100, 100], dtype=np.float32),
                                                dtype=np.float32)  # robot's x, y, z position

        # Initialize robot position and angle
        self.robot_position = np.array([50, 50, 0], dtype=np.float64)  # robot starts at center of the grid, z=0
        self.robot_angle = 0  # 0 degrees, facing right
        self.steps_taken = 0

    def reset(self):
        # Reset the robot's position and angle
        self.robot_position = np.array([50, 50, 0], dtype=np.float64)
        self.robot_angle = 0
        self.steps_taken = 0
        return self.robot_position  # return initial state (position)

    def step(self, action):
        # Action [0: move forward, 1: turn left, 2: turn right, 3: stop]

        if action == 0:  # move forward
            self.robot_position += np.array([np.cos(np.deg2rad(self.robot_angle)),
                                             np.sin(np.deg2rad(self.robot_angle)), 0])
        elif action == 1:  # turn left
            self.robot_angle += 90  # turn left by 90 degrees
        elif action == 2:  # turn right
            self.robot_angle -= 90  # turn right by 90 degrees
        elif action == 3:  # stop
            pass  # no movement

        # Ensure robot position remains within bounds (0-100)
        self.robot_position = np.clip(self.robot_position, 0, 100)

        # Increase steps count
        self.steps_taken += 1

        # Stop if steps exceed a limit (to avoid infinite loop)
        done = self.steps_taken >= 100

        # Small reward for each step
        reward = -1  # negative reward for each step to encourage quicker completion

        info = {}  # additional info (empty in this case)

        return self.robot_position, reward, done, info

    def render(self):
        # Visualization is handled by plotting the robot's position
        pass


# Create the environment
env = DummyVecEnv([lambda: RobotEnv()])

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# After training, use the trained model to predict actions for a full path
obs = env.reset()

# List to store robot's position over time
x_positions = []
y_positions = []
z_positions = []

# Simulate the robot's movement using the trained policy (predict actions step by step)
for _ in range(100):  # simulate 100 steps
    action, _states = model.predict(obs)  # predict the action using the trained policy
    obs, reward, done, info = env.step(action)  # execute the action in the environment

    # Track the robot's position over time
    x_positions.append(obs[0][0])  # x coordinate
    y_positions.append(obs[0][1])  # y coordinate
    z_positions.append(obs[0][2])  # z coordinate

    if done:
        break

# Create a clustering plot to differentiate between paths
# Plot the robot's entire trajectory (path planning)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter the training data to show the robot's initial exploration
ax.scatter(x_positions[:20], y_positions[:20], z_positions[:20], color='orange', label='Trained Path', s=30)

# Now plot the full trajectory predicted by the model
ax.plot(x_positions, y_positions, z_positions, label="Predicted Path (Post-training)", color='blue', linewidth=2)

# Plot the actual trajectory (just for comparison, assuming some optimal path)
# Let's assume an optimal path is a straight line from start to end in 3D space
optimal_path_x = np.linspace(50, 90, 100)
optimal_path_y = np.linspace(50, 90, 100)
optimal_path_z = np.linspace(0, 20, 100)
ax.plot(optimal_path_x, optimal_path_y, optimal_path_z, '--', label='Actual Path (Optimal)', color='green', linewidth=2)

# Highlight the start and end points
ax.scatter(x_positions[0], y_positions[0], z_positions[0], color='green', label='Start', zorder=5)
ax.scatter(x_positions[-1], y_positions[-1], z_positions[-1], color='red', label='End', zorder=5)

# Adding labels and title
ax.set_title("Robot's 3D Path Planning Using Trained Model vs Actual Optimal Path")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.legend()
ax.grid(True)

# Show the plot
plt.show()
