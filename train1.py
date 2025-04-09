import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Simulate EEG data for 3 commands: "move forward", "turn left", "stop"
# These will be represented as numerical features
np.random.seed(42)

# Simulate EEG signals for 3 classes: 0=Move Forward, 1=Turn Left, 2=Stop
n_samples = 500
n_features = 10  # Simulating 10 features per sample (representing EEG channels)

# Create random data for EEG signals
X = np.random.rand(n_samples, n_features)

# Assign labels (e.g., 0 = Move Forward, 1 = Turn Left, 2 = Stop)
y = np.random.choice([0, 1, 2], size=n_samples)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a simple Neural Network (DNN)
model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict the commands for the test set
y_pred = model.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)

# Plot both the confusion matrix and the robot's movement trajectory together

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Confusion Matrix Plot (Left)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Move Forward', 'Turn Left', 'Stop'], yticklabels=['Move Forward', 'Turn Left', 'Stop'], ax=ax[0])
ax[0].set_title('Confusion Matrix - Brain Command Decoding')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

# Robot Movement Plot (Right)
trajectory = {'Move Forward': (1, 0), 'Turn Left': (0, 1), 'Stop': (0, 0)}
x_pos, y_pos = 0, 0  # Start position

# Plot the robot's movement
for command in y_pred:
    if command == 0:
        # Move Forward
        dx, dy = trajectory['Move Forward']
        x_pos += dx
        y_pos += dy
        ax[1].plot(x_pos, y_pos, 'bo', markersize=5)
    elif command == 1:
        # Turn Left
        dx, dy = trajectory['Turn Left']
        x_pos += dx
        y_pos += dy
        ax[1].plot(x_pos, y_pos, 'ro', markersize=5)
    elif command == 2:
        # Stop
        dx, dy = trajectory['Stop']
        x_pos += dx
        y_pos += dy
        ax[1].plot(x_pos, y_pos, 'go', markersize=5)

# Customize robot movement plot
ax[1].set_title('Robot Movement through Brain Commands (Predicted)')
ax[1].set_xlabel('X Position')
ax[1].set_ylabel('Y Position')
ax[1].grid(True)

# Show the plot
plt.tight_layout()
plt.show()
