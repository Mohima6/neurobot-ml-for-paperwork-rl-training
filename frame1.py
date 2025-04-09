import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# Step 1: Generate the dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2,
                            n_clusters_per_class=1, n_informative=2,
                            n_redundant=0, n_repeated=0)

# Step 2: Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title('Generated Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Standardize the dataset (for SVM and neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")

# Step 6: Train DNN model
dnn_model = Sequential([
    Dense(8, activation='relu', input_dim=2),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0)
dnn_pred = (dnn_model.predict(X_test_scaled) > 0.5).astype("int32")
dnn_accuracy = accuracy_score(y_test, dnn_pred)
print(f"DNN Model Accuracy: {dnn_accuracy * 100:.2f}%")

# Step 7: CNN Model (This part is for illustrative purposes, CNNs are usually used for image data)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, 2, 1)  # Reshaping for CNN input
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, 2, 1)

cnn_model = Sequential([
    Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=(1, 2, 1)),
    Flatten(),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_reshaped, y_train, epochs=50, batch_size=10, verbose=0)
cnn_pred = (cnn_model.predict(X_test_reshaped) > 0.5).astype("int32")
cnn_accuracy = accuracy_score(y_test, cnn_pred)
print(f"CNN Model Accuracy: {cnn_accuracy * 100:.2f}%")

# Step 8: Visualize the decision boundaries
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot for SVM
plt.figure(figsize=(8, 6))
Z_svm = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
plt.contourf(xx, yy, Z_svm, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o')
plt.title("SVM Decision Boundary")
plt.show()

# Plot for DNN
plt.figure(figsize=(8, 6))
Z_dnn = (dnn_model.predict(np.c_[xx.ravel(), yy.ravel()]) > 0.5).astype("int32")
Z_dnn = Z_dnn.reshape(xx.shape)
plt.contourf(xx, yy, Z_dnn, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o')
plt.title("DNN Decision Boundary")
plt.show()

# Plot for CNN (visualization after reshaping inputs)
plt.figure(figsize=(8, 6))
Z_cnn = (cnn_model.predict(np.c_[xx.ravel(), yy.ravel()].reshape(-1, 1, 2, 1)) > 0.5).astype("int32")
Z_cnn = Z_cnn.reshape(xx.shape)
plt.contourf(xx, yy, Z_cnn, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o')
plt.title("CNN Decision Boundary")
plt.show()
