import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Step 1: Simulate user accuracy data (time series)
np.random.seed(42)
sessions = np.arange(1, 31)
accuracy = np.clip(60 + sessions * 1.5 + np.random.randn(30) * 2, 60, 98)  # Simulated growth

# Step 2: Train AR model
model = AutoReg(accuracy, lags=3)  # Use last 3 sessions to predict next
model_fit = model.fit()

# Step 3: Forecast next 10 sessions
forecast = model_fit.predict(start=len(accuracy), end=len(accuracy)+9, dynamic=False)

# Step 4: Plot
plt.figure(figsize=(9, 5))
plt.plot(sessions, accuracy, label='Actual Accuracy (Sessions 1–30)', color='blue', marker='o')
plt.plot(np.arange(31, 41), forecast, label='Predicted Accuracy (Sessions 31–40)', color='orange', linestyle='--', marker='x')

plt.title("User Accuracy Forecast Using Autoregression", fontsize=14)
plt.xlabel("Session Number")
plt.ylabel("Accuracy (%)")
plt.ylim(50, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("autoregression_user_accuracy_forecast.png", dpi=300)
plt.show()
