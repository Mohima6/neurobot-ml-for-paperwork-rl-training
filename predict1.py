import matplotlib.pyplot as plt

# Simulated data (you can replace this with real values later)
latency_ms = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
control_accuracy = [96, 94, 92, 89, 86, 82, 78, 72, 68, 60]

plt.figure(figsize=(8, 5))
plt.plot(latency_ms, control_accuracy, marker='o', color='blue', linestyle='--', linewidth=2)

plt.title('System Latency vs Control Accuracy', fontsize=14)
plt.xlabel('Communication Latency (ms)', fontsize=12)
plt.ylabel('Robotic Control Accuracy (%)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("latency_vs_accuracy.png", dpi=300)
plt.show()
