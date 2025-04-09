import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

# Generate simulated EEG data (random for demonstration)
np.random.seed(42)
n_samples = 1000  # Number of samples
n_channels = 5    # Number of EEG channels

# Create random EEG signals (clean signals)
X_clean = np.random.randn(n_samples, n_channels)

# Add noise to simulate a noisy EEG signal
noise = np.random.normal(0, 0.5, size=(n_samples, n_channels))
X_noisy = X_clean + noise

# Function to calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(signal, noise):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

# Calculate SNR before denoising (using noisy signal)
snr_before = calculate_snr(X_clean, noise)

# Apply ICA to denoise the signal
ica = FastICA(n_components=n_channels, random_state=42)
X_denoised = ica.fit_transform(X_noisy)

# Calculate the denoised SNR
denoised_noise = X_noisy - X_denoised
snr_after = calculate_snr(X_clean, denoised_noise)

# Calculate SNR for each signal/channel for better visualization
snr_before_channel = [calculate_snr(X_clean[:, i], noise[:, i]) for i in range(n_channels)]
snr_after_channel = [calculate_snr(X_clean[:, i], denoised_noise[:, i]) for i in range(n_channels)]

# Set up the plot with complex and beautiful aesthetics
plt.figure(figsize=(12, 6))

# Plotting the SNR comparison
ax1 = plt.subplot(121)
sns.lineplot(x=np.arange(n_channels), y=snr_before_channel, marker="o", label="SNR Before Denoising", color='red', linewidth=2)
sns.lineplot(x=np.arange(n_channels), y=snr_after_channel, marker="o", label="SNR After Denoising", color='green', linewidth=2)
ax1.set_title('Signal-to-Noise Ratio (SNR) Comparison', fontsize=16)
ax1.set_xlabel('EEG Channels', fontsize=12)
ax1.set_ylabel('SNR (dB)', fontsize=12)
ax1.set_xticks(np.arange(n_channels))
ax1.set_xticklabels([f'Ch{i+1}' for i in range(n_channels)])
ax1.legend()

# Plotting the histograms of SNR before and after denoising
ax2 = plt.subplot(122)
sns.histplot(snr_before_channel, kde=True, color='red', label="Before Denoising", ax=ax2, bins=15)
sns.histplot(snr_after_channel, kde=True, color='green', label="After Denoising", ax=ax2, bins=15)
ax2.set_title('Histogram of SNR Distribution', fontsize=16)
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()

plt.tight_layout()
plt.show()
