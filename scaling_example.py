from mne.time_frequency import psd_multitaper
from sklearn.preprocessing import RobustScaler
import numpy as np

# Get the data from MNE epochs object
epochs_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

# Reshape data for robust scaling (flatten across time and channels)
n_epochs, n_channels, n_times = epochs_data.shape
epochs_data_reshaped = epochs_data.reshape(n_epochs, -1)

# Apply robust scaling to data
scaler = RobustScaler()
epochs_data_robust_scaled = scaler.fit_transform(epochs_data_reshaped)

# Reshape back to original dimensions
epochs_data_robust_scaled = epochs_data_robust_scaled.reshape(n_epochs, n_channels, n_times)

# Compute PSD for each epoch
freqs = np.logspace(np.log10(1), np.log10(40), 40)
psd, freqs = psd_multitaper(epochs, fmin=1, fmax=40, tmin=0, tmax=None, picks='all', n_jobs=1)

# Normalize the power in specific frequency band (e.g., Alpha: 8-13 Hz)
alpha_band_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
alpha_power = np.mean(psd[:, alpha_band_idx, :], axis=1)

# Normalize across subjects
alpha_power_normalized = (alpha_power - np.mean(alpha_power, axis=0)) / np.std(alpha_power, axis=0)

# Now you can analyze alpha power changes across subjects with preserved temporal dynamics
