import numpy as np
import mne
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load EEG data
npz_data = np.load("your_data.npz")
X = npz_data['X']  # EEG trials (1287, 56, 400)
y = npz_data['y']  # Labels (unlabeled or partially labeled)
subject_ids = npz_data['subject_ids']  # Subject IDs

sfreq = 250  # Assumed sampling frequency (modify if needed)
freq_bands = {'mu': (8, 13), 'beta': (13, 30)}

def compute_band_power(X, band, sfreq):
    """Computes average power in a frequency band."""
    band_power = []
    for trial in X:
        psd, freqs = mne.time_frequency.psd_array_multitaper(trial, sfreq=sfreq, fmin=band[0], fmax=band[1])
        band_power.append(np.mean(psd, axis=1))  # Average power per channel
    return np.array(band_power)

# Extract features
mu_power = compute_band_power(X, freq_bands['mu'], sfreq)
beta_power = compute_band_power(X, freq_bands['beta'], sfreq)
features = np.hstack([mu_power, beta_power])

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering to label data
y_pred = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(features_scaled)

# Assign cluster labels based on ERD/ERS patterns
if np.mean(mu_power[y_pred == 0]) < np.mean(mu_power[y_pred == 1]):
    y_pred = 1 - y_pred  # Swap labels if needed (assume KMI has stronger ERD)

# Save generated labels
np.save("generated_labels.npy", y_pred)

# Train a classifier (SVM)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, y_pred, test_size=0.2, random_state=42)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_test_pred = svm.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
