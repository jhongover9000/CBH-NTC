import numpy as np
import mne
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from mne.decoding import CSP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from imblearn.under_sampling import RandomUnderSampler

# Load EEG data
npz_data = np.load("subject.npz")
X = npz_data['X']  # EEG trials (1287, 56, 400)
y = npz_data['y']  # Labels (unlabeled or partially labeled)
subject_ids = npz_data['subject_ids']  # Subject IDs

sfreq = 256  # Assumed sampling frequency (modify if needed)
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

# Balance Classes using Undersampling
rus = RandomUnderSampler(random_state=42)
X_bal, y_bal = rus.fit_resample(X, y_pred)  # Use original EEG signals, not features

# Display labeling and balancing results
print("Cluster label distribution before balancing:", np.unique(y_pred, return_counts=True))
print("Cluster label distribution after balancing:", np.unique(y_bal, return_counts=True))
print("Mean mu power for each cluster:", [np.mean(mu_power[y_pred == i]) for i in range(2)])
print("Mean beta power for each cluster:", [np.mean(beta_power[y_pred == i]) for i in range(2)])

# Save generated labels
np.save("generated_labels_balanced.npy", y_bal)

# User choice for feature extraction and classification
use_csp = True  # Set to False to use CNN
use_svm = True  # Set to False to use CNN

# Leave-One-Subject-Out (LOSO) Cross-Validation
logo = LeaveOneGroupOut()
accuracies = []
conf_matrices = []

for train_idx, test_idx in logo.split(X_bal, y_bal, groups=subject_ids[rus.sample_indices_]):
    X_train, X_test = X_bal[train_idx], X_bal[test_idx]
    y_train, y_test = y_bal[train_idx], y_bal[test_idx]

    if use_csp:
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_train = csp.fit_transform(X_train, y_train)  # Now correctly using (trials, channels, timepoints)
        X_test = csp.transform(X_test)
    
    if use_svm:
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
    else:
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(56, 400, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
        y_test_pred = (model.predict(X_test) > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_test_pred)
    accuracies.append(acc)
    conf_matrices.append(confusion_matrix(y_test, y_test_pred))
    print(f"LOSO Accuracy for subject {subject_ids[test_idx[0]]}: {acc:.4f}")

# Overall LOSO results
print(f"Mean LOSO Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
