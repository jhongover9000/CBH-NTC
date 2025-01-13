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
npz_data = np.load("subject_data_v4.npz")
X = npz_data['X']  # EEG trials (1287, 56, 400)
y = npz_data['y']  # Labels (unlabeled or partially labeled)
subject_ids = npz_data['subject_ids']  # Subject IDs

sfreq = 256  # Assumed sampling frequency (modify if needed)
freq_bands = {'mu': (6, 11), 'beta': (11, 28)}

# Define channels of interest
kmi_channels = ["C3", "C4", "Cz"]
vmi_channels = ["O1", "O2", "Pz"]
channel_names = [f'Ch{i}' for i in range(56)]  # Placeholder names, replace with actual

kmi_indices = [channel_names.index(ch) for ch in kmi_channels if ch in channel_names]
vmi_indices = [channel_names.index(ch) for ch in vmi_channels if ch in channel_names]

def compute_band_power(X, band, sfreq, channels):
    """Computes average power in a frequency band for specific channels."""
    band_power = []
    for trial in X:
        psd, freqs = mne.time_frequency.psd_array_multitaper(trial, sfreq=sfreq, fmin=band[0], fmax=band[1])
        band_power.append(np.mean(psd[channels], axis=0))  # Average power per selected channels
    return np.array(band_power)

# Perform rule-based labeling per subject
rule_based_labels = np.full(len(y), -1)  # Initialize with -1 (unlabeled)

for subject in np.unique(subject_ids):
    subject_mask = subject_ids == subject
    X_subject = X[subject_mask]
    
    mu_power_kmi = compute_band_power(X_subject, freq_bands['mu'], sfreq, kmi_indices)
    beta_power_kmi = compute_band_power(X_subject, freq_bands['beta'], sfreq, kmi_indices)
    mu_power_vmi = compute_band_power(X_subject, freq_bands['mu'], sfreq, vmi_indices)
    beta_power_vmi = compute_band_power(X_subject, freq_bands['beta'], sfreq, vmi_indices)
    
    threshold_mu = np.median(mu_power_kmi)
    threshold_beta = np.median(beta_power_kmi)
    
    for i, trial_idx in enumerate(np.where(subject_mask)[0]):
        if np.all(mu_power_kmi[i] < threshold_mu) and np.all(beta_power_kmi[i] < threshold_beta):
            rule_based_labels[trial_idx] = 0  # KMI
        elif np.all(mu_power_vmi[i] > threshold_mu) and np.all(beta_power_vmi[i] > threshold_beta):
            rule_based_labels[trial_idx] = 1  # VMI

# Use clustering only for unlabeled data
unlabeled_mask = rule_based_labels == -1
if np.any(unlabeled_mask):
    features = np.hstack([compute_band_power(X, freq_bands['mu'], sfreq, range(56)),
                           compute_band_power(X, freq_bands['beta'], sfreq, range(56))])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    y_pred = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(features_scaled)
    rule_based_labels[unlabeled_mask] = y_pred[unlabeled_mask]

# Balance Classes using Undersampling
rus = RandomUnderSampler(random_state=42)
X_flat = X.reshape(X.shape[0], -1)  # Flatten EEG data
X_bal_flat, y_bal = rus.fit_resample(X_flat, y_pred)
X_bal = X_bal_flat.reshape(-1, X.shape[1], X.shape[2])  # Restore EEG shape

# Display labeling and balancing results
print("Cluster label distribution before balancing:", np.unique(y_pred, return_counts=True))
print("Cluster label distribution after balancing:", np.unique(y_bal, return_counts=True))
# print("Mean mu power for each cluster:", [np.mean(mu_power[y_pred == i]) for i in range(2)])
# print("Mean beta power for each cluster:", [np.mean(beta_power[y_pred == i]) for i in range(2)])

# Save generated labels
np.save("generated_labels_balanced.npy", y_bal)

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
        if len(np.unique(y_train)) < 2:
            print(f"Skipping CSP for this fold: Only one class found in training set (Labels: {np.unique(y_train)})")
            continue  # Skip this iteration
        X_train = csp.fit_transform(X_train, y_train)
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
