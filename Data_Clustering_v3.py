import numpy as np
import mne
from mne.io import read_epochs_eeglab
from mne import Epochs, find_events
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from mne.decoding import CSP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from imblearn.under_sampling import RandomUnderSampler

set_dir = './epoched/'
file_temp = set_dir + 'MIT' + str(1) + "_INT.set"
epochs = read_epochs_eeglab(file_temp)
num_timepoints = 1024

chan2drop = ["T7", "T8", 'FT7', 'FT8']
# chan2use = ['C3', 'O1', 'O2']

# Drop additional channels to fit BFN
epochs = epochs.drop_channels(chan2drop)

# Load EEG data
npz_data = np.load("subject_data_v3.npz")
X = npz_data['X']  # EEG trials (1287, 56, 400)
y = npz_data['y']  # Labels (unlabeled or partially labeled)
subject_ids = npz_data['subject_ids']  # Subject IDs

sfreq = 256  # Assumed sampling frequency (modify if needed)
freq_bands = {'mu': (8, 13), 'beta': (13, 30)}

# Define channels of interest
kmi_channels = ["C3", "C4", "Cz"]
vmi_channels = ["O1", "O2", "Pz"]
channel_names = epochs.info['ch_names']  # Placeholder names, replace with actual

kmi_indices = [i for i, ch in enumerate(channel_names) if ch in kmi_channels]
vmi_indices = [i for i, ch in enumerate(channel_names) if ch in vmi_channels]

def compute_band_power(X, band, sfreq, channels):
    psd, freqs = mne.time_frequency.psd_array_multitaper(
        X, sfreq=sfreq, fmin=band[0], fmax=band[1], adaptive=True
    )
    print(f"Requested band: {band}, Available freqs: {freqs[:10]} ... {freqs[-10:]}")  # Debugging
    print(f"PSD shape: {psd.shape}, Channel indices: {channels}")

    if psd.shape[1] == 0:  # No frequency bins found
        raise ValueError(f"No frequencies found in range {band}. Check fmin, fmax, and sfreq.")

    return np.mean(psd[:, channels, :], axis=2)  # Mean across time dimension

# Perform rule-based labeling per subject using baseline normalization
rule_based_labels = np.full(len(y), -1)  # Initialize with -1 (unlabeled)

for subject in np.unique(subject_ids):
    subject_mask = subject_ids == subject
    X_subject = X[subject_mask]
    print(f"Subject {subject} - Trials: {X_subject.shape[0]}")

    # Compute baseline for the first second (-1 to onset)
    baseline_mu_kmi = compute_band_power(X_subject[:, :, :250], freq_bands['mu'], sfreq, kmi_indices)
    baseline_beta_kmi = compute_band_power(X_subject[:, :, :250], freq_bands['beta'], sfreq, kmi_indices)
    
    baseline_mu_vmi = compute_band_power(X_subject[:, :, :250], freq_bands['mu'], sfreq, vmi_indices)
    baseline_beta_vmi = compute_band_power(X_subject[:, :, :250], freq_bands['beta'], sfreq, vmi_indices)

    print(f"Baseline Mu KMI shape: {baseline_mu_kmi.shape}, Values: {baseline_mu_kmi[:5]}")
    print(f"Baseline Beta KMI shape: {baseline_beta_kmi.shape}, Values: {baseline_beta_kmi[:5]}")

    mu_power_kmi = compute_band_power(X_subject[:, :, 256:], freq_bands['mu'], sfreq, kmi_indices).mean(axis=0)
    beta_power_kmi = compute_band_power(X_subject[:, :, 256:], freq_bands['beta'], sfreq, kmi_indices).mean(axis=0)
    mu_power_vmi = compute_band_power(X_subject[:, :, 256:], freq_bands['mu'], sfreq, vmi_indices).mean(axis=0)
    beta_power_vmi = compute_band_power(X_subject[:, :, 256:], freq_bands['beta'], sfreq, vmi_indices).mean(axis=0)
    
    erd_kmi = (mu_power_kmi < baseline_mu_kmi) & (beta_power_kmi < baseline_beta_kmi)
    ers_vmi = (mu_power_vmi > baseline_mu_vmi) & (beta_power_vmi > baseline_beta_vmi)
    erd_kmi = erd_kmi.mean(axis=1)
    ers_vmi = ers_vmi.mean(axis=1)

    print(f"erd_kmi shape: {erd_kmi.shape}, ers_vmi shape: {ers_vmi.shape}")

    rule_based_labels[subject_mask] = np.where(erd_kmi, 0, np.where(ers_vmi, 1, -1))

unlabeled_mask = rule_based_labels == -1
if np.any(unlabeled_mask):
    features = np.hstack([compute_band_power(X, freq_bands['mu'], sfreq, range(56)),
                           compute_band_power(X, freq_bands['beta'], sfreq, range(56))])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    y_pred = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(features_scaled)
    rule_based_labels[unlabeled_mask] = y_pred[unlabeled_mask]

# rus = RandomUnderSampler(random_state=42)
# X_flat = X.reshape(X.shape[0], -1)  # Flatten EEG trials for resampling
# X_bal, y_bal = rus.fit_resample(X_flat, rule_based_labels)

# print("Shape of X_bal after resampling:", X_bal.shape)  # Inspect the shape of X_bal

# # You can also print and debug this step:
# print("New shape of X_bal:", X_bal.shape)
# X_bal = X_bal.reshape(y_bal.shape[0], 56, num_timepoints)  # Reshape back to original dimensions

# Use the original X and y
X_bal = X  # No resampling needed
y_bal = rule_based_labels

# Display labeling and balancing results
print("Cluster label distribution before balancing:", np.unique(y_pred, return_counts=True))
print("Cluster label distribution after balancing:", np.unique(y_bal, return_counts=True))
print("Mean mu power for each cluster:", [np.mean(mu_power_kmi),np.mean(mu_power_vmi)])
print("Mean beta power for each cluster:", [np.mean(beta_power_kmi),np.mean(beta_power_vmi)])

np.save("generated_labels_balanced.npy", y_bal)

use_csp = True
use_svm = True

logo = LeaveOneGroupOut()
accuracies = []
conf_matrices = []

for train_idx, test_idx in logo.split(X_bal, y_bal, groups=subject_ids[rus.sample_indices_]):
    X_train, X_test = X_bal[train_idx], X_bal[test_idx]
    y_train, y_test = y_bal[train_idx], y_bal[test_idx]

    if len(np.unique(y_train)) < 2:
        continue
    
    if use_csp:
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_train = csp.fit_transform(X_train, y_train)
        X_test = csp.transform(X_test)
    
    if use_svm:
        model = SVC(kernel='linear', class_weight='balanced')
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
    else:
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(56, num_timepoints, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train[..., np.newaxis], y_train, epochs=10, batch_size=16, validation_data=(X_test[..., np.newaxis], y_test))
        y_test_pred = (model.predict(X_test[..., np.newaxis]) > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_test_pred)
    accuracies.append(acc)
    conf_matrices.append(confusion_matrix(y_test, y_test_pred))

print(f"Mean LOSO Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
