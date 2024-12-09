# Preprocess Script
# Joseph Hong
# Desc: for preprocessing and labeling data
# ==================================================================================================
# ==================================================================================================
# Label Subjects
import numpy as np
import atc
import keras
from mne.io import read_epochs_eeglab
from mne import Epochs, find_events
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

set_dir = './epoched/'

print("Preprocessing Subjects...")

# Split subjects by KMI vs VMI
sub_kmi = [1, 2, 3, 5, 6, 12, 13, 14, 15, 21, 22, 23, 26, 27, 28, 30, 31, 33]
sub_vmi = [4, 7, 8, 9, 10, 11, 16, 17, 19, 24, 25, 29, 32]
sub_etc = [18, 20]

files_kmi = []
files_vmi = []

for n in sub_kmi:
    files_kmi.append(set_dir + 'MIT' + str(n) + "_INT.set")

for n in sub_vmi:
    files_vmi.append(set_dir + 'MIT' + str(n) + "_INT.set")

# Load and preprocess .set files
subject_files = {
    'KMI': files_kmi,  # Group 1
    'VMI': files_vmi,  # Group 2
}
group_labels = {'KMI': 0, 'VMI': 1}

all_data = []
all_labels = []
subject_ids = []
chan2drop = ["T7", "T8", 'FT7', 'FT8']
chan2use = ['C3', 'O1', 'O2']

for group, files in subject_files.items():
    group_label = group_labels[group]
    # Use the corresponding subject number (from sub_kmi or sub_vmi)
    if group == 'KMI':
        subjects = sub_kmi
    else:
        subjects = sub_vmi
    
    for file, subject_num in zip(files, subjects):
        epochs = read_epochs_eeglab(file)

        # Drop additional channels to fit BFN
        epochs = epochs.drop_channels(chan2drop)

        # Pick specific channels: C3, O1, O2 (if needed)
        # epochs = epochs.pick_channels(chan2use)

        print("Channels after dropping:", epochs.info['ch_names'])
        # Downsample to 100 Hz (400 timepoints for 4 seconds)
        epochs = epochs.resample(100, verbose=True)
        # Crop from -1 to 3
        epochs = epochs.crop(-1, 3, False, False)

        # Append data, labels, and subject identifiers
        print(f"Processing subject {subject_num}, trials: {len(epochs)}")
        all_data.append(epochs.get_data())  # Shape: (n_epochs, n_channels, n_times)
        all_labels.append(np.full(len(epochs), group_label))
        subject_ids.extend([subject_num] * len(epochs))  # Use actual subject numbers

# Combine all subjects' data
X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)
subject_ids = np.array(subject_ids)

print(f"X shape: {X.shape}, y shape: {y.shape}, Subject IDs shape: {subject_ids.shape}")


print(X.shape)
print(y.shape)

print("Done Preprocessing Subjects.")

# # Save data and labels to a file
save_path = "subject_data.npz"  # Specify your desired file path
np.savez(save_path, X=X, y=y, subject_ids = subject_ids)

print(f"Data saved to {save_path}.")