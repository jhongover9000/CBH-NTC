# Preprocess Script for MIT data comparing ST and NT
# Joseph Hong
# Desc: for preprocessing and labeling data
# ==================================================================================================
# ==================================================================================================
# Label Subjects
import numpy as np
from mne.io import read_epochs_eeglab
from mne import Epochs, find_events
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

nSub = 33
set_dir = './epoched/'

sr = 256
start_time = 0  # Start time in seconds (relative to the epoch start)
end_time = 3

# num_timepoints = 400  # Number of time points to keep
# end_time = start_time + (num_timepoints - 1) / freq

print("Preprocessing Subjects...")

# Split subjects NT and ST data
subs_all = list(range(1,nSub))

files_nt = []
files_st = []

for n in subs_all:
    files_nt.append(set_dir + 'MIT' + str(n) + "_INT.set")

for n in subs_all:
    files_st.append(set_dir + 'MIT' + str(n) + "_IST.set")

# Load and preprocess .set files
subject_files = {
    'NT': files_nt,  # Group 1
    'ST': files_st,  # Group 2
}
group_labels = {'NT': 0, 'ST': 1}

all_data = []
all_labels = []
subject_ids = []
chan2drop = ["T7", "T8", 'FT7', 'FT8']
chan2use = ['C3', 'O1', 'O2']

# Re-label data
for group, files in subject_files.items():
    group_label = group_labels[group]
    for file, subject_num in zip(files, subjects):
        epochs = read_epochs_eeglab(file)

        # Drop additional channels to fit BFN
        epochs = epochs.drop_channels(chan2drop)

        # Pick specific channels: C3, O1, O2 (if needed)
        # epochs = epochs.pick_channels(chan2use)
        print("Channels after dropping:", epochs.info['ch_names'])

        # Downsample to 100 Hz (400 timepoints for 4 seconds)
        epochs = epochs.resample(sr, verbose=True)

        # Crop timeframe
        epochs = epochs.crop(start_time, end_time, False, False)

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
save_path = "subject_data_v4.npz"  # Specify your desired file path
np.savez(save_path, X=X, y=y, subject_ids = subject_ids)

print(f"Data saved to {save_path}.")