import numpy as np
from mne.io import read_raw_eeglab
from mne import Epochs, find_events
from sklearn.model_selection import train_test_split

# Load and preprocess .set files
subject_files = {
    'group1': ['subject1.set', 'subject2.set'],  # Group 1
    'group2': ['subject3.set', 'subject4.set'],  # Group 2
}
group_labels = {'group1': 0, 'group2': 1}

all_data = []
all_labels = []

for group, files in subject_files.items():
    group_label = group_labels[group]
    for file in files:
        raw = read_raw_eeglab(file, preload=True)
        events = find_events(raw)
        event_id = {'specific_event': group_label}
        epochs = Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)

        # Append data and labels
        all_data.append(epochs.get_data())  # Shape: (n_epochs, n_channels, n_times)
        all_labels.append(np.full(len(epochs), group_label))

# Combine all subjects' data
X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
