import mne
from mne.preprocessing import ICA
from mne.datasets import sample

# Step 1: Load the raw data and create epochs (replace with your own data)
data_path = sample.data_path()
raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif', preload=True)

# Find events and create epochs (this is just an example, use your actual data)
events = mne.find_events(raw)
event_id = {'auditory': 1, 'visual': 2}  # Adjust this to your specific event IDs
tmin, tmax = -0.2, 0.5  # Epoch time window (adjust as needed)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)

# Step 2: Apply ICA to the epochs
ica = ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(epochs)

# Step 3: Inspect the ICA components to identify eyeblink artifacts
ica.plot_components()  # View the topographies of each component
ica.plot_sources(epochs)  # View the time series of the components

# Look for components that show typical eyeblink patterns:
# - Eye blink artifacts are typically seen in the frontal electrodes and may have a sharp peak in the time series.

# Step 4: Exclude the eyeblink-related components
# Suppose you identify that components 0 and 1 correspond to eyeblink artifacts
ica.exclude = [0, 1]  # Exclude components 0 and 1 (replace with your identified components)

# Step 5: Apply the ICA solution to remove the eyeblink components
epochs_clean = ica.apply(epochs)

# Step 6: Visualize the cleaned data
epochs.plot()         # Original epochs
epochs_clean.plot()   # Cleaned epochs after ICA

# You can save the cleaned epochs for future use
epochs_clean.save('cleaned_epochs-epo.fif')
