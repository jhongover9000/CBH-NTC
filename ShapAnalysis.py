# Import necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
import keras
from keras.models import Sequential
from collections import defaultdict
import pandas as pd
# import BFN
from datetime import datetime
import gc
import shap
from shap.explainers._deep import deep_tf
import pickle


from tensorflow.keras.models import Model
# from deepexplain.tensorflow import DeepExplain
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from Models.EEGModels import EEGNet,DeepConvNet,
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential

weights_dir = "./Weights/"


# Load Data
data = np.load("subject_data.npz")
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

# ==================================================================================================
# Define Helper Functions

# Fit Transform (for Training Data)
def scaler_fit_transform(X_train, X_test):
    """
    Fits a StandardScaler on the training data and transforms both training and test data.
    
    Args:
        X_train: Training data (n_samples, n_channels, n_timepoints)
        X_test: Test data (n_samples, n_channels, n_timepoints)
    
    Returns:
        X_train_scaled, X_test_scaled: Standardized training and test data
    """
    n_channels = X_train.shape[1]
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)

    scaler = StandardScaler()

    for i in range(n_channels):  # Iterate over each channel
        # Fit scaler on the training data for this channel
        scaler.fit(X_train[:, i, :])
        # Transform both training and test data for this channel
        X_train_scaled[:, i, :] = scaler.transform(X_train[:, i, :])
        X_test_scaled[:, i, :] = scaler.transform(X_test[:, i, :])

    return X_train_scaled, X_test_scaled

# Plot History
def plot_training_history(history, timestamp):
    """Plot the training and validation loss and accuracy."""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    fig_file = f"curves/History_{timestamp}.png"
    plt.savefig(fig_file)
    plt.show()

# ==================================================================================================
# Model Training with 5-Fold Cross-Validation
n_splits = 5
epochs = 90
batch_size = 16
learning_rate = 0.00005
weight_decay = 0.01
samples, chans = X.shape[2], X.shape[1]
nb_classes = 2

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize an accumulator for the confusion matrix
conf_matrix_accum = np.zeros((nb_classes, nb_classes))

# Initialize metrics tracking
accuracy_per_fold = []
loss_per_fold = []
scores_atc = []
history_list = []

shap_values_all = []
y_test_all = []
y_pred_all = []

# Initialize a list to hold SHAP values for each fold
shap_values_all_folds = []

# Loop through each fold
for fold, (train_index, test_index) in enumerate(skf.split(X_data, y_data)):
    print(f"Processing Fold {fold_number}...")

    # Save the fold indices to a .npz file
    np.savez(f"fold_{fold_number}_indices.npz", train_index=train_index, test_index=test_index)
    
    
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Standardize the data
    X_train, X_test = scaler_fit_transform(X_train, X_test)
    
    # Expand dimensions for compatibility with the model
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])  # Swap axes for EEGNet
    X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(X_train.shape)
    print(X_test.shape)
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    # Train the model (replace with your EEGNet model training)
    model = EEGNet(nb_classes, chans,  samples)
    model.load_weights('final_model_fold_' + str(fold) +'.h5', by_name=True, skip_mismatch=True)
    
    # Ensure model is trained, here it's just loaded from a saved model for simplicity

    # Create SHAP explainer for this fold
    explainer = shap.DeepExplainer(model, X_train[:100])  # Use a subset of X_train as background data

    # Calculate SHAP values for this fold (for X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_all_folds.append(shap_values)  # Store SHAP values for later visualization

    # Visualize SHAP values for the current fold
    print(f"Visualizing SHAP values for fold {fold+1}")
    shap.summary_plot(shap_values, X_test)
    shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test[0])
    plt.show()

# If you want to summarize SHAP values for all folds
# Aggregate SHAP values across folds (mean, median, etc.)
aggregated_shap_values = np.mean(np.array(shap_values_all_folds), axis=0)  # Example: mean of all folds

# Visualize aggregated SHAP values across all folds
shap.summary_plot(aggregated_shap_values, X_data)
