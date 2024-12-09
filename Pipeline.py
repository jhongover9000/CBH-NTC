# Training Pipeline
# Author: Joseph Hong
# Description: Training pipeline using a saved dataset.
# ==================================================================================================

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
import BFN
from datetime import datetime
import gc
import shap
from shap.explainers._deep import deep_tf
import pickle

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


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
n_classes = y.shape[-1]
conf_matrix_accum = np.zeros((n_classes, n_classes))

# Initialize metrics tracking
accuracy_per_fold = []
loss_per_fold = []
scores_atc = []
history_list = []

shap_values_all = []
y_test_all = []
y_pred_all = []

# Initialize dictionaries to track misclassified trials and counts per subject
misclassified_trials_all = defaultdict(list)  # Aggregated misclassified trials across all folds
misclassification_stats_all = defaultdict(int)  # Aggregated counts across all folds
misclassified_trials_per_fold = []  # List of dictionaries for each fold

print("Starting Training...")

for fold_number, (train_index, test_index) in enumerate(skf.split(X, y), 1):
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
    
    # Convert labels to categorical
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y[train_index]), y=y[train_index])
    class_weight_dict = dict(enumerate(class_weights))
    
    # Initialize and compile the model
    model = BFN.proposed(samples, chans, nb_classes)
    model.load_weights('./pretrained_VR.h5', by_name=True, skip_mismatch=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f"best_model_fold_{fold_number}.weights.h5", monitor='val_loss', save_weights_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    history_list.append(history)
    
    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    
    # Predict and calculate ATC
    y_pred = model.predict(X_test)
    acc_atc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
    scores_atc.append(acc_atc)

    # Predict on the test set for the current fold
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    ##SHAP STUFF
    # Override operations that SHAP is having trouble with
    deep_tf.op_handlers["AddV2"] = deep_tf.passthrough
    deep_tf.op_handlers["FusedBatchNormV3"] = deep_tf.passthrough
    deep_tf.op_handlers["DepthwiseConv2dNative"] = deep_tf.passthrough
    deep_tf.op_handlers["BatchToSpaceND"] = deep_tf.passthrough
    deep_tf.op_handlers["SpaceToBatchND"] = deep_tf.passthrough
    deep_tf.op_handlers["Einsum"] = deep_tf.passthrough

    # Ensure input shape matches model's requirements
    X_train_S = X_train.reshape(-1, 1, chans, samples)
    X_test_S = X_test.reshape(-1, 1, chans, samples)

    # Setup SHAP DeepExplainer with a simpler background sample
    background = tf.convert_to_tensor(X_train_S[np.random.choice(X_train.shape[0], 300, replace=False)], dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test_S, dtype=tf.float32)
    explainer = shap.DeepExplainer(model, background)

    # Explain SHAP values
    print(explainer)
    print(np.shape(X_test))
    shap_values = explainer.shap_values(X_test)
    print("SHAP values computed:", np.shape(shap_values))

    shap_values_all.append(shap_values)

    # Identify misclassified trials
    misclassified_indices = np.where(y_pred_classes != y_test_classes)[0]
    test_subjects = subject_ids[test_index]  # Map test indices to subject IDs

    # Track misclassified trials for the current fold
    misclassified_trials_current_fold = defaultdict(list)
    for idx in misclassified_indices:
        subject_id = test_subjects[idx]
        misclassified_trials_current_fold[subject_id].append(idx)
        misclassified_trials_all[subject_id].append(idx)
        misclassification_stats_all[subject_id] += 1

    misclassified_trials_per_fold.append(misclassified_trials_current_fold)

    # Compute confusion matrix for the current fold
    fold_conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=range(n_classes))
    
    # Accumulate confusion matrices
    conf_matrix_accum += fold_conf_matrix
    
    print(f"Fold {fold_number} - Accuracy: {scores[1]:.4f}, Loss: {scores[0]:.4f}, ATC: {acc_atc:.4f}")

    # ==================================================================================================
    # Clear memory after each fold
    del model  # Delete model to free memory
    K.clear_session()  # Clear Keras session to free up resources
    gc.collect()  # Run garbage collection to clean up any residual memory usage

# ==================================================================================================
# Model Evaluation and Visualization
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f"\nAverage Accuracy Across Folds: {np.mean(accuracy_per_fold) * 100:.2f}%")

# ==================================================================================================
# Shapley Values


with open("shaps_values_all_bfn", "wb") as fp:  # Pickling
    pickle.dump(shap_values_all, fp)

with open("y_test_all_bfn", "wb") as fp:  # Pickling
    pickle.dump(y_test_all, fp)

with open("y_pred_all_bfn", "wb") as fp:  # Pickling
    pickle.dump(y_pred_all, fp)

# ==================================================================================================
# Misclassification Statistics

# Aggregated misclassification statistics across all folds
print("\nAggregated Misclassification Statistics Across All Folds:")
aggregated_df = pd.DataFrame.from_dict(misclassification_stats_all, orient='index', columns=['Misclassified Trials'])
aggregated_df.index.name = 'Subject ID'
print(aggregated_df)

# Save the aggregated statistics to a CSV
aggregated_csv_file = f"misclassified_trials_aggregated_{timestamp}.csv"
aggregated_df.to_csv(aggregated_csv_file)
print(f"Aggregated misclassification statistics saved to {aggregated_csv_file}.")

# Per-fold misclassification statistics
for fold, trials in enumerate(misclassified_trials_per_fold, start=1):
    print(f"\nMisclassification Statistics for Fold {fold}:")
    fold_df = pd.DataFrame.from_dict({k: len(v) for k, v in trials.items()}, 
                                     orient='index', columns=['Misclassified Trials'])
    fold_df.index.name = 'Subject ID'
    print(fold_df)

    # Save the per-fold statistics to a CSV
    fold_csv_file = f"misclassified_trials_fold_{fold}_{timestamp}.csv"
    fold_df.to_csv(fold_csv_file)
    print(f"Fold {fold} misclassification statistics saved to {fold_csv_file}.")

# ==================================================================================================
# Final Aggregated Results

# Plot Training History
for fold, history in enumerate(history_list, 1):
    plot_training_history(history, f"fold_{fold}_{timestamp}")

# Average confusion matrix over all folds
print(conf_matrix_accum)
print(n_splits)
conf_matrix_avg = conf_matrix_accum / n_splits
print(conf_matrix_avg)

# Compute row sums
row_sums = conf_matrix_avg.sum(axis=1, keepdims=True)
print(row_sums)

# Avoid division by zero
row_sums[row_sums == 0] = 1  # Replace zeros with ones to prevent division by zero

# Normalize confusion matrix to percentages
conf_matrix_percent = conf_matrix_avg / row_sums * 100

# Display the averaged confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Average Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the confusion matrix as an image
plt.savefig(f"matrix/con_matrix_avg_{timestamp}.png")
plt.show()


