# Training Pipeline
# Author: Joseph Hong
# Description: Training pipeline using a saved dataset. Splitting 5-fold CV into 60-20-20(%) for 
# training, validation, and testing sets respectively.
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
from Models.EEGModels import EEGNet,DeepConvNet,ShallowConvNet

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs available:", gpus)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

weights_dir = "./Weights/"


# Load Data
data = np.load("subject_data_v2.npz")
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

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=55)

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

# Initialize dictionaries to track misclassified trials and counts per subject
misclassified_trials_all = defaultdict(list)  # Aggregated misclassified trials across all folds
misclassification_stats_all = defaultdict(int)  # Aggregated counts across all folds
misclassified_trials_per_fold = []  # List of dictionaries for each fold

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

print("Starting Training...")

for fold_number, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"Processing Fold {fold_number}...")

    # Save the fold indices to a .npz file
    np.savez(f"fold_{fold_number}_indices.npz", train_index=train_index, test_index=test_index)

    # Split the data into train and test sets
    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]

    # Further split X_train_full into training and validation sets (6:2 ratio)
    val_split_index = int(0.8 * len(X_train_full))  # 80% for training
    X_train, X_val = X_train_full[:val_split_index], X_train_full[val_split_index:]
    y_train, y_val = y_train_full[:val_split_index], y_train_full[val_split_index:]

    # Output the shapes to verify
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")

    # Standardize the data
    X_train, X_val = scaler_fit_transform(X_train, X_val)

    # Expand dimensions for compatibility with the model
    X_train = np.expand_dims(X_train,1)
    X_val = np.expand_dims(X_val,1)
    X_test = np.expand_dims(X_test,1)

    X_train = tf.transpose(X_train, perm=[0, 2, 3, 1])  # Swap axes for EEGNet
    X_val = tf.transpose(X_val, perm=[0, 2, 3, 1])
    X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    # Convert labels to categorical
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    # Compute class weights for the training data
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train_full[:val_split_index]),  # Use training indices
        y=y_train_full[:val_split_index]
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"Class weights for fold {fold_number}: {class_weight_dict}")

    # Initialize and compile the model
    model = DeepConvNet(nb_classes, chans, samples)
    model.load_weights(weights_dir + "EEGNet-8-2-weights.h5", by_name=True, skip_mismatch=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f"best_model_fold_{fold_number}.weights.h5",
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    history_list.append(history)
    model.save_weights(f"final_model_fold_{fold_number}.weights.h5")

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    accuracy_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    # Predict on the test set for the current fold
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate accuracy
    acc_atc = np.mean(y_pred_classes == y_test_classes)
    scores_atc.append(acc_atc)

    
    # Shapley Analysis
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  # this solves the "shap_ADDV2" problem but another one will appear
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough  # this solves the next problem which allows you to run the DeepExplainer.
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough  
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    shap.explainers._deep.deep_tf.op_handlers["DepthwiseConv2dNative"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    shap.explainers._deep.deep_tf.op_handlers["BatchToSpaceND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    shap.explainers._deep.deep_tf.op_handlers["SpaceToBatchND"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough #this solves the next problem which allows you to run the DeepExplainer.  
    # Set up 300 random points for shap
    background = np.array(X_train[np.random.choice(X_train.shape[0], 300, replace=False)])


    print(np.shape(X_test))
    # Create DeepExplainer model
    e = shap.DeepExplainer(model, background)
    print(e)
    shap_values = e.shap_values(X_test, check_additivity=False)
    shap_values_all.append(shap_values)
    y_test_all.append(y_test.argmax(axis=-1))
    y_pred_all.append(y_pred)
    print(len(shap_values_all))
    print(len(y_test_all))
    print(len(y_pred_all))

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

    # Compute fold confusion matrix
    fold_conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=range(nb_classes))
    conf_matrix_accum += fold_conf_matrix
    
    print(f"Fold {fold_number} - Accuracy: {scores[1]:.4f}, Loss: {scores[0]:.4f}, ATC: {acc_atc:.4f}, Confusion Matrix:\n{fold_conf_matrix}")

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

# Compute average confusion matrix
conf_matrix_avg = conf_matrix_accum / n_splits
print(f"Average Confusion Matrix:\n{conf_matrix_avg}")

# Normalize confusion matrix to percentages
row_sums = conf_matrix_avg.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # Avoid division by zero
conf_matrix_percent = conf_matrix_avg / row_sums * 100
print(f"Normalized Confusion Matrix (Percent):\n{conf_matrix_percent}")

conf_matrix_percent[conf_matrix_percent == 0] = np.nan

# Plot and save averaged confusion matrix

print(conf_matrix_percent.shape)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['KMI', 'VMI'], yticklabels=['KMI', 'VMI'])
plt.title('Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the figure
fig_file = f"matrix/con_matrix_{timestamp}.png"
plt.savefig(fig_file)

# Show the plot
plt.show()

## SAVING SHAP STUFF

with open("shaps_values_all_full", "wb") as fp:  # Pickling
    pickle.dump(shap_values_all, fp)

with open("y_test_all_full", "wb") as fp:  # Pickling
    pickle.dump(y_test_all, fp)

with open("y_pred_all_full", "wb") as fp:  # Pickling
    pickle.dump(y_pred_all, fp)