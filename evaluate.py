# Evaluate Model
# Description: This script evaluates saved weights of the model using an aggregate approach.
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


# Load Data
data = np.load("subject_data.npz")
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

# Parameters
n_splits = 5
n_classes = y.shape[-1]
batch_size = 16
samples, chans = X.shape[2], X.shape[1]

# Initialize accumulators
conf_matrix_accum = np.zeros((n_classes, n_classes), dtype=np.float64)
accuracy_per_fold = []
loss_per_fold = []
scores_atc = []

# Initialize misclassification tracking
misclassified_trials_all = defaultdict(list)
misclassification_stats_all = defaultdict(int)


print("\nStarting Evaluation...")

# Iterate over folds
for fold_number in range(1, n_splits + 1):
    print(f"Processing Fold {fold_number}...")
    
    # Load weights for the fold
    weight_file = f"best_model_fold_{fold_number}.weights.h5"
    if not os.path.exists(weight_file):
        print(f"Warning: Weight file {weight_file} not found. Skipping this fold.")
        continue
    
    # Split indices for the fold
    fold_data = np.load(f"fold_{fold_number}_indices.npz")  # Assumes indices saved during training
    test_index = fold_data['test_indices']
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Standardize test data
    X_test_scaled = np.zeros_like(X_test)
    scaler = StandardScaler()
    for i in range(X_test.shape[1]):
        scaler.fit(X[:, i, :])  # Fit on the entire data for consistency
        X_test_scaled[:, i, :] = scaler.transform(X_test[:, i, :])
    
    # Expand dimensions for compatibility with the model
    X_test_scaled = np.expand_dims(X_test_scaled, 1)
    y_test_categorical = to_categorical(y_test, n_classes)
    
    # Load and compile the model
    model = BFN.proposed(samples, chans, n_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.load_weights("best_model_fold_" + str(fold_number)+ ".weights.h5", skip_mismatch=True)
    
    # Evaluate the model
    scores = model.evaluate(X_test_scaled, y_test_categorical, batch_size=batch_size, verbose=1)
    accuracy_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    
    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test_categorical, axis=1)
    
    # Compute confusion matrix for this fold
    fold_conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=range(n_classes))
    conf_matrix_accum += fold_conf_matrix
    
    # Identify misclassified trials
    misclassified_indices = np.where(y_pred_classes != y_test_classes)[0]
    test_subjects = subject_ids[test_index]  # Map test indices to subject IDs
    
    for idx in misclassified_indices:
        subject_id = test_subjects[idx]
        misclassified_trials_all[subject_id].append(idx)
        misclassification_stats_all[subject_id] += 1

    print(f"Fold {fold_number} - Accuracy: {scores[1]:.4f}, Loss: {scores[0]:.4f}")

    # ==================================================================================================
    # Clear memory after each fold
    del model  # Delete model to free memory
    K.clear_session()  # Clear Keras session to free up resources
    gc.collect()  # Run garbage collection to clean up any residual memory usage

# ==================================================================================================
# Aggregate Results
print("\nAggregating Results...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Print aggregated misclassification statistics
print("\nAggregated Misclassification Statistics:")
aggregated_df = pd.DataFrame.from_dict(misclassification_stats_all, orient='index', columns=['Misclassified Trials'])
aggregated_df.index.name = 'Subject ID'
print(aggregated_df)

# Save aggregated statistics to CSV
aggregated_csv_file = f"misclassified_trials_aggregated_{timestamp}.csv"
aggregated_df.to_csv(aggregated_csv_file)
print(f"Aggregated misclassification statistics saved to {aggregated_csv_file}.")

# Print overall metrics
print("\nFinal Metrics:")
print(f"Average Accuracy Across Folds: {np.mean(accuracy_per_fold) * 100:.2f}%")
print(f"Average Loss Across Folds: {np.mean(loss_per_fold):.4f}")

# Average confusion matrix over all folds
print("avg")
conf_matrix_avg = conf_matrix_accum / n_splits

# Compute row sums
print("sum row")
row_sums = conf_matrix_avg.sum(axis=1, keepdims=True)

# Avoid division by zero
row_sums[row_sums == 0] = 1  # Replace zeros with ones to prevent division by zero

# Normalize confusion matrix to percentages
conf_matrix_percent = conf_matrix_avg / row_sums * 100

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Average Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')


conf_matrix_file = f"matrix/con_matrix_avg_{timestamp}.png"
plt.savefig(conf_matrix_file)
plt.show()


