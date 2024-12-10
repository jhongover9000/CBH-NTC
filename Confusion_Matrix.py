# Script to Compute Aggregated Confusion Matrix

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from EEGModels import EEGNet
import tensorflow as tf

# Set parameters
n_splits = 5  # Number of folds
n_classes = 2  # Number of classes
weights_dir = "./weights/"  # Directory where fold weights are saved
data_file = "subject_data.npz"  # Data file
timestamp = "final"  # Timestamp for saved results]
set_dir = "./EEGNET_V1/"

# Load data
data = np.load(data_file)
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, Subject IDs: {subject_ids.shape}")

# Initialize confusion matrix accumulator
conf_matrix_accum = np.zeros((n_classes, n_classes))

# Loop through folds and evaluate
for fold in range(1, n_splits + 1):
    print(f"Processing Fold {fold}...")
    
    # Load fold indices
    fold_indices = np.load(f"{set_dir}fold_{fold}_indices.npz")
    train_index = fold_indices['train_index']
    test_index = fold_indices['test_index']
    
    # Prepare data
    X_test = X[test_index]
    y_test = y[test_index]

    print(X_test.shape)
    print(y_test.shape)
    
    # Expand dimensions for model compatibility
    X_test = np.expand_dims(X_test, 1)
    X_test = tf.transpose(X_test, perm=[0, 2, 3, 1])  # Transpose to match model input shape
    y_test = to_categorical(y_test, n_classes)
    
    # Load model and weights
    samples, chans = X_test.shape[2], X_test.shape[1]
    model = EEGNet(n_classes, chans, Samples=samples)
    model.load_weights(os.path.join(f"{set_dir}best_model_fold_{fold}.weights.h5"))
    
    # Predict on the test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Compute fold confusion matrix
    fold_conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=range(n_classes))
    conf_matrix_accum += fold_conf_matrix
    print(f"Fold {fold} Confusion Matrix:\n{fold_conf_matrix}")
    
# Compute average confusion matrix
conf_matrix_avg = conf_matrix_accum / n_splits
print(f"Average Confusion Matrix:\n{conf_matrix_avg}")

# Normalize confusion matrix to percentages
row_sums = conf_matrix_avg.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # Avoid division by zero
conf_matrix_percent = conf_matrix_avg / row_sums * 100
print(f"Normalized Confusion Matrix (Percent):\n{conf_matrix_percent}")

# Plot and save averaged confusion matrix
os.makedirs("matrix", exist_ok=True)  # Ensure directory exists
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Average Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f"matrix/con_matrix_avg_{timestamp}.png")
plt.show()
