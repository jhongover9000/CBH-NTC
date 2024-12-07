
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
import BFN

set_dir = './epoched/'

print("Preprocessing Subjects...")

# Split subjects by KMI vs VMI
sub_kmi = [1,2,3,5,6,12,13,14,15,21,22,23,26,27,28,30,31,33]
sub_vmi = [4,7,8,9,10,11,16,17,19,24,25,29,32]
sub_etc = [18,20]

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

for group, files in subject_files.items():
    group_label = group_labels[group]
    for file in files:
        epochs = read_epochs_eeglab(file)
        # downsample to 100 Hz (400 timepoints for 4 seconds)
        epochs = epochs.resample(100, verbose = True)
        # crop from -1 to 3
        epochs = epochs.crop(-1,3,True,False)

        # Append data and labels
        print(len(epochs))
        all_data.append(epochs.get_data())  # Shape: (n_epochs, n_channels, n_times)
        all_labels.append(np.full(len(epochs), group_label))

# Combine all subjects' data
X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

print("Done Preprocessing Subjects.")



# ==================================================================================================
# ==================================================================================================
# Train Model with 5 fold validation
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import keras
import numpy as np

print("Starting")

# Assume X (EEG data) and y (labels) are already prepared
# X.shape: (n_samples, n_channels, n_times), y.shape: (n_samples,)
n_splits = 5  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to track metrics for each fold
accuracy_per_fold = []
loss_per_fold = []

fold_number = 1

lr = 0.00005
w_decay = 0.01

for train_index, test_index in skf.split(X, y):
    print(f"Processing Fold {fold_number}...")
    
    # Split the data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Normalize across trials
    X_train = StandardScaler().fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test = StandardScaler().fit_transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    # Define the ATCNet model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (n_channels, n_times)
    nb_classes = len(np.unique(y))
    model = atc.ATCNet(input_shape=input_shape, nb_classes=nb_classes)
    opt_atc = keras.optimizers.Adam(learning_rate = lr)
    # Compile the model
    model.compile(optimizer=opt_atc, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())
    

    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f'best_model_fold_{fold_number}.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model on the test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(f"Fold {fold_number} - Loss: {scores[0]}, Accuracy: {scores[1]}")

    # Track metrics
    loss_per_fold.append(scores[0])
    accuracy_per_fold.append(scores[1])

    # Increment fold number
    fold_number += 1

# Display average metrics across all folds
print('Average metrics across all folds:')
print(f"Average Accuracy: {np.mean(accuracy_per_fold) * 100:.2f}%")
print(f"Average Loss: {np.mean(loss_per_fold):.4f}")



# ==================================================================================================
# ==================================================================================================
# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Test evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Group 1', 'Group 2']))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Group 1', 'Group 2'], yticklabels=['Group 1', 'Group 2'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


def plot_training_history(history):
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
    plt.show()

plot_training_history(history)

