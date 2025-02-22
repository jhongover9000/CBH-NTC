# Training Pipeline
# Joseph Hong
# Desc: training pipeline using saved dataset.
# ==================================================================================================
# ==================================================================================================
# Load data and labels
data = np.load("subject_data.npz")
X = data['X']
y = data['y']
subject_ids = data['subject_ids']
print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}, subject IDs: {subject_ids.shape}")

# ==================================================================================================
# ==================================================================================================
# Train Model with 5 fold validation
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import keras
import numpy as np
import BFN
import matplotlib.pyplot as plt
import shap
from datetime import datetime
import gc
from tensorflow.keras.utils import to_categorical

print("Starting")

# X (EEG data) and y (labels) are already prepared
# X.shape: (n_samples, n_channels, n_times), y.shape: (n_samples)
n_splits = 5  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to track metrics for each fold
accuracy_per_fold = []
loss_per_fold = []

fold_number = 1

lr = 0.00005
w_decay = 0.01
samples = 400
epochs = 90

nSub = 31  # number of subjects
bs_t = 16  # batch size

scores_atc = []
scores_dcn = []
scores_soft = []
nb_classes = 2
chans = 56


confx = np.zeros((nSub, nb_classes, nb_classes))

shap_values_all = []
y_test_all = []
y_pred_all = []

history_list=[]

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()

# debug
print('Comment: Test')
model_test = BFN.proposed(samples, chans, nb_classes)
print(model_test.summary())
print(len(model_test.layers))

# Initialize StandardScaler
scaler = StandardScaler()

# Normalize across trials, for each channel and time point, apply scaling for each trial
def standardize_epochs_training(X):
    # Apply scaling to each channel independently across trials and time
    n_trials, n_channels, n_times = X.shape
    X_scaled = np.zeros_like(X)
    
    for i in range(n_channels):  # Iterate over each channel
        # Fit and transform the data for this channel (normalize across trials and times)
        X_scaled[:, i, :] = scaler.transform(X[:, i, :])
    
    return X_scaled

def standardize_epochs_test(X):
    # Apply scaling to each channel independently across trials and time
    n_trials, n_channels, n_times = X.shape
    X_scaled = np.zeros_like(X)
    
    for i in range(n_channels):  # Iterate over each channel
        # Fit and transform the data for this channel (normalize across trials and times)
        X_scaled[:, i, :] = scaler.fit_transform(X[:, i, :])
    
    return X_scaled


for train_index, test_index in skf.split(X, y):

    # Split the data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]

    # Apply standardization to the training and testing data
    X_train = standardize_epochs_training(X_train)
    X_test = standardize_epochs_test(X_test)

    # expand dimension to match input type, (n_trials, 1, n_channels, n_timepoints)
    X_train = np.expand_dims(X_train,1)
    X_test = np.expand_dims(X_test,1)

    print(np.shape(X_train))

    y_train, y_test = y[train_index], y[test_index]

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = BFN.proposed(samples, chans, nb_classes)
    model.load_weights('./pretrained_VR.h5', by_name = True, skip_mismatch = True)

    opt_atc = keras.optimizers.Adam(learning_rate=lr ,weight_decay=w_decay)

    # Compile the model
    model.compile(optimizer=opt_atc, 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

    print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())


    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00005),
        ModelCheckpoint(f'best_model_fold_{fold_number}.weights.h5', monitor='val_loss', save_weights_only=True)
    ]

    history = model.fit(X_train, y_train, validation_data = (X_test,y_test), batch_size=bs_t, epochs=epochs, callbacks = callbacks, class_weight=class_weight_dict, verbose=1)

    # Evaluate the model on the test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(f"Fold {fold_number} - Loss: {scores[0]}, Accuracy: {scores[1]}")

    # Track metrics
    loss_per_fold.append(scores[0])
    accuracy_per_fold.append(scores[1])

    # Increment fold number
    fold_number += 1

    probs_atc = model.predict(X_test)
    preds_atc = probs_atc.argmax(axis=-1)
    acc_atc = np.mean(preds_atc == y_test.argmax(axis=-1))
    print(f'ATC:{acc_atc} %')
    history_list.append(history)

    scores_atc.append(acc_atc)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')



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

# Normalize by row (true labels)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=['KMI', 'VMI'], yticklabels=['KMI', 'VMI'])
plt.title('Confusion Matrix (Percentages)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the figure
fig_file = f"matrix/con_matrix_{timestamp}.png"
plt.savefig(fig_file)

# Show the plot
plt.show()

print(accuracy_per_fold)


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
    fig_file = f"curves/History_{timestamp}.png"
    plt.savefig(fig_file)
    plt.show()

plot_training_history(history)

# #plot accuracy history
# train_acc=np.zeros((epochs,1))
# val_acc=np.zeros((epochs,1))
# for sub in range(n_splits):
#     train_acc=train_acc.flatten() + np.array(history_list[sub].history['accuracy']).flatten()
#     val_acc=val_acc.flatten()+ np.array(history_list[sub].history['val_accuracy']).flatten()
# train_acc=train_acc/(n_splits)
# val_acc= val_acc/(n_splits)
# BFN.plot_history(train_acc,val_acc,timestamp)

# print(f'Avg Accuracy ATC:{np.mean(scores_atc)} %')
# print(f'All Accuracy ATC:{scores_atc} ')


