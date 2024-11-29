
# ==================================================================================================
# ==================================================================================================
# Label Subjects
import numpy as np
from mne.io import read_raw_eeglab
from mne import Epochs, find_events
from sklearn.model_selection import train_test_split

# Split subjects by KMI vs VMI
sub_kmi = []
sub_vmi = []
sub_etc = []

files_kmi = []
files_vmi = []

for n in sub_kmi:
    files_kmi.append('')

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


# ==================================================================================================
# ==================================================================================================
# Set up ATCNet Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Dropout, BatchNormalization, Activation, Multiply
from tensorflow.keras.optimizers import Adam

def AttentionBlock(input_tensor):
    """Applies an attention mechanism over temporal features."""
    attention = Dense(input_tensor.shape[-1], activation='softmax')(input_tensor)
    return Multiply()([input_tensor, attention])

def ATCNet(input_shape, nb_classes, dropout_rate=0.5, num_filters=16, kernel_size=64):
    inputs = Input(shape=input_shape)

    # Temporal convolutional block
    x = Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Attention mechanism
    x = AttentionBlock(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# Define the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (n_channels, n_times)
nb_classes = len(np.unique(y_train))

atcnet_model = ATCNet(input_shape=input_shape, nb_classes=nb_classes)

# Compile the model
atcnet_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])


# ==================================================================================================
# ==================================================================================================
# Train Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_atcnet_model.h5', monitor='val_loss', save_best_only=True)
]

# Train the model
history = atcnet_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)


# ==================================================================================================
# ==================================================================================================
# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Test evaluation
test_loss, test_accuracy = atcnet_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on test set
y_pred = atcnet_model.predict(X_test)
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