
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
chan2drop = ["T7","T8",'FT7','FT8']

for group, files in subject_files.items():
    group_label = group_labels[group]
    for file in files:
        epochs = read_epochs_eeglab(file)
        # drop additional channels to fit BFN
        epochs = epochs.drop_channels(chan2drop)
        print("Channels after dropping:", epochs.info['ch_names'])
        # downsample to 100 Hz (400 timepoints for 4 seconds)
        epochs = epochs.resample(100, verbose = True)
        # crop from -1 to 3
        epochs = epochs.crop(-1,3,False,False)
        

        # Append data and labels
        print(len(epochs))
        all_data.append(epochs.get_data())  # Shape: (n_epochs, n_channels, n_times)
        all_labels.append(np.full(len(epochs), group_label))

# Combine all subjects' data
X = np.concatenate(all_data, axis=0)
y = np.concatenate(all_labels, axis=0)

print(X.shape)
print(y.shape)

print("Done Preprocessing Subjects.")

# ==================================================================================================
# ==================================================================================================
# Train Model with 5 fold validation
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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

nSub = 31  # number of subjects
bs_t = 16  # batch size
epochs = 30
lr = 0.00005
scores_atc = []
scores_dcn = []
scores_soft = []
nb_classes = 2
chans = 56
samples = 400
w_decay = 0.01
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


for train_index, test_index in skf.split(X, y):

    # Split the data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    # expand dimension to match input type, (n_trials, 1, n_channels, n_timepoints)
    X_train = np.expand_dims(X_train,1)
    X_test = np.expand_dims(X_test,1)

    # Normalize across trials
    X_train = StandardScaler().fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    X_test = StandardScaler().fit_transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    print(np.shape(X_train))

    y_train, y_test = y[train_index], y[test_index]

    model = BFN.proposed(samples, chans, nb_classes)
    model.load_weights('./pretrained_VR.h5', by_name = True, skip_mismatch = True)

    opt_atc = keras.optimizers.Adam(learning_rate=lr ,weight_decay=w_decay)

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

    history_atc = model.fit(X_train, y_train, validation_data = (X_test,y_test), batch_size=bs_t, epochs=epochs, callbacks = callbacks verbose=1)
    probs_atc = model.predict(X_test)
    preds_atc = probs_atc.argmax(axis=-1)
    acc_atc = np.mean(preds_atc == y_test.argmax(axis=-1))
    print(f'ATC:{acc_atc} %')
    history_list.append(history_atc)

    scores_atc.append(acc_atc)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

#plot accuracy history
train_acc=np.zeros((epochs,1))
val_acc=np.zeros((epochs,1))
for sub in range(nSub-4):
    train_acc=train_acc.flatten() + np.array(history_list[sub].history['accuracy']).flatten()
    val_acc=val_acc.flatten()+ np.array(history_list[sub].history['val_accuracy']).flatten()
train_acc=train_acc/(nSub-4)
val_acc= val_acc/(nSub-4)
BFN.plot_history(train_acc,val_acc)


print(f'Avg Accuracy ATC:{np.mean(scores_atc)} %')
print(f'All Accuracy ATC:{scores_atc} ')
