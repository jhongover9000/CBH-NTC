# ==================================================================================================
# ==================================================================================================
# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



#plot accuracy history
# train_acc=np.zeros((epochs,1))
# val_acc=np.zeros((epochs,1))
# for sub in range(nSub-4):
#     train_acc=train_acc.flatten() + np.array(history_list[sub].history['accuracy']).flatten()
#     val_acc=val_acc.flatten()+ np.array(history_list[sub].history['val_accuracy']).flatten()
# train_acc=train_acc/(nSub-4)
# val_acc= val_acc/(nSub-4)
# BFN.plot_history(train_acc,val_acc,timestamp)


# print(f'Avg Accuracy ATC:{np.mean(scores_atc)} %')
# print(f'All Accuracy ATC:{scores_atc} ')

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