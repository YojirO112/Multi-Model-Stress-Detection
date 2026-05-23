from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from backend.utils.emotions_util import get_emotions

# accuracy, loss, f1, confusion matrix
def evaluate_model(history, model, test_gen, y_true):
    print('Evaluating model...')

    # loss curve
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # accuracy curve
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    print("Evaluating model on test set ....")

    # Predictions
    y_pred_probs = model.predict(test_gen)

    # Convert probabilities to class index
    y_pred = np.argmax(y_pred_probs, axis=1)

    # F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nWeighted F1 Score: {f1:.4f}")

    # Classification Report
    print("\nClassification Report:\n")

    emotion_labels = get_emotions()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names = emotion_labels
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.show()