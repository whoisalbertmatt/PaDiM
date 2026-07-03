import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

from src.padim.predict_padim import compute_threshold
from src.padim.score_padim import score_image

def evaluate():
    test_root = "data/raw/mvtec/bottle/test"
    good_dir = os.path.join(test_root, "good")

    threshold = compute_threshold(good_dir)

    y_true = []
    y_pred = []
    y_scores = []

    # Loop through every folder
    for folder in os.listdir(test_root):

        folder_path = os.path.join(test_root, folder)

        if not os.path.isdir(folder_path):
            continue

        # Assign label
        label = 0 if folder == "good" else 1

        # Loop through every image
        for image_name in os.listdir(folder_path):

            if not image_name.endswith(".png"):
                continue

            image_path = os.path.join(folder_path, image_name)

            score = score_image(image_path)

            prediction = 1 if score > threshold else 0

            y_true.append(label)
            y_pred.append(prediction)
            y_scores.append(score)

    print()

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("ROC AUC  :", roc_auc_score(y_true, y_scores))

    print()

    print("Confusion Matrix")

    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()