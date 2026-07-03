import os
import numpy as np

from src.padim.score_padim import score_image


def compute_threshold(good_dir, percentile=95):
    scores = []

    for img_name in os.listdir(good_dir):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(good_dir, img_name)
        score = score_image(img_path)
        scores.append(score)

    threshold = np.percentile(scores, percentile)

    print("Good scores count:", len(scores))
    print("Mean good score:", np.mean(scores))
    print("Max good score:", np.max(scores))
    print("Threshold:", threshold)

    return threshold


def predict(image_path, threshold):
    score = score_image(image_path)

    if score > threshold:
        prediction = "ANOMALY"
    else:
        prediction = "NORMAL"

    print("Image:", image_path)
    print("Score:", score)
    print("Prediction:", prediction)

    return prediction, score


if __name__ == "__main__":
    good_dir = "data/raw/mvtec/bottle/test/good"

    threshold = compute_threshold(good_dir)

    test_good = "data/raw/mvtec/bottle/test/good/000.png"
    test_defect = "data/raw/mvtec/bottle/test/broken_large/000.png"

    print("\nTesting good image:")
    predict(test_good, threshold)

    print("\nTesting defect image:")
    predict(test_defect, threshold)