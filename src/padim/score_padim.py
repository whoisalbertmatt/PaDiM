import numpy as np
import torch
from PIL import Image
from src.features.extractor import FeatureExtractor
from src.utils.transforms import get_transforms


def mahalanobis_distance(x, mean, cov):
    diff = x - mean

    eps = 1e-3
    cov_reg = cov + eps * np.eye(cov.shape[0])

    cov_inv = np.linalg.pinv(cov_reg)

    value = diff.T @ cov_inv @ diff
    value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=0.0)
    value = np.maximum(value, 0)

    return np.sqrt(value)


def score_image(image_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # load statistics
    mean = np.load("src/models/saved/padim_mean.npy")
    cov = np.load("src/models/saved/padim_cov.npy")

    # load model
    model = FeatureExtractor().to(device)
    model.eval()

    # load image
    image = Image.open(image_path).convert("RGB")

    transform = get_transforms()
    image = transform(image).unsqueeze(0).to(device)

    # extract features
    with torch.no_grad():
        feature = model(image)

    feature = feature.cpu().numpy().flatten()

    score = mahalanobis_distance(
        feature,
        mean,
        cov
    )

    return score


if __name__ == "__main__":

    image_path = (
        "data/raw/mvtec/bottle/test/good/000.png"
    )

    score = score_image(image_path)

    print("Anomaly Score:", score)

if __name__ == "__main__":

    good_image = "data/raw/mvtec/bottle/test/good/000.png"

    defect_image = "data/raw/mvtec/bottle/test/broken_large/000.png"
    # If broken_large/000.png does not exist, use any image inside another defect folder.

    good_score = score_image(good_image)
    defect_score = score_image(defect_image)

    print("Good image score:", good_score)
    print("Defect image score:", defect_score)