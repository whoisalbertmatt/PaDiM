import os
import cv2
import numpy as np
import torch
from PIL import Image

from src.features.extractor import FeatureExtractor
from src.utils.transforms import get_transforms


def mahalanobis_distance(x, mean, cov):
    diff = x - mean

    cov_inv = np.linalg.pinv(cov)

    value = diff.T @ cov_inv @ diff
    value = np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=0.0)
    value = max(value, 0)

    return np.sqrt(value)


def generate_heatmap(image_path, output_path="outputs/heatmap.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    means = np.load("models/saved/padim_means_spatial.npy")
    covs = np.load("models/saved/padim_covs_spatial.npy")
    selected_dims = np.load("models/saved/padim_selected_dims.npy")

    model = FeatureExtractor().to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor)  # [1, 512, 7, 7]

    features = features.cpu().numpy()[0]          # [512, 7, 7]
    features = features[selected_dims, :, :]      # [100, 7, 7]

    C, H, W = features.shape

    features = features.transpose(1, 2, 0)        # [7, 7, 100]
    features = features.reshape(H * W, C)         # [49, 100]

    scores = []

    for position in range(H * W):
        score = mahalanobis_distance(
            features[position],
            means[position],
            covs[position]
        )
        scores.append(score)

    anomaly_map = np.array(scores).reshape(H, W)  # [7, 7]

    anomaly_map = cv2.resize(anomaly_map, (224, 224))

    anomaly_map = anomaly_map - anomaly_map.min()
    anomaly_map = anomaly_map / (anomaly_map.max() + 1e-8)

    heatmap = np.uint8(255 * anomaly_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)

    print("Saved heatmap to:", output_path)


if __name__ == "__main__":
    image_path = "data/raw/mvtec/bottle/test/broken_large/000.png"
    generate_heatmap(image_path, "outputs/bottle_defect_heatmap.png")