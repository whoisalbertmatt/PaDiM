import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.mvtec_dataset import MVTecDataset
from src.utils.transforms import get_transforms
from src.features.extractor import FeatureExtractor


def train_padim():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MVTecDataset(
        root_dir="data/raw/mvtec/bottle/train",
        transform=get_transforms()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = FeatureExtractor().to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            features = model(images)  # [B, 512, 7, 7]
            features = features.cpu().numpy()

            all_features.append(features)

    features = np.concatenate(all_features, axis=0)  # [N, 512, 7, 7]

    N, C, H, W = features.shape

    # reduce dimensions for stable covariance
    np.random.seed(42)
    selected_dims = np.random.choice(C, size=100, replace=False)

    features = features[:, selected_dims, :, :]  # [N, 100, 7, 7]

    # reshape to [N, H*W, C]
    features = features.transpose(0, 2, 3, 1)
    features = features.reshape(N, H * W, 100)

    means = []
    covs = []

    for position in range(H * W):
        position_features = features[:, position, :]  # [N, 100]

        mean = np.mean(position_features, axis=0)
        cov = np.cov(position_features, rowvar=False)

        cov += 1e-3 * np.eye(cov.shape[0])

        means.append(mean)
        covs.append(cov)

    means = np.array(means)  # [49, 100]
    covs = np.array(covs)    # [49, 100, 100]

    os.makedirs("models/saved", exist_ok=True)

    np.save("models/saved/padim_means_spatial.npy", means)
    np.save("models/saved/padim_covs_spatial.npy", covs)
    np.save("models/saved/padim_selected_dims.npy", selected_dims)

    print("Feature map shape:", (C, H, W))
    print("Selected dims shape:", selected_dims.shape)
    print("Means shape:", means.shape)
    print("Covs shape:", covs.shape)


if __name__ == "__main__":
    train_padim()