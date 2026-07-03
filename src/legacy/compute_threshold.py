import torch
from torch.utils.data import DataLoader

from datasets.mvtec_dataset import MVTecDataset
from utils.transforms import get_transforms
from models.autoencoder import ConvAutoencoder


def get_normal_scores(model, loader, device):
    scores = []

    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            recon = model(images)
            error = torch.mean((recon - images) ** 2, dim=[1,2,3])

            scores.extend(error.cpu().numpy())

    return scores


def compute_threshold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load("src/models/saved/autoencoder.pth", map_location=device))

    dataset = MVTecDataset(
        root_dir="data/raw/mvtec/bottle/train",
        transform=get_transforms()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    scores = get_normal_scores(model, loader, device)

    threshold = sum(scores) / len(scores) + 2 * (max(scores) - min(scores)) / 2

    print("Mean Score:", sum(scores) / len(scores))
    print("Min Score:", min(scores))
    print("Max Score:", max(scores))
    print("Threshold:", threshold)


if __name__ == "__main__":
    compute_threshold()