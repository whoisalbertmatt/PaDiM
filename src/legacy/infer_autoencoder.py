import torch
import numpy as np
from PIL import Image

from models.autoencoder import ConvAutoencoder
from utils.transforms import get_transforms


def compute_anomaly_score(model, image_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(image_tensor.unsqueeze(0))

        error = torch.mean((recon - image_tensor.unsqueeze(0)) ** 2)

    return error.item()


def test_sample(image_path, model_path="src/models/saved/autoencoder.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    transform = get_transforms()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).to(device)

    score = compute_anomaly_score(model, image)

    print("Anomaly Score:", score)


if __name__ == "__main__":
    # test on a single image
    test_sample("data/raw/mvtec/bottle/test/good/000.png")