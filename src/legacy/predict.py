import torch
from PIL import Image

from models.autoencoder import ConvAutoencoder
from utils.transforms import get_transforms


def predict(image_path, threshold, model_path="src/models/saved/autoencoder.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_transforms()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).to(device)

    with torch.no_grad():
        recon = model(image.unsqueeze(0))
        score = torch.mean((recon - image.unsqueeze(0)) ** 2).item()

    print(f"Anomaly Score: {score}")

    if score > threshold:
        print("Prediction: ❌ ANOMALY")
    else:
        print("Prediction: ✅ NORMAL")


if __name__ == "__main__":
    THRESHOLD = 1.4214  # from your computed value

    test_image = "data/raw/mvtec/bottle/test/good/000.png"
    predict(test_image, THRESHOLD)