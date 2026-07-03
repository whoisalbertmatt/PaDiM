import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.mvtec_dataset import MVTecDataset
from utils.transforms import get_transforms
from models.autoencoder import ConvAutoencoder


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: TRAIN ONLY ON GOOD IMAGES
    dataset = MVTecDataset(
        root_dir="data/raw/mvtec/bottle/train",
        transform=get_transforms()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ConvAutoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, _ in loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "src/models/saved/autoencoder.pth")


if __name__ == "__main__":
    train()