import torch
from torch.utils.data import DataLoader
from datasets.mvtec_dataset import MVTecDataset
from utils.transforms import get_transforms
from models.resnet import DefectResNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MVTecDataset(
        root_dir="data/raw/mvtec/bottle/train",
        transform=get_transforms()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DefectResNet().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

    torch.save(model.state_dict(), "src/models/saved/model.pth")

if __name__ == "__main__":
    train()