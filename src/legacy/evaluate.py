import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from datasets.mvtec_dataset import MVTecDataset
from utils.transforms import get_transforms
from models.resnet import DefectResNet

def evaluate(model_path="src/models/saved/model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # IMPORTANT: include both good + defect in test set
    test_dataset = MVTecDataset(
        root_dir="data/raw/mvtec/bottle/test",
        transform=get_transforms()
    )

    loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = DefectResNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()