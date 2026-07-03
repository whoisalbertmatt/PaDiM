from torch.utils.data import DataLoader
from datasets.mvtec_dataset import MVTecDataset
from utils.transforms import get_transforms

def test_dataloader():
    dataset = MVTecDataset(
        root_dir="data/raw/mvtec/bottle/train",
        transform=get_transforms()
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print("Dataset size:", len(dataset))
    print("Number of batches:", len(loader))

    images, labels = next(iter(loader))

    print("Batch image shape:", images.shape)
    print("Batch labels shape:", labels.shape)

if __name__ == "__main__":
    test_dataloader()