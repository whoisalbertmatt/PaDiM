import os
from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Walk through ALL folders inside root_dir
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            # Label logic:
            # good = 0, everything else = 1 (anomaly)
            label = 0 if class_name == "good" else 1

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # skip hidden/system files
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label