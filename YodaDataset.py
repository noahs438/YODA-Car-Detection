import os
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch


class YodaDataset(Dataset):
    def __init__(self, labels_path, root_dir, transform=None):
        labels = []
        with open(labels_path, 'r') as file:
            for line in file:
                labels.append(line.strip())

        self.root_dir = root_dir
        self.yoda_labels = labels

        self.transform = transform

    def __len__(self):
        return len(self.yoda_labels)

    def __getitem__(self, idx):
        img_name = self.yoda_labels[idx].split()[0]
        img_path = os.path.join(self.root_dir, img_name)
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Ensure the image is a tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()

        # # Print the size of the image tensor
        # print(f"Image size for index {idx}: {image.size()}")

        label = int(self.yoda_labels[idx].split()[1])

        return image, label
