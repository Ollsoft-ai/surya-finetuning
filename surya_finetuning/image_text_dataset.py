import torch
import os
import csv
from PIL import Image

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, samples_dir: str):
        super().__init__()

        self._data = []

        with open(os.path.join(samples_dir, "text", "labels.csv")) as label_file:
            label_csv = csv.reader(label_file)
            for row in label_csv:
                img_name = row[0]
                label = row[1]
                image = Image.open(os.path.join(samples_dir, "images", img_name)).convert("RGB")
                self._data.append((image, label))
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]



