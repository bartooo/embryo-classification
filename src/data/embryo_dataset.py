import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class EmbryoDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        self.data_df = (
            pd.read_csv(os.path.join(data_dir, "train.csv"))
            if train
            else pd.read_csv(os.path.join(data_dir, "test.csv"))
        )

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        image_name = self.data_df.iloc[idx]["Image"]
        image = Image.open(
            os.path.join(self.data_dir, "train" if self.train else "test", image_name)
        )

        if self.transform:
            image = self.transform(image)

        day = int(image_name.startswith("D3"))

        return (image, self.data_df.iloc[idx]["Class"], day) if self.train else (image, day)
