#third party imports
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn.functional as F

#custom imports
from clearshape.constants import PATHS


def encode_sequence(seq):
    return [VOCAB[token] for token in seq]

class VecsetDataset(Dataset):
    def __init__(self, mode = "train", target="class_name", transform=None, target_transform=None, one_hot=True):
        self.cache = {}

        self.target = target
        self.one_hot = one_hot



        self.df_data = pd.read_csv("/home/michelkruse/Nextcloud/Ressourcen-KI-Gruppe/Datensätze/fabwave/4_feature/fabwave_targets_split.csv")
        self.path_to_vecset = Path("/home/michelkruse/Nextcloud/Ressourcen-KI-Gruppe/Datensätze/fabwave/feature/vecset/fabwave/")

        self.train_df = self.df_data[self.df_data["split"] == "train"]
        self.valid_df = self.df_data[self.df_data["split"] == "valid"]
        self.test_df = self.df_data[self.df_data["split"] == "test"]

        self.num_classes = self.df_data["class_id"].nunique()

        # set mode
        if mode == "train":
            self.samples = self.train_df
        elif mode == "valid":
            self.samples = self.valid_df
        elif mode == "test":
            self.samples = self.test_df


        # optional transformations
        self.transform = transform
        self.target_transform = target_transform

    def parse_part(self, idx):
        """parse_part loads the vecset and plan item
        """
        sample_data = self.samples.iloc[idx]
        sample_path = sample_data["path"]

        # load vecset
        vecset_file = (self.path_to_vecset / sample_path).with_suffix(".npy")

        assert vecset_file.exists(), f"Vecset file {vecset_file} does not exist"

        # load sample

        vecset_item = torch.Tensor(np.load(vecset_file.as_posix(), allow_pickle=True))


        # load plan
        # TODO read from target file 
        target_item = sample_data[self.target]

        if self.one_hot:
            target_item = F.one_hot(torch.tensor(target_item), num_classes=self.num_classes+1).float()

        # apply transformations
        if self.transform:
            vecset_item = self.transform(vecset_item)
        if self.target_transform:
            plan_item = self.target_transform(plan_item)

        return vecset_item, target_item

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """__getitem__ checks if image file is in chache -> self.data. If not the function parse_part is called to apply preprocessing.
        if only 2D or 3D is needed the function will return an empty Tensor for the other item, but al
        """

        vecset_item = None
        target_item = None

        if idx in self.cache.keys():
            vecset_item, plan_item = self.cache[idx]
        else:
            vecset_item, plan_item = self.parse_part(idx)
            self.cache[idx] = (vecset_item, plan_item)
        
        return vecset_item, plan_item


class VecsetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, target="class_id"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.target = target
        
    def setup(self, stage=None):
        # Datasets werden einmal pro Prozess geladen
        if stage == "fit" or stage is None:
            self.train_dataset = VecsetDataset(mode="train", target=self.target)
            self.val_dataset = VecsetDataset(mode="valid", target=self.target)

        if stage == "test" or stage is None:
            self.test_dataset = VecsetDataset(mode="test", target=self.target)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

#checks
if __name__ == "__main__":
    # Example usage
    vecset_data_module = VecsetDataModule(batch_size=4000, num_workers=4)
    vecset_data_module.setup(stage="fit")
    
    train_loader = vecset_data_module.train_dataloader()
    validation_loader = vecset_data_module.val_dataloader()

    train_batch = next(iter(train_loader))
    validation_batch = next(iter(validation_loader))

    print("Train batch shape:", train_batch[0].shape, train_batch[1].shape)
    print("Validation batch shape:", validation_batch[0].shape, validation_batch[1].shape)

    print("traget_cls:", train_batch[1])

