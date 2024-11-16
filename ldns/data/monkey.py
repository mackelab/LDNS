# %%

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors
import pickle

from ldns.utils.utils import set_seed


class MonkeyDataset(Dataset):
    """Dataset for monkey reaching data with neural recordings"""
    
    def __init__(
        self,
        task: str,
        datapath: str,
        split: str = "train",
        bin_width: int = 5,
        time_last: bool = True,
        new_data: bool = False,
    ):
        """Initialize dataset

        Args:
            task: task name (e.g. 'mc_maze')
            datapath: path to data directory
            split: data split ('train' or 'val')
            bin_width: bin width for neural data in ms
            time_last: if true, return tensors with time as last dimension
            new_data: if true, reprocess data from raw files
        """
        super().__init__()
        self.task = task
        self.datapath = datapath
        self.time_last = time_last
        self.split = split

        # load or create preprocessed data
        data_dict = self._load_or_create_data(bin_width, new_data)
        
        # process neural data
        spikes_heldin = data_dict["train_spikes_heldin"]
        spikes_heldout = data_dict["train_spikes_heldout"]
        self.spikes = np.concatenate([spikes_heldin, spikes_heldout], axis=-1)
        self.behavior = data_dict["train_behavior"]

        # handle validation split differently
        if split == "val":
            self.spikes = self.spikes[len(self.spikes)//2:]
            self.behavior = self.behavior[len(self.behavior)//2:]
        elif split == "train":
            # for training, include first half of validation data
            val_spikes_heldin = data_dict["train_spikes_heldin"]
            val_spikes_heldout = data_dict["train_spikes_heldout"]
            val_spikes = np.concatenate([val_spikes_heldin, val_spikes_heldout], axis=-1)
            val_behavior = data_dict["train_behavior"]
            
            # use only first half of validation data
            val_spikes = val_spikes[:len(val_spikes)//2]
            val_behavior = val_behavior[:len(val_behavior)//2]
            
            # concatenate with training data
            self.spikes = np.concatenate([self.spikes, val_spikes], axis=0)
            self.behavior = np.concatenate([self.behavior, val_behavior], axis=0)

        # convert to torch tensors
        self.spikes = torch.from_numpy(self.spikes).float()
        self.behavior = torch.from_numpy(self.behavior).float()

    def _load_or_create_data(self, bin_width: int, new_data: bool) -> dict:
        """load preprocessed data or create from raw files"""
        cache_file = os.path.join(
            self.datapath, 
            f"monkey_{self.task}_data_dict_{self.split}_split_{bin_width}.pkl"
        )
        
        if not new_data and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        # create new preprocessed data
        print(f"Creating new preprocessed data from dandi archive at {self.datapath}...")
        dataset = NWBDataset(self.datapath)
        dataset.resample(bin_width)
        
        data_dict = make_train_input_tensors(
            dataset,
            dataset_name=self.task,
            trial_split=self.split,
            save_file=False,
            include_behavior=True,
        )
        
        # save preprocessed data
        with open(cache_file, "wb") as f:
            pickle.dump(data_dict, f)
            
        return data_dict

    def __len__(self) -> int:
        return len(self.spikes)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset

        Args:
            idx: index of sample to get

        returns:
            dictionary containing:
                signal: spike counts [C, T] if time_last else [T, C]
                behavior: behavioral variables [2, T] if time_last else [T, 2]
        """
        if self.time_last:
            return {
                "signal": self.spikes[idx].T,
                "behavior": self.behavior[idx].T,
            }
        return {
            "signal": self.spikes[idx],
            "behavior": self.behavior[idx],
        }


def get_monkey_dataloaders(
    task: str,
    datapath: str,
    bin_width: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    new_data: bool = False,
    time_last: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for monkey data

    Caution: since the val and test sets are created by splitting the validation set,
    to get the dataset from them use dataloader.dataset.dataset.
    (e.g. val_dataset = val_dataloader.dataset.dataset)

    Args:
        task: task name
        datapath: path to data directory
        bin_width: bin width for neural data in ms
        batch_size: batch size for dataloaders
        num_workers: number of workers for dataloaders
        new_data: if true, reprocess data from raw files
        time_last: if true, return tensors with time as last dimension

    Returns:
        train, validation and test dataloaders
    """
    # create datasets
    train_dataset = MonkeyDataset(
        task=task,
        datapath=datapath,
        split="train",
        bin_width=bin_width,
        time_last=time_last,
        new_data=new_data,
    )
    
    val_dataset = MonkeyDataset(
        task=task,
        datapath=datapath,
        split="val",
        bin_width=bin_width,
        time_last=time_last,
        new_data=new_data,
    )

    # split validation into val and test
    val_size = len(val_dataset)
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset,
        [val_size // 4, val_size - val_size // 4],
        generator=torch.Generator().manual_seed(42),
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Task: {task}, Bin width: {bin_width} ms")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

# %%
if __name__ == "__main__":

    # run from root directory of repo
    # if dataset is not present, download it
    # dandi download DANDI:000128/0.220113.0400

    # test dataset and dataloaders
    task = "mc_maze"
    datapath = "data/000128/sub-Jenkins/"
    
    # test dataset creation
    dataset = MonkeyDataset(task=task, datapath=datapath)
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shapes:")
    print(f"Signal: {sample['signal'].shape}")
    print(f"Behavior: {sample['behavior'].shape}")
    
    # visualize sample
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 4))
    plt.matshow(sample["signal"], aspect="auto")
    plt.colorbar()
    plt.title("Spike counts")
    plt.show()

    # test dataloader creation
    train_loader, val_loader, test_loader = get_monkey_dataloaders(
        task=task,
        datapath=datapath,
    )

    # print shapes from each loader
    for loader, name in [(train_loader, "Train"), 
                        (val_loader, "Valid"), 
                        (test_loader, "Test")]:
        batch = next(iter(loader))
        print(f"\n{name} loader shapes:")
        print(f"Signal: {batch['signal'].shape}")
        print(f"Behavior: {batch['behavior'].shape}")

# %%
