# %%
import os
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.io import loadmat
from torch.utils.data import Subset


def sequential_split(dataset, lengths):
    """
    sequentially splits a dataset into non-overlapping new datasets
    :param dataset: input dataset which is a torch.utils.data.Dataset
    :param lengths: lengths of splits to be produced, should sum to the length of the dataset
    :return: a list of torch.utils.data.Subset
    """
    # ensure the lengths sum up to the total length of the dataset
    assert sum(lengths) == len(
        dataset
    ), "Sum of input lengths does not equal the total length of the dataset"

    # generate split points
    indices = torch.arange(0, len(dataset))
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(torch.cumsum(torch.tensor(lengths), 0), lengths)
    ]

class HumanDataset(Dataset):
    """Dataset for neural recordings with variable sequence lengths and masking"""
    
    def __init__(
        self,
        datapath: str,
        split: str = "train",
        max_seqlen: int = 512,
        time_last: bool = True,
    ):
        """Initialize dataset

        Args:
            datapath: path to data directory
            split: data split ('train' or 'test')
            max_seqlen: maximum sequence length for padding
            time_last: if true, return tensors with time as last dimension
        """
        super().__init__()
        self.datapath = datapath
        self.time_last = time_last
        self.max_seqlen = max_seqlen

        # load data
        data = self._load_data(split)
        
        self.spikes, self.masks = self._process_spikes(data["spikes"])
        
        # convert to torch tensors
        self.spikes = torch.from_numpy(np.array(self.spikes)).float()
        self.masks = torch.from_numpy(np.array(self.masks)).float()

    def _load_data(self, split: str) -> Dict:
        """Load raw data from files"""
        filenames = os.listdir(os.path.join(self.datapath, split))
        all_spikes = []

        for f in tqdm(filenames, desc=f"Loading {split} data"):
            if not f.endswith('.mat'):  # skip non-mat files
                continue
            matfile = loadmat(os.path.join(self.datapath, split, f))
            spikes = matfile["tx1"][0]  # extract spikes from mat file structure
            
            for spike_array in spikes:  # handle multiple trials in each file
                if len(spike_array) <= self.max_seqlen:
                    all_spikes.append(spike_array)

        return {"spikes": all_spikes}

    def _process_spikes(self, spikes: list) -> Tuple[np.ndarray, np.ndarray]:
        """Process spikes and create masks for variable length sequences
        
        Args:
            spikes: list of spike arrays with variable lengths
            
        Returns:
            processed_spikes: padded spike arrays
            masks: binary masks indicating valid timesteps
        """
        processed_spikes = []
        masks = []
        
        for spike in spikes:
            spike = spike[:, :128]  # limit to first 128 channels, as in the original paper
            mask = np.ones((self.max_seqlen, spike.shape[1]), dtype=np.float32)
            
            if len(spike) < self.max_seqlen:
                padding_length = self.max_seqlen - len(spike)
                mask[len(spike):] *= 0
                spike = np.pad(spike, ((0, padding_length), (0, 0)), mode="constant")
                
            processed_spikes.append(spike)
            masks.append(mask)
            
        return processed_spikes, masks

    def __len__(self) -> int:
        return len(self.spikes)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset

        Args:
            idx: index of sample to get

        Returns:
            dictionary containing:
                signal: spike counts [C, T] if time_last else [T, C]
                mask: binary mask [C, T] if time_last else [T, C]
        """
        if self.time_last:
            return {
                "signal": self.spikes[idx].T,
                "mask": self.masks[idx].T,
            }
        return {
            "signal": self.spikes[idx],
            "mask": self.masks[idx],
        }


def get_masked_neural_dataloaders(
    datapath: str,
    batch_size: int = 32,
    max_seqlen: int = 512,
    num_workers: int = 4,
    time_last: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for masked neural data

    Args:
        datapath: path to data directory
        batch_size: batch size for dataloaders
        max_seqlen: maximum sequence length for padding
        num_workers: number of workers for dataloaders
        time_last: if true, return tensors with time as last dimension

    Returns:
        train, validation and test dataloaders
    """
    # create datasets
    train_dataset = HumanDataset(
        datapath=datapath,
        split="train",
        max_seqlen=max_seqlen,
        time_last=time_last,
    )
    
    val_dataset = HumanDataset(
        datapath=datapath,
        split="test",
        max_seqlen=max_seqlen,
        time_last=time_last,
    )

    # split validation into val and test
    val_dataset, test_dataset = sequential_split(
        val_dataset, 
        [len(val_dataset) // 4, len(val_dataset) - len(val_dataset) // 4]
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

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# %%
if __name__ == "__main__":

    # run from root directory of repo
    # os.chdir("../../")
    # if dataset is not present, download and untar

    # test dataset and dataloaders
    datapath = "data/human/competitionData"
    
    # test dataset creation
    dataset = HumanDataset(datapath=datapath)
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shapes:")
    print(f"Signal: {sample['signal'].shape}")
    print(f"Mask: {sample['mask'].shape}")
    # visualize sample
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
    
    im1 = ax1.imshow(sample["signal"], aspect="auto")
    plt.colorbar(im1, ax=ax1)
    ax1.set_title("spike counts")
    
    im2 = ax2.imshow(sample["mask"], aspect="auto")
    plt.colorbar(im2, ax=ax2)
    ax2.set_title("mask")
    
    plt.tight_layout()
    plt.show()

    # test dataloader creation
    train_loader, val_loader, test_loader = get_masked_neural_dataloaders(
        datapath=datapath,
    )

    # print shapes from each loader
    for loader, name in [(train_loader, "Train"), 
                        (val_loader, "Valid"), 
                        (test_loader, "Test")]:
        batch = next(iter(loader))
        print(f"\n{name} loader shapes:")
        print(f"Signal: {batch['signal'].shape}")
        print(f"Mask: {batch['mask'].shape}")

# %%
