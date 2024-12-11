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
from einops import rearrange

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
        
        # load train data dict
        train_cache_file = os.path.join(
            datapath, f"monkey_{task}_data_dict_train_split_{bin_width}.pkl"
        )
        
        if not new_data and os.path.exists(train_cache_file):
            data_dict_train = pickle.load(open(train_cache_file, "rb"))
            print(f"Loaded train data dict from {train_cache_file}")
        else:
            # create it and save it
            dataset = NWBDataset(datapath)
            dataset.resample(bin_width)
            data_dict_train = make_train_input_tensors(
                dataset,
                dataset_name=task,
                trial_split="train",
                save_file=False,
                include_behavior=True,
            )
            pickle.dump(data_dict_train, open(train_cache_file, "wb"))
            print(f"Saved train data dict to {train_cache_file}")

        # load val data dict
        val_cache_file = os.path.join(
            datapath, f"monkey_{task}_data_dict_val_split_{bin_width}.pkl"
        )
        
        if not new_data and os.path.exists(val_cache_file):
            data_dict_val = pickle.load(open(val_cache_file, "rb"))
            print(f"Loaded val data dict from {val_cache_file}")
        else:
            # create it and save it
            dataset = NWBDataset(datapath)
            dataset.resample(bin_width)
            data_dict_val = make_train_input_tensors(
                dataset,
                dataset_name=task,
                trial_split="val",
                save_file=False,
                include_behavior=True,
            )
            pickle.dump(data_dict_val, open(val_cache_file, "wb"))
            print(f"Saved val data dict to {val_cache_file}")

        if split == "train":
            # Process train data
            train_spikes_heldin = data_dict_train["train_spikes_heldin"]
            train_spikes_heldout = data_dict_train["train_spikes_heldout"]
            train_spikes = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=-1)
            train_behavior = data_dict_train["train_behavior"]

            # load 50% of val
            train_spikes_heldin = data_dict_val["train_spikes_heldin"]
            train_spikes_heldout = data_dict_val["train_spikes_heldout"]
            train_spikes2 = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=-1)
            
            train_behavior2 = data_dict_val["train_behavior"][:len(train_spikes2)//2]
            train_spikes2 = train_spikes2[:len(train_spikes2)//2]

            self.train_spikes = np.concatenate([train_spikes, train_spikes2], axis=0)
            self.behavior = np.concatenate([train_behavior, train_behavior2], axis=0)

        else:  # val split
            # load rest 50% of val
            train_spikes_heldin = data_dict_val["train_spikes_heldin"]
            train_spikes_heldout = data_dict_val["train_spikes_heldout"]
            train_spikes = np.concatenate([train_spikes_heldin, train_spikes_heldout], axis=-1)
            train_behavior = data_dict_val["train_behavior"]
            
            self.train_spikes = train_spikes[len(train_spikes)//2:]
            self.behavior = train_behavior[len(train_behavior)//2:]

        # convert to torch tensors
        self.train_spikes = torch.from_numpy(self.train_spikes).float()
        self.behavior = torch.from_numpy(self.behavior).float()

    def __len__(self):
        return len(self.train_spikes)

    def __getitem__(self, idx):
        if self.time_last:
            return {
                "signal": self.train_spikes[idx].T,  # [C, L]
                "behavior": self.behavior[idx].T,  # [2, L]
            }
        return {
            "signal": self.train_spikes[idx],  # [L, C]
            "behavior": self.behavior[idx],  # [L, 2]
        }



class LatentMonkeyDataset(torch.utils.data.Dataset):
    """Dataset class for latent representations of monkey neural data.
    
    Takes a dataloader of neural activity and behavior, and uses a trained autoencoder
    to create latent representations. Also computes behavior angles and normalizes data.

    Args:
        dataloader: DataLoader containing neural activity and behavior
        ae_model: Trained autoencoder model for encoding neural activity
        clip: Whether to clip latent values to [-5,5]. Defaults to True.
        latent_means: Optional precomputed means for latent normalization
        latent_stds: Optional precomputed stds for latent normalization
    """
    def __init__(
        self, dataloader, ae_model, clip=True, latent_means=None, latent_stds=None
    ):
        self.full_dataloader = dataloader
        self.ae_model = ae_model
        # get latent representations and original data
        self.latents, self.train_spikes, self.behavior = self.create_latents()
        
        # normalize latents to N(0,1) using provided or computed stats
        if latent_means is None or latent_stds is None:
            self.latent_means = self.latents.mean(dim=(0, 2)).unsqueeze(0).unsqueeze(2)
            self.latent_stds = self.latents.std(dim=(0, 2)).unsqueeze(0).unsqueeze(2)
        else:
            self.latent_means = latent_means
            self.latent_stds = latent_stds
        self.latents = (self.latents - self.latent_means) / self.latent_stds
        
        # optionally clip extreme values
        if clip:
            self.latents = self.latents.clamp(-5, 5)

        # verify data alignment
        assert len(self.latents) == len(self.behavior) and len(self.latents) == len(
            self.train_spikes
        ), f"Lengths of latents, behavior, and spikes do not match: {len(self.latents)}, {len(self.behavior)}, {len(self.train_spikes)}"

        # scale behavior values
        self.behavior = self.behavior / 1e3  # convert to meters

        # compute cumulative behavior trajectory
        self.behavior_cumsum = torch.cumsum(self.behavior, dim=-1)

        # compute reach angles from behavior at 50th timestep
        self.behavior_angles = torch.atan2(
            self.behavior[:, 1, 50], self.behavior[:, 0, 50]
        )
        self.behavior_angles = rearrange(self.behavior_angles, "B -> B 1")

    def create_latents(self):
        """Create latent representations using the autoencoder.
        
        Returns:
            tuple of (latents, spikes, behavior) tensors
        """
        latent_dataset = []
        train_spikes = []
        behavior = []
        self.ae_model.eval()
        for i, batch in tqdm(
            enumerate(self.full_dataloader),
            total=len(self.full_dataloader),
            desc="Creating latent dataset",
        ):
            with torch.no_grad():
                z = self.ae_model.encode(batch["signal"])
                latent_dataset.append(z.cpu())
                train_spikes.append(batch["signal"].cpu())
                behavior.append(batch["behavior"].cpu())
        return torch.cat(latent_dataset), torch.cat(train_spikes), torch.cat(behavior)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of sample to get
            
        Returns:
            dict containing:
                signal: Original spike counts
                latent: Latent representation
                behavior: Behavioral variables
                behavior_angle: Reach angle
        """
        return {
            "signal": self.train_spikes[idx],
            "latent": self.latents[idx],
            "behavior": self.behavior[idx],
            "behavior_angle": self.behavior_angles[idx],
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
    print("Sample shapes:")
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
