# %%
import multiprocessing
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from dysts.flows import Lorenz

from ldns.utils.utils import set_seed


def simulate_single_trajectory(args: Tuple) -> np.ndarray:
    """simulate a single trajectory of the attractor system

    args:
        args: tuple containing (system_name, initial_condition, sequence_length, burn_in, random_seed)

    returns:
        normalized trajectory array
    """
    initial_condition, sequence_length, burn_in, random_seed = args
    system = Lorenz()
    set_seed(random_seed)

    system.random_state = random_seed
    system.ic = initial_condition.tolist()

    trace = system.make_trajectory(sequence_length + burn_in)
    trace = trace[burn_in:]  # discard burn-in period

    # normalize the trace
    trace = np.array(trace)
    trace = trace - np.mean(trace, axis=0)
    trace = trace / np.std(trace, axis=0)
    return trace

def simulate_attractor_parallel(
    num_seq: int = 10, 
    burn_in: int = 100, 
    sequence_length: int = 1000, 
    random_seed: int = 42
) -> torch.Tensor:
    """simulate the attractor system in parallel using multiprocessing

    args:
        system_name: name of the attractor system
        num_seq: number of sequences to generate
        burn_in: number of initial timesteps to discard
        sequence_length: length of each sequence
        random_seed: random seed for reproducibility

    returns:
        tensor of shape (num_seq, sequence_length, system_dimension)
    """
    system = Lorenz()
    initial_conditions = system.ic
    set_seed(random_seed)
    
    # prepare initial conditions
    initial_conditions = (
        np.array(initial_conditions).reshape(1, -1).repeat(num_seq, axis=0)
    )
    initial_conditions += np.random.randn(*initial_conditions.shape) * 2.0

    args_list = [
        (initial_conditions[i], sequence_length, burn_in, random_seed + i)
        for i in range(num_seq)
    ]

    num_workers = min(num_seq, multiprocessing.cpu_count())

    with multiprocessing.Pool(num_workers) as pool:
        results = []
        for result in tqdm(
            pool.imap(simulate_single_trajectory, args_list),
            total=num_seq,
            desc="Simulating Lorenz",
        ):
            results.append(result)

    return torch.tensor(results, dtype=torch.float32)

class AttractorDataset(Dataset):
    """dataset for attractor dynamics with poisson observations"""
    
    def __init__(
        self,
        system_name: str,
        n_neurons: int,
        sequence_length: int = 100,
        n_ic: int = 5,
        mean_spike_count: float = 500.0,
        random_seed: int = 42,
        softplus_beta: float = 1.0,
        time_last: bool = True,
    ):
        """initialize dataset

        args:
            system_name: name of attractor system from dysts.flows
            n_neurons: number of output dimensions
            sequence_length: length of sequences to generate
            n_ic: number of initial conditions
            mean_spike_count: mean spike count for poisson observations
            random_seed: random seed for reproducibility
            softplus_beta: beta parameter for softplus nonlinearity
            time_last: if true, return tensors with time as last dimension
        """
        super().__init__()
           
        self.time_last = time_last
        set_seed(random_seed)

        # generate latent trajectories
        latents = simulate_attractor_parallel(
            num_seq=n_ic,
            sequence_length=sequence_length,
            random_seed=random_seed,
        )

        # create observation matrix
        self.C = torch.randn(n_neurons, latents.shape[-1], dtype=torch.float32)
        self.C /= torch.norm(self.C, dim=1, keepdim=True)
        self.C = self.C[self.C[:, 0].argsort()]  # sort based on first column

        # create bias term
        self.b = torch.log(
            torch.tensor(mean_spike_count) / sequence_length
        ) * torch.ones(n_neurons, 1)

        # compute rates and samples
        self.log_rates = torch.einsum("ij,klj->kli", self.C, latents) + self.b.squeeze()
        self.rates = torch.nn.functional.softplus(self.log_rates, beta=softplus_beta)
        self.samples = torch.poisson(self.rates)
        self.latents = latents

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """get a sample from the dataset

        args:
            idx: index of sample to get

        returns:
            dictionary containing:
                signal: spike counts [C, T] if time_last else [T, C]
                latents: latent states [D, T] if time_last else [T, D]
                rates: firing rates [C, T] if time_last else [T, C]
        """
        if self.time_last:
            return {
                "signal": self.samples[idx].permute(1, 0),
                "latents": self.latents[idx].permute(1, 0),
                "rates": self.rates[idx].permute(1, 0)
            }
        return {
            "signal": self.samples[idx],
            "latents": self.latents[idx],
            "rates": self.rates[idx]
        }

class LatentDataset(torch.utils.data.Dataset):
    """Dataset class to store latents from autoencoder for training the diffusion model.
    init takes a dataloader and an autoencoder model.
    """
    def __init__(
        self, dataloader, ae_model, clip=True, latent_means=None, latent_stds=None
    ):
        """
        Args:
            dataloader: dataloader to get the data from
            ae_model: autoencoder model to encode the data
            clip: whether to clip the latents to -5, 5 (does not change perf, but stabilizes training)
            latent_means: means of the latent dataset (if None, compute from the dataset)
            latent_stds: stds of the latent dataset (if None, compute from the dataset)
        """
        self.full_dataloader = dataloader
        self.ae_model = ae_model
        self.latents = self.create_latents()
        # normalize to N(0, 1)
        if latent_means is None or latent_stds is None:
            self.latent_means = self.latents.mean(dim=(0, 2)).unsqueeze(0).unsqueeze(2)
            self.latent_stds = self.latents.std(dim=(0, 2)).unsqueeze(0).unsqueeze(2)
        else:
            self.latent_means = latent_means
            self.latent_stds = latent_stds
        self.latents = (self.latents - self.latent_means) / self.latent_stds
        if clip:
            self.latents = self.latents.clamp(-5, 5)

    def create_latents(self):
        """
        Create the latents from the autoencoder.
        """
        latent_dataset = []
        self.ae_model.eval()
        for i, batch in tqdm(
            enumerate(self.full_dataloader),
            total=len(self.full_dataloader),
            desc="Creating latent dataset",
        ):
            with torch.no_grad():
                z = self.ae_model.encode(batch["signal"])
                latent_dataset.append(z.cpu())
        return torch.cat(latent_dataset)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx]


def get_attractor_dataloaders(
    system_name: str,
    n_neurons: int = 128,
    sequence_length: int = 500,
    n_ic: int = 100,
    mean_spike_count: float = 200.0,
    batch_size: int = 100,
    train_frac: float = 0.7,
    valid_frac: float = 0.15,
    random_seed: int = 42,
    softplus_beta: float = 1.0,
    time_last: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """create train/val/test dataloaders for attractor data

    args:
        system_name: name of attractor system
        n_neurons: number of output dimensions
        sequence_length: length of sequences
        n_ic: number of initial conditions
        mean_spike_count: mean spike count for poisson observations
        batch_size: batch size for dataloaders
        train_frac: fraction of data for training
        valid_frac: fraction of data for validation
        random_seed: random seed
        softplus_beta: beta parameter for softplus
        time_last: if true, return tensors with time as last dimension
        num_workers: number of workers for dataloaders

    returns:
        train, validation and test dataloaders
    """
    # create dataset
    dataset = AttractorDataset(
        system_name=system_name,
        n_neurons=n_neurons,
        sequence_length=sequence_length,
        n_ic=n_ic,
        mean_spike_count=mean_spike_count,
        random_seed=random_seed,
        softplus_beta=softplus_beta,
        time_last=time_last,
    )

    # calculate split sizes
    total_size = len(dataset)
    train_size = int(train_frac * total_size)
    valid_size = int(valid_frac * total_size)
    test_size = total_size - train_size - valid_size

    # split dataset
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader

# %%
if __name__ == "__main__":
    # test dataset creation
    dataset = AttractorDataset(
        system_name="LorenzCoupled",
        n_neurons=128,
        sequence_length=1024,
        n_ic=100,
        mean_spike_count=200,
        random_seed=42,
    )
    
    # visualize sample
    import matplotlib.pyplot as plt

    sample = dataset[1]
    plt.figure(figsize=(6, 4))
    plt.matshow(sample["signal"], aspect="auto")
    plt.colorbar()
    plt.title("Spike counts")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.matshow(sample["rates"], aspect="auto")
    plt.colorbar()
    plt.title("Firing rates")
    plt.show()

    # test dataloader creation
    train_loader, valid_loader, test_loader = get_attractor_dataloaders(
        "Lorenz", n_neurons=128, sequence_length=500, n_ic=100
    )

    # print shapes from each loader
    for loader, name in [(train_loader, "Train"), 
                        (valid_loader, "Valid"), 
                        (test_loader, "Test")]:
        batch = next(iter(loader))
        print(f"\n{name} loader shapes:")
        print(f"Signal: {batch['signal'].shape}")
        print(f"Latents: {batch['latents'].shape}")
        print(f"Rates: {batch['rates'].shape}")

# %%
