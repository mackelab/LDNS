# ---- Imports ---- #
import torch
import h5py

# ---- Run params ---- #
# make sure to match these to the config!
# these are the default values in the paper
system_name = "Lorenz"
signal_length = 256  # length of each sequence
C_in = 128  # number of neurons
n_ic = 5000  # number of initial conditions (total sequences)
mean_rate = 0.3  # mean firing rate in Hz
split_frac_train = 0.7  # fraction of data for training
split_frac_val = 0.1  # fraction of data for validation
random_seed = 42  # for reproducibility
softplus_beta = 2.0  # controls sharpness of rate nonlinearity

save_path = "baselines/lfads/data/datasets/lorenz.h5"

# ---- Helper funcs ---- #
from ldns.data.latent_attractor import get_attractor_dataloaders

# ---- Generate data ---- #
# create dataloaders for train/val/test splits
train_dataloader, val_dataloader, test_dataloader = get_attractor_dataloaders(
    system_name=system_name,
    n_neurons=C_in,
    sequence_length=signal_length,
    noise_std=0.05,
    n_ic=n_ic,
    mean_spike_count=mean_rate * signal_length,
    train_frac=split_frac_train,
    valid_frac=split_frac_val,  # test is 1 - train - valid
    random_seed=random_seed,
    batch_size=1,
    softplus_beta=softplus_beta,
)

# ---- Extract data from dataloaders ---- #
# extract spikes (shape: [batch, time, neurons])
train_spikes = torch.stack(
    [train_dataloader.dataset[i]["signal"] for i in range(len(train_dataloader.dataset))]
).permute(0, 2, 1)

val_spikes = torch.stack(
    [val_dataloader.dataset[i]["signal"] for i in range(len(val_dataloader.dataset))]
).permute(0, 2, 1)

test_spikes = torch.stack(
    [test_dataloader.dataset[i]["signal"] for i in range(len(test_dataloader.dataset))]
).permute(0, 2, 1)

# extract rates (shape: [batch, time, neurons])
train_rates = torch.stack(
    [train_dataloader.dataset[i]["rates"] for i in range(len(train_dataloader.dataset))]
).permute(0, 2, 1)

val_rates = torch.stack(
    [val_dataloader.dataset[i]["rates"] for i in range(len(val_dataloader.dataset))]
).permute(0, 2, 1)

test_rates = torch.stack(
    [test_dataloader.dataset[i]["rates"] for i in range(len(test_dataloader.dataset))]
).permute(0, 2, 1)

# extract latents (shape: [batch, time, latent_dim])
train_latents = torch.stack(
    [train_dataloader.dataset[i]["latents"] for i in range(len(train_dataloader.dataset))]
).permute(0, 2, 1)

val_latents = torch.stack(
    [val_dataloader.dataset[i]["latents"] for i in range(len(val_dataloader.dataset))]
).permute(0, 2, 1)

test_latents = torch.stack(
    [test_dataloader.dataset[i]["latents"] for i in range(len(test_dataloader.dataset))]
).permute(0, 2, 1)

# print data shapes for verification
print(f"Train data shape: {train_spikes.shape}")
print(f"Valid data shape: {val_spikes.shape}")
print(f"Test data shape: {test_spikes.shape}")
print(f"Train rates shape: {train_rates.shape}")
print(f"Train latents shape: {train_latents.shape}")

# ---- Save data to HDF5 ---- #
with h5py.File(save_path, 'w') as h5file:
    # save spike data (both encoding and reconstruction targets)
    h5file.create_dataset('train_encod_data', data=train_spikes, compression="gzip")
    h5file.create_dataset('train_recon_data', data=train_spikes, compression="gzip")
    h5file.create_dataset('valid_encod_data', data=val_spikes, compression="gzip")
    h5file.create_dataset('valid_recon_data', data=val_spikes, compression="gzip")
    h5file.create_dataset('test_encod_data', data=test_spikes, compression="gzip")
    h5file.create_dataset('test_recon_data', data=test_spikes, compression="gzip")
    
    # save ground truth rates
    h5file.create_dataset('train_truth', data=train_rates, compression="gzip")
    h5file.create_dataset('valid_truth', data=val_rates, compression="gzip")
    h5file.create_dataset('test_truth', data=test_rates, compression="gzip")
    
    # save ground truth latents
    h5file.create_dataset('train_latents', data=train_latents, compression="gzip")
    h5file.create_dataset('valid_latents', data=val_latents, compression="gzip")
    h5file.create_dataset('test_latents', data=test_latents, compression="gzip")
    
    # save conversion factor (used in some analyses)
    h5file.create_dataset('conversion_factor', data=1.0)
