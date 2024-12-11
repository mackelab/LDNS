# ---- Imports ---- #
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors
import numpy as np
import h5py
from pathlib import Path

# ---- Run params ---- #
# make sure to match these to the config!
DANDI_ROOT = Path("LDNS/data").expanduser()  # change appropriately where you downloaded the monkey data
dataset_name = "monkey"  # which dataset to use
bin_size_ms = 5  # temporal bin size in ms
split_frac_train = 0.8  # fraction of data for training
split_frac_val = 0.05  # fraction of data for validation

# ---- Dataset locations ---- #
datapath_dict = {
    "monkey": DANDI_ROOT / "000128/sub-Jenkins/",
}
prefix_dict = {
    "monkey": "*full",
}
datapath = datapath_dict[dataset_name]
prefix = prefix_dict.get(dataset_name, "")

# construct save path based on parameters
save_path = f"baselines/lfads/data/datasets/{dataset_name}_{bin_size_ms}ms.h5"

# ---- Load dataset ---- #
dataset = NWBDataset(datapath, prefix)
dataset.resample(bin_size_ms)

# ---- Extract data from dataset ---- #
# get train and validation data
train_dict = make_train_input_tensors(
    dataset, dataset_name, "train", save_file=False, include_forward_pred=False, include_behavior=True
)
val_dict = make_train_input_tensors(
    dataset, dataset_name, "val", save_file=False, include_forward_pred=False, include_behavior=True
)

# combine heldin and heldout neurons
train_spikes = np.dstack([train_dict["train_spikes_heldin"], train_dict["train_spikes_heldout"]])
val_spikes = np.dstack([val_dict["train_spikes_heldin"], val_dict["train_spikes_heldout"]])

# ---- Create train/val/test splits ---- #
# combine all spikes
spikes = np.concatenate([train_spikes, val_spikes], axis=0)
train_data = spikes[0 : len(train_spikes) + len(val_spikes) // 2]
n_val_test = len(spikes) - len(train_data)
valid_data = spikes[
    len(train_spikes) + len(val_spikes) // 2 : len(train_spikes) + len(val_spikes) // 2 + n_val_test // 4
]
test_data = spikes[len(train_spikes) + len(val_spikes) // 2 + n_val_test // 4 :]

# split behavior data similarly
train_behavior = train_dict["train_behavior"]
val_behavior = val_dict["train_behavior"]
behavior = np.concatenate([train_behavior, val_behavior], axis=0)
train_beh = behavior[0 : len(train_behavior) + len(val_behavior) // 2]
n_val_test = len(behavior) - len(train_beh)
valid_beh = behavior[
    len(train_behavior) + len(val_behavior) // 2 : len(train_behavior) + len(val_behavior) // 2 + n_val_test // 4
]
test_beh = behavior[len(train_behavior) + len(val_behavior) // 2 + n_val_test // 4 :]

# ---- Save data to HDF5 ---- #
with h5py.File(save_path, "w") as h5file:
    # save spike data (both encoding and reconstruction targets)
    h5file.create_dataset("train_encod_data", data=train_data, compression="gzip")
    h5file.create_dataset("train_recon_data", data=train_data, compression="gzip")
    h5file.create_dataset("valid_encod_data", data=valid_data, compression="gzip")
    h5file.create_dataset("valid_recon_data", data=valid_data, compression="gzip")
    h5file.create_dataset("test_encod_data", data=test_data, compression="gzip")
    h5file.create_dataset("test_recon_data", data=test_data, compression="gzip")

    # save behavior data
    h5file.create_dataset("train_behavior", data=train_beh, compression="gzip")
    h5file.create_dataset("valid_behavior", data=valid_beh, compression="gzip")
    h5file.create_dataset("test_behavior", data=test_beh, compression="gzip")

    # save conversion factor (used in some analyses)
    h5file.create_dataset("conversion_factor", data=1.0)

# ---- Print data shapes for verification ---- #
print(f"Train data shape: {train_data.shape}")
print(f"Valid data shape: {valid_data.shape}")
print(f"Test data shape: {test_data.shape}")
