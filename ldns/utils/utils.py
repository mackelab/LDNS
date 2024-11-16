import csv
import datetime
import os
import pickle
import random
import tempfile

import numpy as np
import wandb
from einops import rearrange
from omegaconf import OmegaConf as OC
from scipy import signal

import torch


# -------------------------- ntd utils --------------------------
def standardize_array(arr, ax, set_mean=None, set_std=None, return_mean_std=False):
    """
    Standardize array along given axis. set_mean and set_std can be used to manually set mean and standard deviation.

    Args:
        arr: Array to be standardized.
        ax: Axis along which to standardize.
        set_mean: If not None, use this value as mean.
        set_std: If not None, use this value as standard deviation.
        return_mean_std: If True, return mean and standard deviation that were used for standardization.

    Returns:
        Standardized array.
        If return_mean_std is True: Mean
        If return_mean_std is True: Standard deviation
    """

    if set_mean is None:
        arr_mean = np.mean(arr, axis=ax, keepdims=True)
    else:
        arr_mean = set_mean
    if set_std is None:
        arr_std = np.std(arr, axis=ax, keepdims=True)
    else:
        arr_std = set_std

    assert np.min(arr_std) > 0.0
    if return_mean_std:
        return (arr - arr_mean) / arr_std, arr_mean, arr_std
    else:
        return (arr - arr_mean) / arr_std
    
    

def l2_distances(tens_one, tens_two):
    """
    Compute pairwise L2 distances between two signal tensors.

    Args:
        tens_one: First tensor.
        tens_two: Second tensor.

    Returns:
        Pairwise L2 distances.
        Tuple of IDs of two nearest neighbors.
        Nearest neighbor from first signal tensor.
        Nearest neighbor from second signal tensor.
    """

    res = torch.cdist(
        rearrange(tens_one, "b c l -> b (c l)"), rearrange(tens_two, "b c l -> b (c l)")
    )

    flat_argmin = torch.argmin(res).item()
    row = flat_argmin // res.shape[1]
    col = flat_argmin % res.shape[1]

    return res, (row, col), tens_one[row], tens_two[col]




def count_parameters(model):
    """
    Count number of trainable parameters in model.

    Args:
        model: Pytorch model.

    Returns:
        Number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_pickle(obj, file_path):
    """
    Save object as pickle file.

    Args:
        obj: Object to be saved.
        file_path: Path to save file to.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        raise FileExistsError(f"{file_path} already exists!")

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(file_name, folder_path):
    """
    Read pickle file.

    Args:
        file_name: Name of file.
        folder_path: Path to folder containing file.
    """
    file_path = os.path.join(folder_path, file_name)
    assert file_path.endswith(".pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
    
def config_saver(
    file,
    filename,
    cfg=None,
    experiment=None,
    tag=None,
    project=None,
    entity=None,
    dir=None,
    mode=None,
    local_path=None,
):
    """
    Save file to wandb or locally.

    Args:
        file: File to be saved.
        filename: Name of file.
        cfg: Config object.
        experiment: Name of wandb experiment.
        tag: Tag for file.
        project: Name of wandb project.
        entity: Name of wandb entity.
        dir: wandb directory.
        mode: wandb mode. "online" for uploading, "disabled" to deactivate wandb.
        local_path: Path to save file to locally.
            None or "". None will check in cfg, "" will not save locally
    """
    # infer params from cfg if not given
    if cfg is not None:
        if experiment is None:
            experiment = cfg.base.experiment
        if tag is None:
            tag = cfg.base.tag
        if project is None:
            project = cfg.base.wandb_project
        if entity is None:
            entity = cfg.base.wandb_entity
        if dir is None:
            dir = cfg.base.home_path
        if mode is None:
            mode = cfg.base.wandb_mode
        if local_path is None:
            local_path = cfg.base.save_path

    # save to wandb
    if mode == "online":
        config = (
            OC.to_container(cfg, resolve=True, throw_on_missing=True)
            if cfg is not None
            else None
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run = wandb.init(
                mode=mode,
                project=project,
                entity=entity,
                dir=dir,
                group=experiment,
                config=config,
                name=f"SAVING_RUN:{tag}_{filename}",
            )
            temp_file_path = os.path.join(tmpdir, filename)
            with open(temp_file_path, "wb") as f:
                pickle.dump(file, f)
            run.save(temp_file_path, base_path=tmpdir)
            run.finish()

    # save locally
    if not (local_path is None or local_path == ""):
        save_pickle(file, os.path.join(local_path, experiment, f"{tag}_{filename}"))


# -------------------------- new utils --------------------------

def set_seeds(seed):
    """
    Sets the seed for various libraries to ensure reproducibility.
    
    Parameters:
    - seed: The seed value to use for random number generators.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups, use manual_seed_all
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed set to {seed}.")


def set_seed(seed: int):
    """
    Set the seed for all random number generators and switch to deterministic algorithms.
    This can hurt performance!

    Args:
        seed: The random seed.
        JAIs version 
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)




def load_and_flatten(config):
    """ Load the configuration and flatten it for logging."""
    if OC.is_config(config):
        flat_config = flatten_config(OC.to_object(config))
    else:
        # If it's already a dict or has been inadvertently converted, use it as is
        flat_config = flatten_config(config)
    return flat_config


def flatten_config(config, sep="."):
    """
    Flatten a nested configuration into a flat dictionary.
    """
    items = []
    for key in config.keys():
        v = config[key]
        if isinstance(v, dict):
            for small_key in config[key].keys():
                new_key = f"{key}{sep}{small_key}"
                items.append((new_key, config[key][small_key]))
        else:
            items.append((key, v))
    return dict(items)


def write_config_to_csv(flat_config, csv_file_path):
    """
    Write flattened configuration to a CSV file.
    """
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        for key, value in flat_config.items():
            writer.writerow([key, value])




def log_run_details_to_csv(csv_file_path, config, added_field=None, added_name="add"):
    """ Log run details to a CSV file.
        check if the file exists, if not, write headers
        if the file exists, check if the headers match, if not, create a new file
        if the headers match, append the new row
    """
    # Flatten configuration to a single-level dictionary
    if OC.is_config(config):
        flat_config = flatten_config(OC.to_object(config))
    else:
        flat_config = flatten_config(config)

    # Optionally add an additional field
    if added_field is not None:
        flat_config[added_name] = added_field

    # Prepare to write to CSV
    headers = list(flat_config.keys())
    row_data = flat_config

    # Check if CSV file exists and read the existing headers if it does
    if os.path.isfile(csv_file_path):
        with open(csv_file_path, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            existing_headers = next(reader, None)  # Read the first line for headers

        # If headers do not match or the file is empty, create a new file
        if not existing_headers or sorted(existing_headers) != sorted(headers):
            # Optionally, rename the old file instead of overwriting with timestamp
            os.rename(
                csv_file_path,
                csv_file_path[:-4]
                + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
                + ".csv",
            )
            with open(csv_file_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerow(row_data)
        else:
            # Headers match, append the new row
            with open(csv_file_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writerow(row_data)
    else:
        # File doesn't exist, create and write header and first row
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerow(row_data)
