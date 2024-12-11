import hydra
import pickle
import torch
import h5py
import numpy as np
from pathlib import Path

# model paths
config_path = Path("../configs/pbt.yaml")
checkpoint_path = Path("../../data/lfads/best_model/model.ckpt")
dataset_path = Path("../../data/datasets/monkey_5ms.h5")
inference_path = Path("../../data/lfads/best_model/lfads_output_monkey_5ms.h5")

# Compose the train config with properly formatted overrides
DATASET_STR = "ldns_monkey"
overrides = {
    "datamodule": DATASET_STR,
    "model": DATASET_STR,
}
overrides = [f"{k}={v}" for k, v in (overrides).items()]
config_path = Path(config_path)
with hydra.initialize(
    config_path=config_path.parent,
    job_name="run_model",
    version_base="1.1",
):
    config = hydra.compose(config_name=config_path.name, overrides=overrides)

# load checkpoint
model = hydra.utils.instantiate(config.model)
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
sampler_state_dict = dict(
    ic_prior=model.ic_prior.state_dict(),
    decoder=model.decoder.state_dict(),
    readout=model.readout[0].state_dict(),
)

# train regressor to behavior
with h5py.File(dataset_path, "r") as h5f:
    assert "train_behavior" in h5f.keys()
    train_behavior = h5f["train_behavior"][()]
    valid_behavior = h5f["valid_behavior"][()]
with h5py.File(inference_path, "r") as h5f:
    train_rates = h5f["train_output_params"][()]
    valid_rates = h5f["valid_output_params"][()]
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=np.logspace(-6, 0, 7))
ridge.fit(
    train_rates.reshape(-1, train_rates.shape[-1]), 
    train_behavior.reshape(-1, train_behavior.shape[-1]),
)
print(ridge.score(
    valid_rates.reshape(-1, valid_rates.shape[-1]), 
    valid_behavior.reshape(-1, valid_behavior.shape[-1]), 
))
behavior_linear = torch.nn.Linear(train_rates.shape[-1], train_behavior.shape[-1])
with torch.no_grad():
    behavior_linear.weight = torch.nn.Parameter(torch.tensor(ridge.coef_, dtype=torch.float))
    behavior_linear.bias.data[:] = torch.nn.Parameter(torch.tensor(ridge.intercept_, dtype=torch.float))

# add behavior readout to sampler state dict, save to file to be used by LFADS sampler
sampler_state_dict["behavior_readout"] = behavior_linear.state_dict()

with open("../../data/samplers/lfads_sampler.pkl", "wb") as f:
    pickle.dump(sampler_state_dict, f)