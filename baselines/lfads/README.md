# LFADS

This code is for running LFADS baselines for Lorenz and Monkey datasets.

The LFADS code and config files use the format in [LFADS-Torch](https://github.com/arsedler9/lfads-torch).

To avoid conflicting package installations, we recommend installing LFADS-Torch in a separate environment from LDNS.

```bash
conda create -n lfads python=3.9
conda activate lfads
# install from lfads-torch repo
wget https://github.com/arsedler9/lfads-torch/archive/d8b4b3ba87a49fd74a3c06afb3ec1b695c6a2227.zip
# extract the zip file
unzip d8b4b3ba87a49fd74a3c06afb3ec1b695c6a2227.zip
pip install lfads-torch-0.1.0.dev20241204.tar.gz  # ensure this file exists after extraction
```

Then, we run the scripts in this format:

### 1. Data preparation

Run the scripts in `baselines/lfads/` -- `prep_lorenz_data.py` and `prep_monkey_data.py`. This will create HDF5 files in `baselines/lfads/data/datasets/` -- `lorenz.h5` and `monkey_5ms.h5`.

### 2. Run LFADS

Run `baselines/lfads/scripts/run_pbt.py` to train a model on the monkey and lorenz datasets. Change the `DATASET_STR` variable to switch between datasets. This runs a population-based training (PBT) search over the hyperparameter space, and the best model is saved separately at the end.

### 3. Create sampler state dict

Run `baselines/lfads/create_unconditional_sampler.py` to create a sampler with a behavior readout. This will save the sampler state dict to `baselines/lfads/data/samplers/`. We will use this in `sample_lfads.ipynb` to generate samples from the LFADS model.

### 4. Generate samples from LFADS model

Run `sample_lfads.ipynb` to generate samples from the LFADS model.

### 5. Evaluate LFADS model

This is done in `LDNS/notebooks` under baseline comparisons.

