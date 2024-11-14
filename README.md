# Latent Diffusion for Neural Spiking Data

![LDNS Method Overview](./assets/LDNS_schematic.png)


This repository contains research code for the NeurIPS 2024 Spotlight paper:   
 [***Latent Diffusion for Neural Spiking Data***](https://arxiv.org/abs/2407.08751)   
 by [Kapoor*](https://jkapoor.me), [Schulz*](https://www.linkedin.com/in/auguste-schulz-b5a57a168/), [Vetter](https://www.linkedin.com/in/julius-vetter-060ab11b8?originalSubdomain=de), [Pei](https://www.linkedin.com/in/felix-pei-b41742196/), [Gao†](https://www.rdgao.com), and [Macke†](https://mackelab.org) (2024).



## Installation

To run the scripts make sure to first install all the requirements. We recommend creating a conda environment first.
A GPU is recommend but not necessary.

```bash 
git clone git@github.com:mackelab/LDNS.git
cd LDNS
conda create --name ldns python=3.9
conda activate ldns
pip install -e . # install from requirements.txt
# optional: install jupyter notebook
pip install jupyter
```

### Downloading human and monkey data

If you want to run the experiments on the human and monkey data, download and store the datasets:
- Monkey data: https://dandiarchive.org/dandiset/000128 (store in `data/monkey`)
- Human data (Willet et al, 2023): https://datadryad.org/stash/downloads/file_stream/2547369 (store in `data/human`)


## Running the experiments

The core model and data loading code is in the `ldns` directory. The classes and functions in this code are used in the notebooks and scripts in the `notebooks` directory.

### Lorenz system



### Human BCI data



### Monkey reach data (unconditional)


### Monkey reach data (conditional)



## Baselines for the Lorenz and Monkey Reach Data

In the paper, we compared LDNS to a number of VAE-based baselines. The code for these baselines can be found in the `baselines` directory.

### LFADS
> Latent Factor Analysis via Dynamical Systems (Pandarinath et al, 2018)

**Lorenz system** We 


### pi-VAE
> Poisson Identifiable VAE (Zhou et al, 2024)

### TNDM
> Targeted Neural Dynamical Modeling (Hurwitz et al, 2022)




> Note that this repository is work in progress. 

## Citation

```
@inproceedings{kapoorschulz2024ldns,
	author = {Jaivardhan Kapoor and Auguste Schulz and Julius Vetter and Felix C Pei and Richard Gao and Jakob H. Macke},  
	title = {Latent Diffusion for Neural Spiking Data},  
	journal = {Advances in Neural Information Processing Systems},
	year = {2024}  
}
```
## Contact
Questions regarding the code should be addressed to jaivardhan.kapoor@uni-tuebingen.de or auguste.schulz@uni-tuebingen.de.
