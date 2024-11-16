import torch
from tqdm import tqdm
import numpy as np


def sample(
    ema_denoiser,
    scheduler,
    cfg,
    batch_size=1,
    generator=None,
    device="cuda",
    signal_length=None
):  
    if signal_length is None:
        signal_length = cfg.dataset.signal_length
    z_t = torch.randn(
        (batch_size, cfg.denoiser_model.C_in, signal_length)
    ).to(device)
    ema_denoiser_avg = ema_denoiser.averaged_model
    ema_denoiser_avg.eval()


    scheduler.set_timesteps(cfg.denoiser_model.num_train_timesteps)

    for t in tqdm(scheduler.timesteps, desc="Sampling DDPM"):
        with torch.no_grad():
            model_output = ema_denoiser_avg(
                z_t, torch.tensor([t] * batch_size).to(device).long()
            )
        z_t = scheduler.step(
            model_output, t, z_t, generator=generator, return_dict=False
        )[0]

    return z_t


def sample_spikes(ema_denoiser, scheduler, ae, cfg, latent_dataset_train, batch_size=1, device="cuda"):
    z_t = torch.randn(
        (batch_size, cfg.denoiser_model.C_in, cfg.dataset.signal_length)
    ).to(device)
    ema_denoiser_avg = ema_denoiser.averaged_model
    ema_denoiser_avg.eval()
    scheduler.set_timesteps(cfg.denoiser_model.num_train_timesteps)

    for t in tqdm(scheduler.timesteps, desc="Sampling DDPM"):
        with torch.no_grad():
            model_output = ema_denoiser_avg(
                z_t, torch.tensor([t] * batch_size).to(device).long()
            )
        z_t = scheduler.step(model_output, t, z_t, return_dict=False)[0]

    z_t = z_t * latent_dataset_train.latent_stds.to(z_t.device) + latent_dataset_train.latent_means.to(z_t.device)

    with torch.no_grad():
        rates = ae.decode(z_t).cpu()
    
    spikes = torch.poisson(rates)

    return spikes, rates
    


def split_tensor_randomly(gt_spikes, num_splits=5):
    """Split tensor into two parts randomly for a number of times."""
    results = {'split_1': [], 'split_2': []}
    
    num_elements = gt_spikes.shape[0]
    half_size = num_elements // 2
    
    for _ in range(num_splits):
        indices = np.random.permutation(num_elements)
        first_half_indices = indices[:half_size]
        second_half_indices = indices[half_size:]
        # assert no overlap
        assert len(set(first_half_indices).intersection(set(second_half_indices))) == 0
        #print(first_half_indices[:5], second_half_indices[:5])
        
        results['split_1'].append(gt_spikes[first_half_indices])
        results['split_2'].append(gt_spikes[second_half_indices])
        
    return results


def sample_with_replacement(data, num_samples):
    """Sample given data with replacement num_samples times."""
    indices = np.random.choice(len(data), num_samples, replace=True)
    return data[indices]