import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from scipy.signal import correlate
from scipy.special import gammaln
from scipy.stats import gaussian_kde
from sklearn.metrics import explained_variance_score, r2_score


class ReconDataset(torch.utils.data.Dataset):
    def __init__(self, latents, samples, rates):
        self.latents = latents
        self.samples = samples
        self.rates = rates

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.samples[idx], self.rates[idx]


def mse(true, predicted):
    """Calculate the mean squared error, averaged across samples and neurons."""
    return np.mean((true - predicted) ** 2)


def rmse(true, predicted):
    """Calculate the root mean squared error, averaged across samples and neurons."""
    return np.sqrt(mse(true, predicted))


def neg_log_likelihood(rates, spikes, zero_warning=True, reduction="sum"):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    by Felix Pei 2021 NLB

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
        spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            print(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)

    if reduction == "sum":
        return np.sum(result)
    elif reduction == "mean":
        return np.mean(result)
    elif reduction == "none":
        return result


def average_rates(data, mode="neur", fps=None):
    """
    Computes the average of spike counts or rates with the option to do neuronwise averaging
    or treating each neuron and time series individually.

    Parameters:
    ----------
    data : np.ndarray
        3D numpy array containing spike counts with shape (n_samples, n_seqlen, n_neurons).
    mode : str, optional
        The mode of averaging. 'neur' for averaging across all samples and time
        series for each neuron, 'neurtime' for averaging each neuron's spike count
        for each time series across all samples. 'neursample' for averaging each neuron's spike count
        for each time series across all time points.
        Default is 'neur'.
    """
    if mode == "neur":
        # Average across the first two dimensions (samples and time series)
        averaged = np.nanmean(data, axis=(0, 1)).flatten()
    elif mode == "neurtime":
        # Average across the samples dimension only
        averaged = np.nanmean(data, axis=0).flatten()
    elif mode == "neursample":
        # Average across the time dimension only
        averaged = np.nanmean(data, axis=1).flatten()
    else:
        raise ValueError(
            "Invalid mode. Choose either 'neuronwise' or 'neurontimewise' or 'neuronsamplewise'."
        )

    if fps is not None:
        # Convert to Hz by multiplying with the frame rate
        averaged = averaged * fps
    return averaged


def std_rates(data, mode="neur", fps=None):
    """
    Computes the standard deviation of spike counts or rates with the option to do neuronwise averaging
    or treating each neuron and time series individually.
    """
    if mode == "neur":
        # Average across the first two dimensions (samples and time series)
        stdev = np.nanstd(data, axis=(0, 1)).flatten()
    elif mode == "neurtime":
        # Average across the samples dimension only
        stdev = np.nanstd(data, axis=0).flatten()
    elif mode == "neursample":
        # Average across the time dimension only
        stdev = np.nanstd(data, axis=1).flatten()
    else:
        raise ValueError(
            "Invalid mode. Choose either 'neuronwise' or 'neurontimewise' or 'neuronsamplewise'."
        )

    if fps is not None:
        # Convert to Hz by multiplying with the frame rate
        stdev = stdev * fps
    return stdev


def correlation_matrix(data, sample=None, mode="concatenate"):
    """Compute averaged correlation matrix across samples."""
    if len(data.shape) > 2:
        if sample is not None:
            return np.corrcoef(data[sample, :, :].T)
        elif mode == "average":
            n_samples = data.shape[0]
            # Compute correlation matrix for each sample
            corrs = np.array([np.corrcoef(data[i, :, :].T) for i in range(n_samples)])
            return np.nanmean(corrs, axis=0)
        elif mode == "concatenate":
            # Concatenate all samples and n_seqlen
            return np.corrcoef(data.reshape(-1, data.shape[-1]).T)
    else:
        return np.corrcoef(data.T)


def autocorrelation(series):
    """Compute the autocorrelation of a series."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    autocorr = np.correlate(series - mean, series - mean, mode="full")[-n:]
    return autocorr[: n // 2] / (var * np.arange(n, n // 2, -1))


def compute_autocorrelations(data, sample=None):
    """Compute autocorrelations for all neurons/latents, optionally averaging over samples."""
    n_samples, n_seqlen, n_neurons = data.shape
    autocorrs = []

    if sample is not None:
        # Compute autocorrelation for each neuron in a specific sample
        autocorrs = [autocorrelation(data[sample, :, i]) for i in range(n_neurons)]
    else:
        # Compute and average autocorrelation over all samples for each neuron
        for i in range(n_neurons):
            # Compute autocorrelation for each sample, then average
            neuron_autocorrs = np.array(
                [autocorrelation(data[j, :, i]) for j in range(n_samples)]
            )
            avg_autocorr = np.mean(neuron_autocorrs, axis=0)
            autocorrs.append(avg_autocorr)

    return autocorrs


def compute_cross_correlation(series1, series2):
    """Compute cross-correlation between two series."""
    mean1, mean2 = series1.mean(), series2.mean()
    series1, series2 = series1 - mean1, series2 - mean2
    cross_corr = np.correlate(series1, series2, mode="full")
    return cross_corr[cross_corr.size // 2 :]


def counts_to_spike_trains(bin_counts, fps):
    """
    Generate spike trains from bin counts.

    Parameters:
    - bin_counts: 3D numpy array (n_samples, n_seqlen, n_neurons) with spike counts.
    - fps: Frames per second, defining the duration of each bin.

    Returns:
    - A dictionary with keys as (sample_index, neuron_index) and values as arrays of spike times.
    """
    n_samples, n_seqlen, n_neurons = bin_counts.shape
    bin_duration = 1.0 / fps
    spike_trains = {}

    for sample_idx in range(n_samples):
        for neuron_idx in range(n_neurons):
            spike_times = []
            for bin_idx in range(n_seqlen):
                count = int(bin_counts[sample_idx, bin_idx, neuron_idx])
                if count > 0:
                    start_time = bin_idx * bin_duration
                    spikes = np.linspace(
                        start_time, start_time + bin_duration, count + 2
                    )[1:-1]
                    spike_times.extend(spikes)
            spike_trains[(sample_idx, neuron_idx)] = np.array(spike_times)

    return spike_trains


def compute_spike_stats(spike_trains, n_samples, n_neurons, mean_output=True):
    """
    Compute statistics for spike trains including Mean ISI, Coefficient of Variation, and CV2.

    Parameters:
    - spike_trains: Dictionary of spike trains with keys as (sample_index, neuron_idx).
    - n_samples: Total number of samples.
    - n_neurons: Total number of neurons.

    Returns:
    - A dictionary of statistics including mean ISIs per neuron, CV, and CV2 across all samples.
    """
    
    # set up arrays and fill with nans
    mean_isis_per_sample_neuron = np.full((n_samples, n_neurons), np.nan)
    std_isis_per_sample_neuron = np.full((n_samples, n_neurons), np.nan)
    cv_isis_per_sample_neuron = np.full((n_samples, n_neurons), np.nan)
    cv2_per_sample_neuron = np.full((n_samples, n_neurons), np.nan)

    for (sample_idx, neuron_idx), spikes in spike_trains.items():
        if len(spikes) > 1:
            isis = np.diff(spikes)
            mean_isi = np.nanmean(isis) if len(isis) > 0 else np.nan
            std_isi = np.nanstd(isis) if len(isis) > 0 else np.nan
            cv_isi = np.nanstd(isis) / mean_isi if mean_isi != 0 else np.nan
            cv2 = compute_cv2(isis)

            mean_isis_per_sample_neuron[sample_idx, neuron_idx] = mean_isi
            std_isis_per_sample_neuron[sample_idx, neuron_idx] = std_isi
            cv_isis_per_sample_neuron[sample_idx, neuron_idx] = cv_isi
            cv2_per_sample_neuron[sample_idx, neuron_idx] = cv2

    if mean_output:
        return {
            "mean_isi": np.nanmean(mean_isis_per_sample_neuron, axis=0),
            "std_isi": np.nanmean(std_isis_per_sample_neuron, axis=0),
            "cv_isi": np.nanmean(cv_isis_per_sample_neuron, axis=0),
            "cv2": np.nanmean(cv2_per_sample_neuron, axis=0),
        }
    else:
        return {
            "mean_isi": mean_isis_per_sample_neuron,
            "std_isi": std_isis_per_sample_neuron,
            "cv_isi": cv_isis_per_sample_neuron,
            "cv2": cv2_per_sample_neuron,
        }
        
        
def compute_spike_stats_per_neuron(spike_trains, n_samples, n_neurons, mean_output=True):
    """
    Compute statistics for spike trains including Mean ISI, Coefficient of Variation, and CV2.

    Parameters:
    - spike_trains: Dictionary of spike trains with keys as (sample_index, neuron_idx).
    - n_samples: Total number of samples.
    - n_neurons: Total number of neurons.

    Returns:
    - A dictionary of statistics including mean ISIs per neuron, CV, and CV2 across all samples.
    """
    isi = {n: [] for n in range(n_neurons)}
    for (sample_idx, neuron_idx), spikes in spike_trains.items():
            if len(spikes) > 1:
                isis = np.diff(spikes)
                isi[neuron_idx].extend(isis)
                
    mean_isi = np.full(n_neurons, np.nan)
    std_isi = np.full(n_neurons, np.nan)
    cv_isi = np.full(n_neurons, np.nan)
    cv2_isi = np.full(n_neurons, np.nan)

    for n in range(n_neurons):
        mean_isi[n] = np.mean(isi[n]) if len(isi[n]) > 0 else np.nan
        std_isi[n] = np.std(isi[n]) if len(isi[n]) > 0 else np.nan
        cv_isi[n] = std_isi[n] / mean_isi[n] if mean_isi[n] != 0 else np.nan
        cv2_isi[n] = compute_cv2(np.array(isi[n]))
        
    return {
            "mean_isi": mean_isi,
            "std_isi": std_isi,
            "cv_isi": cv_isi,
            "cv2": cv2_isi,
        }



def compute_cv2(isis):
    """
    Compute CV2 from ISIs.

    Parameters:
    - isis: 1D numpy array of inter-spike intervals.

    Returns:
    - Mean CV2.
    """
    if len(isis) < 2:
        return np.nan
    cv2_values = 2 * np.abs(isis[:-1] - isis[1:]) / (isis[:-1] + isis[1:])
    return np.nanmean(cv2_values)


def compute_fano_factor(bin_counts):
    """
    Compute the Fano factor of spike counts.

    Parameters:
    - bin_counts: 3D numpy array (n_samples, n_seqlen, n_neurons) with spike counts.

    Returns:
    - A 1D numpy array with the Fano factor for each neuron.
    """
    mean_counts = np.mean(bin_counts, axis=(0, 1))
    var_counts = np.var(bin_counts, axis=(0, 1))
    return var_counts / mean_counts


def bits_per_spike(rates, spikes, mean_rates=None):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    by Felix Pei 2021 NLB modified by Auguste to allow for mean_rates to be passed as naive baseline
    and to allow for reduction to be set to none to allow for element-wise division

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes, reduction="none")

    # allow for mean rates from training set to be passed as naive baseline
    if mean_rates is not None:
        null_rates = np.tile(mean_rates, rates.shape[:-1] + (1,))
    else:
        null_rates = np.tile(
            np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
            spikes.shape[:-1] + (1,),
        )
    nll_null = neg_log_likelihood(
        null_rates, spikes, zero_warning=False, reduction="none"
    )
    # remove reduction to ensure that the division is done element-wise for each neuron
    return (
        (nll_null - nll_model)
        / np.nansum(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True)
        / np.log(2)
    )


def bits_per_spike_original(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    by Felix Pei 2021 NLB

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)



# TODO clean these up

def get_corr_mat(data, mod='trial', name=''):
    name = 'Corr Matrix ' + name
    _, no_samples, xdim = data.shape
    if mod=='trial':
        C_corr = np.zeros((xdim, xdim))

        for s in range(no_samples):
            C_corr += np.corrcoef(data[:, s, :].T)
        C_corr = C_corr / no_samples
        name = name + "av. across trials"
    else:
        data_reshaped = np.reshape(np.transpose(data, (2, 1, 0)), (xdim, -1))
        C_corr = np.corrcoef(data_reshaped)
    # fig_cov = plot_heatmap(C, name=fn, save=False)
    return C_corr, name

def group_neurons_temp_corr(data_x, num_groups=4):
    """
    For display purposes, we divided neurons into 4 groups
    according to their total correlation (using summed correlation coefficients with all other neurons)
    and calculated the average pairwise correlation in each group.
    :param data: time-series in form: [seq_len ,num_trials,x_dim]
    :return: indices of the 4 groups!
    """
    assert isinstance(data_x, np.ndarray), "Input data has to be numpy array!"
    corr_mat, _ = get_corr_mat(data_x, mod='all')
    np.fill_diagonal(corr_mat, 0)
    c_val = np.sum(corr_mat ** 2, axis=0)
    sorted_neurons = np.argsort(-c_val)
    # sorted_c_val = c_val[sorted_neurons]
    g_size = data_x.shape[-1] // num_groups
    neuron_groups = []
    neuron_groups = [sorted_neurons[i:i + g_size] for i in range(0, len(sorted_neurons), g_size)]
    if len(neuron_groups) > num_groups:
        neuron_groups[-2] = np.concatenate((neuron_groups[-2], neuron_groups[-1]), axis=0)
        del neuron_groups[-1]
    return neuron_groups



def get_temp_corr(x1, x2, nlags=10, mode='biased'):
    # biased: 1/T
    # unbiased: abs(np.range(-nlags,nlags))
    # Expected shape: [seq_len, 1]
    T = x1.shape[0]
    assert (x2.shape[0] == T), 'Must be same length!'

    norm_factor = np.ones((nlags * 2 + 1,))
    if mode == 'biased':
        norm_factor = norm_factor * T
    elif mode == 'unbiased':
        norm_factor = T - abs(np.arange(-nlags, nlags + 1))

    full_corr = np.correlate(x1, x2, 'full')
    zero_lag_ind = int(((2 * T - 1) + 1) / 2 - 1)

    corr_result = full_corr[zero_lag_ind - nlags:zero_lag_ind + nlags + 1]
    return np.divide(corr_result, norm_factor)


def get_temp_corr_trial_av(data_x, nlags=10, mode='biased'):
    cross_corr = []
    auto_corr = []

    for i_trial in range(data_x.shape[1]):
        for ii in range(data_x.shape[-1]):
            for jj in range(ii, data_x.shape[-1]):
                xc = get_temp_corr(data_x[:, i_trial, ii], data_x[:, i_trial, jj], nlags, mode=mode)
                if ii == jj:
                    xc[nlags] = 0
                    auto_corr.append(xc)
                else:
                    cross_corr.append(xc)
    return np.array(cross_corr), np.array(auto_corr)


def get_temp_corr_summary(data_x, groups, nlags=10, binWidth=100, mode='biased', batch_first=False):
    if batch_first:
        data_x = np.transpose(data_x, (1, 0, 2))
    seq_len, no_samples, xdim = data_x.shape

    # Subtract trial-based-mean:
    mean_tensor = np.mean(data_x, axis=0)
    data_x = data_x - np.tile(mean_tensor, (seq_len, 1, 1))

    num_groups = len(groups)
    cross_corr_groups = []
    auto_corr_groups = []

    for k in range(num_groups):
        data_group = data_x[:, :, groups[k]]
        cross_corr_g, auto_corr_g = get_temp_corr_trial_av(data_group, nlags, mode)
        cross_corr_groups.append(np.mean(cross_corr_g, axis=0))
        auto_corr_groups.append(np.mean(auto_corr_g, axis=0))

    return cross_corr_groups, auto_corr_groups



# ------------------ spike and rare reconstruction ------------------
def reconstruct_spikes(model, full_dataloader):
    model.eval()
    latents = []
    spikes = []
    rec_spikes = []
    for batch in full_dataloader:
        signal = batch["signal"]
        with torch.no_grad():
            output_rates, z = model(signal)
            z = z.cpu()
        latents.append(z)
        spikes.append(signal.cpu())
        rec_spikes.append(torch.poisson(output_rates.cpu()))
        
    return {
        'latents': torch.cat(latents, 0),
        'spikes': torch.cat(spikes, 0),
        'rec_spikes': torch.cat(rec_spikes, 0)
    }


# ------------------ comparison stats  ------------------


def rmse_nan(y_pred, y):
    return np.sqrt(np.nanmean((y_pred - y)**2))

def kl_div_nan(p, q):
    return np.nansum(p * np.log(p / q))

def corr(y_pred, y):
    return np.corrcoef(y_pred, y)[0, 1]

# -------------------------- from velovity conditioning --------------------------
# import torch
# from einops import rearrange
# from sklearn.linear_model import Ridge
# from sklearn.metrics import r2_score
# from einops import repeat


# # function to train ridge regression on training data
# def train_ridge_regression(train_rates, train_behavior, alpha=1e-6):
#     # reshape and convert to numpy
#     train_rates = rearrange(train_rates, "b c l -> (b l) c").numpy()
#     train_behavior = rearrange(train_behavior, "b c l -> (b l) c").numpy()

#     # train ridge regression
#     ridge_regression_model = Ridge(alpha=alpha)
#     ridge_regression_model.fit(train_rates, train_behavior)

#     return ridge_regression_model


# # function to evaluate ridge regression model on validation data
# def evaluate_ridge_regression(ridge_regression_model, val_rates, val_behavior):
#     # reshape and convert to numpy
#     bs_val = val_rates.shape[0]
#     val_rates = rearrange(val_rates, "b c l -> (b l) c").numpy()
#     val_behavior = rearrange(val_behavior, "b c l -> (b l) c").numpy()

#     # predict behavior using ridge regression model
#     predicted_behavior = ridge_regression_model.predict(val_rates)

#     # calculate r2 score
#     r2 = r2_score(val_behavior, predicted_behavior)
#     print(f"r2 score on val: {r2:.3f}")

#     return {
#         "predicted_val_behavior": rearrange(
#             predicted_behavior, "(b l) c -> b c l", b=bs_val
#         ),
#         "real_val_behavior": rearrange(val_behavior, "(b l) c -> b c l", b=bs_val),
#     }

# # function to generate rates and train decoded behavior
# def gen_rates_and_train_decoded_behavior(
#     ema_denoiser,
#     scheduler,
#     ae,
#     cfg,
#     train_latent_dataloader,
#     val_latent_dataloader,
#     num_samples=100,
#     device="cuda",
#     num_samples_per_batch=1,
#     test_velocity=None,
# ):
#     avg_denoiser = ema_denoiser.averaged_model
#     avg_denoiser.eval()
#     ae.eval()

#     train_rates = []
#     train_behavior = []
#     for batch in train_latent_dataloader:
#         signal = batch["signal"]
#         behavior = batch["behavior"]
#         with torch.no_grad():
#             output_rates = ae(signal)[0].cpu()
#         train_rates.append(output_rates)
#         train_behavior.append(behavior.cpu())

#     train_rates = torch.cat(train_rates, 0)  # [b c l]
#     train_behavior = torch.cat(train_behavior, 0)  # [b 2 l]
#     print(train_rates, train_behavior)

#     val_rates = []
#     val_behavior = []
#     for batch in val_latent_dataloader:
#         signal = batch["signal"]
#         behavior = batch["behavior"]
#         with torch.no_grad():
#             output_rates = ae(signal)[0].cpu()
#         val_rates.append(output_rates)
#         val_behavior.append(behavior.cpu())

#     val_rates = torch.cat(val_rates, 0)  # [b c l]
#     val_behavior = torch.cat(val_behavior, 0)  # [b 2 l]
#     print(val_rates, val_behavior)

#     # train ridge regression model on training data
#     ridge_regression_model = train_ridge_regression(train_rates, train_behavior)

#     # evaluate ridge regression model on validation data
#     evaluation_results = evaluate_ridge_regression(
#         ridge_regression_model, val_rates, val_behavior
#     )

#     # sample from the denoiser
#     print(val_latent_dataloader.dataset.behavior[:num_samples].shape)
#     if num_samples > len(val_latent_dataloader.dataset.behavior):
#         num_samples = len(val_latent_dataloader.dataset.behavior)
    
#     sampled_latents = sample_with_velocity(
#         ema_denoiser=ema_denoiser,
#         scheduler=scheduler,
#         cfg=cfg,
#         batch_size=num_samples * num_samples_per_batch,
#         behavior_vel=repeat(
#             val_latent_dataloader.dataset.behavior[:num_samples],
#             "B C L -> (S B) C L",
#             S=num_samples_per_batch,
#         ),
#         device=device,
#     )
#     print(sampled_latents.shape)
#     sampled_latents = sampled_latents * latent_dataset_train.latent_stds.to(
#         sampled_latents.device
#     ) + latent_dataset_train.latent_means.to(
#         sampled_latents.device
#     )

#     with torch.no_grad():
#         sampled_rates = ae.decode(sampled_latents).cpu()

#     sampled_rates = rearrange(sampled_rates, "(s b) c l -> (s b l) c", s=num_samples_per_batch).numpy()
#     predicted_sampled_behavior = ridge_regression_model.predict(sampled_rates)
#     predicted_sampled_behavior = rearrange(
#         predicted_sampled_behavior, "(s b l) c -> s b c l", b=num_samples, s=num_samples_per_batch
#     )

#     target_sampled_behavior = (
#         val_latent_dataloader.dataset.behavior[:num_samples].cpu().numpy()
#     )
#     evaluation_results["sampled_behavior"] = predicted_sampled_behavior
#     evaluation_results["real_behavior"] = (
#         val_latent_dataloader.dataset.behavior[:num_samples].cpu().numpy()
#     )
    

#     return evaluation_results


def calculate_curvature(traj):
    """Calculate the curvature of a trajectory."""
    dx = np.gradient(traj[0])
    dy = np.gradient(traj[1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / (dx ** 2 + dy ** 2) ** 1.5
    return np.nanmean(curvature)

def filter_straight_trajectories(trajs, threshold):
    """Filter trajectories with curvature below the threshold."""
    straight_trajs = [traj for traj in trajs if calculate_curvature(traj) < threshold]
    return np.array(straight_trajs)

def get_straight_trajectory_indices(trajs, threshold):
    """Get indices of trajectories with curvature below the threshold."""
    indices = [i for i, traj in enumerate(trajs) if calculate_curvature(traj) < threshold]
    curvature = [calculate_curvature(traj) for i, traj in enumerate(trajs)]
    return indices, curvature
