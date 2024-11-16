# %%
import torch
import torch.nn as nn

# for diffusion, we use torch.nn.SmoothL1Loss

def poisson_loss(output, target):
    # Assuming output is log rate (for numerical stability), convert to rate
    rate = torch.exp(output)
    loss = torch.mean(rate - target * output)  # Simplified negative log likelihood
    return loss


def neg_log_likelihood(output, target):
    # output has gone through a softplus
    loss = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")
    return loss(output, target).sum() / output.size(0)


def latent_regularizer(z, cfg):
    """ regualarizer that penalizes the squared difference between latents at neighbouring time steps. This returns sum of squared differences, NOT mean.
    
    Args:
        z: [B, C, L]
        cfg: OmegaConf object
    Returns:
        loss: scalar (torch.Tensor)
    """
    l2_reg = torch.sum(z**2)
    k = cfg.training.get("td_k", 5)  # number of time differences
    temporal_difference_loss = 0

    z_diff = 0
    for i in range(1, k + 1):
        z_diff += ((z[:, :, :-i] - z[:, :, i:]) ** 2 * (1 / (1 + i))).sum()

    temporal_difference_loss = z_diff  # gp prior-like loss
    # it later gets scaled by latent_beta which then only affects l2_reg
    # so the temporal difference loss is multiplied only by training.latent_td_beta
    return l2_reg + temporal_difference_loss / (cfg.training.latent_beta) * (
        cfg.training.latent_td_beta
    )


# %%
if __name__ == "__main__":
    import lovely_tensors

    lovely_tensors.monkey_patch()
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"training": {"latent_beta": 1.0, "latent_td_beta": 1.0}})

    # with sharp latents
    z = torch.randn(4, 8, 1000)
    loss = latent_regularizer(z, cfg) / z.numel()
    print(loss)

    # with smooth latents
    # upscale last dim to 1000 using bilinear interpolation
    z = torch.nn.functional.interpolate(z[:, :, :50], size=1000, mode="linear", align_corners=False)
    loss = latent_regularizer(z, cfg) / z.numel()
    print(loss)

    # loss will be higher for sharp latents, e.g. 4.0 vs 0.8

# %%
