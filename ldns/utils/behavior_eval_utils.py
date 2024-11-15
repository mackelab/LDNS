import torch
from einops import rearrange
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge


def compute_decoded_behavior(model, train_latent_dataloader, val_latent_dataloader):
    """
    Compute decoded behavior from neural activity using ridge regression.

    Args:
        model: PyTorch model that takes signal as input and outputs rates
        train_latent_dataloader: DataLoader containing training data with signal and behavior
        val_latent_dataloader: DataLoader containing validation data with signal and behavior

    Returns:
        r2s_per_alpha (list): List of R2 scores for each alpha value tested in ridge regression
        predicted_behavior (np.array): Predicted behavior values from validation data
        val_behavior (np.array): True behavior values from validation data
    """
    model.eval()

    train_rates = []
    train_behavior = []
    for batch in train_latent_dataloader:
        signal = batch["signal"]
        behavior = batch["behavior"]
        with torch.no_grad():
            output_rates = model(signal)[0].cpu()
        train_rates.append(output_rates)
        train_behavior.append(behavior.cpu())

    train_rates = torch.cat(train_rates, 0)  # [B C L]
    train_behavior = torch.cat(train_behavior, 0)  # [B 2 L]
    print(train_rates, train_behavior)

    val_rates = []
    val_behavior = []
    for batch in val_latent_dataloader:
        signal = batch["signal"]
        behavior = batch["behavior"]
        with torch.no_grad():
            output_rates = model(signal)[0].cpu()
        val_rates.append(output_rates)
        val_behavior.append(behavior.cpu())

    val_rates = torch.cat(val_rates, 0)  # [B C L]
    val_behavior = torch.cat(val_behavior, 0)  # [B 2 L]
    print(val_rates, val_behavior)

    # decode rates to behavior using ridge regression
    from sklearn.linear_model import Ridge

    bs_val = val_rates.shape[0]
    train_rates = rearrange(train_rates, "b c l -> (b l) c").numpy()
    val_rates = rearrange(val_rates, "b c l -> (b l) c").numpy()
    train_behavior = rearrange(train_behavior, "b c l -> (b l) c").numpy()
    val_behavior = rearrange(val_behavior, "b c l -> (b l) c").numpy()

    r2s_per_alpha = []

    for i, alpha in enumerate([1e-6]):
        RidgeRegressionModel = Ridge(alpha=alpha)
        RidgeRegressionModel.fit(train_rates, train_behavior)
        predicted_behavior = RidgeRegressionModel.predict(val_rates)
        # r2 score
        from sklearn.metrics import r2_score

        r2 = r2_score(val_behavior, predicted_behavior)
        r2s_per_alpha.append(r2)
        return {
            "predicted_val_behavior": rearrange(
                predicted_behavior, "(b l) c -> b c l", b=bs_val
            ),
            "real_val_behavior": rearrange(val_behavior, "(b l) c -> b c l", b=bs_val),
        }


