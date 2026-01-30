from pathlib import Path

from omegaconf import OmegaConf

from ldns.networks import AutoEncoder, Denoiser
from ldns.utils.utils import count_parameters


def _get_signal_length(dataset_cfg):
    if "signal_length" in dataset_cfg:
        return dataset_cfg.signal_length
    if "max_seqlen" in dataset_cfg:
        return dataset_cfg.max_seqlen
    raise ValueError("Dataset config must define signal_length or max_seqlen.")


def _load_config(path):
    return OmegaConf.load(path)


def _init_autoencoder(cfg):
    return AutoEncoder(
        C_in=cfg.model.C_in,
        C=cfg.model.C,
        C_latent=cfg.model.C_latent,
        L=_get_signal_length(cfg.dataset),
        kernel=cfg.model.get("kernel", "s4"),
        num_blocks=cfg.model.num_blocks,
        num_blocks_decoder=cfg.model.get(
            "num_blocks_decoder", cfg.model.num_blocks
        ),
        num_lin_per_mlp=cfg.model.get("num_lin_per_mlp", 2),
        bidirectional=cfg.model.get("bidirectional", True),
    )


def _init_denoiser(cfg, dataset_cfg):
    return Denoiser(
        C_in=cfg.denoiser_model.C_in,
        C=cfg.denoiser_model.C,
        L=_get_signal_length(dataset_cfg),
        kernel=cfg.denoiser_model.get("kernel", "s4"),
        num_blocks=cfg.denoiser_model.num_blocks,
        bidirectional=cfg.denoiser_model.get("bidirectional", True),
    )


def report_parameter_counts():
    repo_root = Path(__file__).resolve().parents[2]
    conf_dir = repo_root / "conf"
    datasets = {
        "Lorenz": {
            "autoencoder": "autoencoder-Lorenz_z=8.yaml",
            "diffusion": "diffusion_Lorenz.yaml",
        },
        "Human": {
            "autoencoder": "autoencoder-human.yaml",
            "diffusion": "diffusion_human.yaml",
        },
        "Monkey": {
            "autoencoder": "autoencoder-monkey_z=16.yaml",
            "diffusion": "diffusion_monkey_unconditional.yaml",
        },
    }
    results = {}
    for name, paths in datasets.items():
        ae_cfg = _load_config(conf_dir / paths["autoencoder"])
        diff_cfg = _load_config(conf_dir / paths["diffusion"])
        diff_dataset_cfg = diff_cfg.get("dataset") or ae_cfg.dataset
        autoencoder = _init_autoencoder(ae_cfg)
        denoiser = _init_denoiser(diff_cfg, diff_dataset_cfg)
        results[name] = {
            "autoencoder": count_parameters(autoencoder),
            "diffusion": count_parameters(denoiser),
        }

    for name, counts in results.items():
        print(
            f"{name}: AutoEncoder params={counts['autoencoder']:,} | "
            f"Denoiser params={counts['diffusion']:,}"
        )
    return results


if __name__ == "__main__":
    report_parameter_counts()
