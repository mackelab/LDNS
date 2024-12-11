"""
Implements LFADS sampling functionality for generating synthetic neural data.
Provides classes and utilities for sampling from trained LFADS models,
including latent factors, rates, spikes and behavior.
"""

import pickle
import torch
from types import SimpleNamespace
from typing import Union, Optional
from pathlib import Path

from lfads_modules.decoder import Decoder
from lfads_modules.priors import MultivariateNormal
from lfads_modules.readout import FanInLinear


class LFADSSampler:
    """LFADS sampler for generating synthetic neural data"""
    
    def __init__(
        self,
        decoder: Decoder,
        ic_prior: MultivariateNormal,
        readout: FanInLinear,
        behavior_readout: Optional[torch.nn.Linear] = None,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ):
        # ensure no controller is used
        assert not decoder.rnn.cell.use_con, "LFADS generator with controller not supported"
        
        # setup device and move models
        self.device = device
        self.decoder = decoder.to(device)
        self.ic_prior = ic_prior.to(device)
        self.readout = readout.to(device)
        
        # set all models to eval mode
        self.decoder.eval()
        self.ic_prior.eval() 
        self.readout.eval()
        
        # setup optional behavior readout
        if behavior_readout is not None:
            self.behavior_readout = behavior_readout.to(device)
            self.behavior_readout.eval()
        else:
            self.behavior_readout = None
            
        # setup random generator
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
    
    def set_seed(self, seed: int = None):
        """Set random seed for reproducibility"""
        if seed is not None:
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def sample_prior(self, n: int):
        """Sample from initial condition prior"""
        mean = self.ic_prior.mean.data
        std = torch.exp(0.5 * self.ic_prior.logvar)
        return torch.normal(
            mean.view(1, -1).expand(n, -1), 
            std.view(1, -1).expand(n, -1), 
            generator=self.generator,
        )
    
    @torch.no_grad()
    def sample_latents(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
    ):
        """Sample latent factors"""
        ic_samp = self.sample_prior(n)
        ci = torch.zeros(n, t, 0, device=self.device)
        if ext_input is None:
            ext_input = torch.zeros(n, t, self.decoder.hparams.ext_input_dim, device=self.device)
        self.decoder.hparams.recon_seq_len = t
        factors = self.decoder(ic_samp, ci, ext_input)[-1]
        return factors
    
    @torch.no_grad()
    def sample_observations(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
        return_rates: bool = False,
    ):
        """Sample spikes and optionally rates"""
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        spikes = torch.poisson(rates, generator=self.generator)
        if return_rates:
            return spikes, rates
        return spikes
    
    @torch.no_grad()
    def sample_behavior(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
    ):
        """Sample behavior predictions"""
        if self.behavior_readout is None:
            raise AttributeError(
                "`LFADSSampler.behavior_readout` is not assigned. "
            )
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        behavior = self.behavior_readout(rates)
        return behavior
    
    @torch.no_grad()
    def sample_everything(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
        include_behavior: bool = False,
    ):
        """Sample all variables - spikes, rates, factors and optionally behavior"""
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        spikes = torch.poisson(rates, generator=self.generator)     
        returns = (spikes, rates, factors)
        if include_behavior and self.behavior_readout is None:
            raise AttributeError(
                "`LFADSSampler.behavior_readout` is not assigned. "
            )
        elif include_behavior:
            behavior = self.behavior_readout(rates)
            returns += (behavior,)
        return returns


def load_lfads_sampler(
    file_path: Union[str, Path], 
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
):
    """Load saved LFADS sampler from file"""
    with open(file_path, "rb") as f:
        sampler_state_dict = pickle.load(f)
    # load prior
    ic_prior_state_dict = sampler_state_dict["ic_prior"]
    ic_prior = MultivariateNormal(0.0, 0.1, shape=ic_prior_state_dict["mean"].shape[0])
    ic_prior.load_state_dict(ic_prior_state_dict)
    # load decoder
    decoder_state_dict = sampler_state_dict["decoder"]
    hps = SimpleNamespace(  # hacky solution to not preserving full model config
        dropout_rate=0.0,
        ic_dim=decoder_state_dict["ic_to_g0.weight"].shape[1],
        gen_dim=decoder_state_dict["ic_to_g0.weight"].shape[0],
        con_dim=decoder_state_dict["con_h0"].shape[1],
        recon_seq_len=0,
        ext_input_dim=decoder_state_dict["rnn.cell.gen_cell.weight_ih"].shape[1],
        co_dim=0,
        cell_clip=5.0,
        fac_dim=decoder_state_dict["rnn.cell.fac_linear.weight"].shape[0],
        ci_enc_dim=0,
    )
    decoder = Decoder(hps)
    decoder.load_state_dict(decoder_state_dict)
    # load readout
    readout_state_dict = sampler_state_dict["readout"]
    readout = FanInLinear(readout_state_dict['weight'].shape[1], readout_state_dict['weight'].shape[0])
    readout.load_state_dict(readout_state_dict)
    # load behavior readout
    if "behavior_readout" in sampler_state_dict:
        behavior_readout_state_dict = sampler_state_dict["behavior_readout"]
        behavior_readout = torch.nn.Linear(
            behavior_readout_state_dict['weight'].shape[1], 
            behavior_readout_state_dict['weight'].shape[0],
        )
        behavior_readout.load_state_dict(behavior_readout_state_dict)
    else:
        behavior_readout = None
    # create sampler
    sampler = LFADSSampler(
        decoder=decoder,
        ic_prior=ic_prior,
        readout=readout,
        behavior_readout=behavior_readout,
        device=device,
        seed=seed,
    )
    return sampler


def save_lfads_sampler(sampler: LFADSSampler, file_path: Union[str, Path]):
    """Save LFADS sampler to file"""
    sampler_state_dict = dict(
        ic_prior=sampler.ic_prior.state_dict(),
        decoder=sampler.decoder.state_dict(),
        readout=sampler.readout.state_dict(),
    )
    if sampler.behavior_readout is not None:
        sampler_state_dict["behavior_readout"] = sampler.behavior_readout.state_dict()  # Fixed tuple bug
    with open(file_path, "wb") as f:
        pickle.dump(sampler_state_dict, f)