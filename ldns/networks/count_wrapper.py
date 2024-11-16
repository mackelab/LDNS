# %%
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CountWrapper(nn.Module):

    """Wrapper module for converting real valued input/output to
    spike trains and rates respectively
    """


    def __init__(self, ae_net):
        super().__init__()
        self.ae_net = ae_net

    def forward(self, x):
        # x: [B, C_in, L]
        # rates: [B, C_in, L]
        logrates, z = self.ae_net(x)
        return F.softplus(logrates), z

    def encode(self, x):
        return self.ae_net.encode(x)

    def decode(self, z):
        return F.softplus(self.ae_net.decode(z))

# %%
if __name__ == "__main__":

    # test CountWrapper
    from ldns.networks import AutoEncoder

    autoencoder = AutoEncoder(C_in=8, C=256, C_latent=8, L=500, kernel="s4")
    count_wrapper = CountWrapper(autoencoder)
    x = torch.randn(10, 8, 500)
    rates, z = count_wrapper(x)
    print(rates.shape, z.shape)
    print(count_wrapper)
    print("CountWrapper test passed")

# %%
