import os

import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scvi.nn._base_components import Encoder, DecoderMeth

assert pyro.__version__.startswith('1.7.0')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

class methVAE(nn.Module):
    def __init__(self, input_size, bottleneck_size=8, hidden_size=16, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(n_input = input_size,
                              n_output = bottleneck_size,
                              n_hidden = hidden_size)
        self.decoder = DecoderMeth(n_input = bottleneck_size,
                               n_output = input_size,
                               n_hidden = hidden_size)
        
        if use_cuda:
            # for GPU support
            self.cuda()
            
        self.use_cuda = use_cuda
        self.bottleneck_size = bottleneck_size
        self.input_size = input_size
        
    def model(self, dat):
        x = dat['X']
        pyro.module("decoder", self.decoder)
        mean_nb1, alpha_min, alpha_max, disp_nb1  = torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.zeros(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1])))
        with pyro.plate("data", x.shape[0]):
            gauss_mean = torch.zeros(torch.Size((x.shape[0], self.bottleneck_size)))
            gauss_sd = torch.ones(torch.Size((x.shape[0], self.bottleneck_size)))
            bottleneck = pyro.sample("bottleneck", dist.Normal(gauss_mean, gauss_sd).to_event(1))
            pi_reconstructed = self.decoder(bottleneck)
            mean_nb1_sampled = pyro.sample("mean_nb1", dist.LogNormal(mean_nb1,1).to_event(1))
            alpha_sampled = pyro.sample("alpha", dist.Uniform(alpha_min, alpha_max).to_event(1))
            disp_nb1_sampled = pyro.sample("disp_nb1", dist.Normal(disp_nb1,1).to_event(1))
            mean_nb2_sampled = mean_nb1_sampled*alpha_sampled
            pi_reconstructed = pyro.sample("pi",dist.Bernoulli(pi_reconstructed).to_event(1))>0.5
            pyro.sample("obs", 
                       dist.MaskedMixture(pi_reconstructed,dist.NegativeBinomial(mean_nb1_sampled, disp_nb1_sampled),dist.Poisson(mean_nb2_sampled)).to_event(1),
                       obs=x)

    def guide(self, dat):
        x = dat['X']
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            gauss_mean, gauss_scale, latent = self.encoder(x)
            pyro.sample("latent", dist.Normal(gauss_mean, gauss_scale).to_event(1))
            
