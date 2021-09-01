import os

import numpy as np
import torch
import torch.nn as nn
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scvi.nn._base_components import Encoder, DecoderMeth

assert pyro.__version__.startswith('1.7.0')
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

class methVAE(nn.Module):
    def __init__(self,
                 input_size,
                 bottleneck_size=8,
                 hidden_size=16,
                 use_cuda=False,
                scale_factor=1.0):
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
        self.scale_factor = scale_factor
        self.mean_nb1 = torch.nn.Parameter(torch.randn(input_size))
        self.mean_nb2 = torch.nn.Parameter(torch.randn(input_size))
        self.alpha = torch.nn.Parameter(torch.randn(input_size))
        self.disp_nb1 = torch.nn.Parameter(torch.randn(input_size))
        self.disp_nb2 = torch.nn.Parameter(torch.randn(input_size))
        self.eps = 1e-5
        
    def model(self, dat):
        x = dat['X']
#        pyro.module("decoder", self.decoder)
        mean_nb1, mean_nb2, alpha, disp_nb1, disp_nb2 = torch.ones(x.shape[1]), torch.ones(x.shape[1]), torch.ones(x.shape[1]), torch.ones(x.shape[1]), torch.ones(x.shape[1])-.1
        pi_init   = torch.ones(torch.Size((x.shape[0], x.shape[1])))
        mean_nb1 = pyro.param("mean_nb1", mean_nb1)
        disp_nb2 = pyro.param("disp_nb2", disp_nb2)
        disp_nb1 = pyro.param("disp_nb1", disp_nb1, constraint=constraints.greater_than(disp_nb2))
        mean_nb2 = pyro.param("mean_nb2", mean_nb2)
        pi_p = pyro.param("pi",pi_init)
        with pyro.plate("data", x.shape[0]):
#            foreground = pyro.sample("foreground", dist.NegativeBinomial(mean_nb1,disp_nb1).to_event(1))
#            background = pyro.sample("background", dist.NegativeBinomial(mean_nb2,disp_nb2).to_event(1))
#            pi = pyro.param("pi",pi)
            disp_nb1_s = pyro.sample("dispersion_nb1", dist.Exponential(disp_nb1).to_event(1))
            disp_nb2_s = pyro.sample("dispersion_nb2", dist.Exponential(disp_nb2).to_event(1))
            pi_reconconstructed = pyro.sample("pi_s", dist.Bernoulli(pi_p).to_event(1))
#                bottleneck = pyro.sample("bottleneck_%d" % ampli, dist.Normal(gauss_mean, gauss_sd).to_event(1))
            pi_reconconstructed = pi_reconconstructed>0.5
#            pyro.sample("obs",
#                        dist.NegativeBinomial(mean_nb1,disp_nb1).to_event(1),
#                        obs=x)
#            pyro.sample("obs_neg",
#                        dist.NegativeBinomial(mean_nb2,disp_nb2).to_event(1),
#                        obs=x)
#                foreground = pyro.sample("foreground_%d" % ampli, dist.NegativeBinomial(mean_nb1, disp_nb1))
#                background = pyro.sample("background_%d" % ampli, dist.NegativeBinomial(mean_nb2, (disp_nb1*alpha)+self.eps))
            dist_res = dist.MaskedMixture(pi_reconconstructed,
                          dist.NegativeBinomial(mean_nb1,disp_nb1_s),
                          dist.NegativeBinomial(mean_nb2,disp_nb2_s))
            pyro.sample("obs", 
                       dist_res.to_event(1),
                       obs=x)

    def guide(self, dat):
        x = dat['X']
#        pyro.module("encoder", self.encoder)
        mean_nb1, mean_nb2, alpha, disp_nb1, disp_nb2 = torch.ones(x.shape[1]), torch.ones(x.shape[1]), torch.ones(x.shape[1]), torch.ones(x.shape[1]), torch.ones(x.shape[1])-.1
        pi_init   = torch.ones(torch.Size((x.shape[0], x.shape[1])))
        mean_nb1 = pyro.param("mean_nb1", mean_nb1)
        mean_nb2 = pyro.param("mean_nb2", mean_nb2)
#        mean_nb1 = mean_nb1.expand(x.shape[0], x.shape[1]).t()
#        mean_nb2 = mean_nb2.expand(x.shape[0], x.shape[1]).t()
#        alpha = pyro.param("alpha", alpha, constraint=constraints.interval(0,1))
#        alpha = alpha.expand(x.shape[0], x.shape[1]).t()
        disp_nb2 = pyro.param("disp_nb2", disp_nb2)
        disp_nb1 = pyro.param("disp_nb1", disp_nb1, constraint=constraints.greater_than(disp_nb2))
        pi_p = pyro.param("pi",pi_init)
#        disp_nb1 = disp_nb1.expand(x.shape[0], x.shape[1]).t()
        with pyro.plate("data", x.shape[0]):
            pyro.sample("dispersion_nb1", dist.Exponential(disp_nb1).to_event(1))
            pyro.sample("dispersion_nb2", dist.Exponential(disp_nb2).to_event(1))
#            foreground = pyro.sample("foreground",dist.NegativeBinomial(mean_nb1,disp_nb1).to_event(1))
#            background = pyro.sample("background",dist.NegativeBinomial(mean_nb2,disp_nb2).to_event(1))
            pi_reconstructred = pyro.sample("pi_s",dist.Bernoulli(pi_p).to_event(1))
            pi_reconstructred = pi_reconstructred>0.5
#                pyro.sample("bottleneck_%d" % ampli, dist.Normal(gauss_mean, gauss_scale).to_event(1))
#                pyro.sample("foreground_%d" % ampli, dist.NegativeBinomial(mean_nb1, disp_nb1))
#                pyro.sample("background_%d" % ampli, dist.NegativeBinomial(mean_nb2, (disp_nb1*alpha)+self.eps))
            
    def generate_counts(self, x):
        gauss_mean, gauss_scale, latent = self.encoder(x)
        latent = dist.Normal(gauss_mean, gauss_scale).sample()
        pi_reconstructed = self.decoder(latent)
        return pi_reconstructed