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
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            mean_nb1, alpha_min, alpha_max, disp_nb1, disp_nb2  = torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.zeros(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1])))
            gauss_mean = torch.zeros(torch.Size((x.shape[0], self.bottleneck_size)))
            gauss_sd = torch.ones(torch.Size((x.shape[0], self.bottleneck_size)))
            bottleneck = pyro.sample("bottleneck", dist.Normal(gauss_mean, gauss_sd).to_event(1))
#            mean_nb1_sampled = pyro.sample("mean_nb", dist.HalfNormal(mean_nb1).to_event(1))
#            alpha_sampled = pyro.sample("alpha", dist.Uniform(alpha_min, alpha_max).to_event(1))
#            disp_nb1_sampled = pyro.sample("disp_nb1", dist.HalfNormal(disp_nb1).to_event(1))
#            disp_nb2_sampled = pyro.sample("disp_nb2", dist.HalfNormal(disp_nb2).to_event(1))
#            print(alpha_sampled)
#            print(x.index)
            mean_nb1_sampled = pyro.param("mean_nb1", self.mean_nb1)
            mean_nb2_sampled = pyro.param("mean_nb2", self.mean_nb2)
            mean_nb1_sampled = torch.exp(mean_nb1_sampled)
            mean_nb2_sampled = torch.exp(mean_nb2_sampled)
            alpha_sampled = pyro.sample("alpha", dist.Uniform(alpha_min,alpha_max))
            alpha_sampled = torch.exp(alpha_sampled)/(1+torch.exp(alpha_sampled))
#            enzyme_activity = pyro.param("enzyme_activity", torch.nn.Parameter(torch.randn(x.shape[0])), constraint=constraints.interval(0,1))
#            enzyme_activity = enzyme_activity[range(0,x.shape[0])]
#            print(mean_nb1_sampled)
#            alpha_sampled = pyro.param("alpha", self.alpha, constraint=constraints.interval(0,1))
#            print(alpha_sampled)
#            pi_reconstructed = self.decoder(bottleneck)
#            print(mean_nb1_sampled)
#            print(mean_nb2_sampled)
            pi = torch.nn.Parameter(torch.randn(x.shape[0]))
            pi = pi.expand(x.shape[1], x.shape[0]).t()
            pi = torch.exp(pi)/(1+torch.exp(pi))
            pi_reconstructed = pyro.param("pi",dist.Bernoulli(pi).to_event(1))>0.5
            disp_nb1 = pyro.param("disp_nb1", self.disp_nb1)
            disp_nb1 = torch.exp(disp_nb1)
            #disp_nb2 = pyro.param("disp_nb2", self.disp_nb2)
            #disp_nb2 = torch.exp(disp_nb2)
            #disp_nb2 = (disp_nb1*alpha_sampled)+self.eps
            disp_nb1 = (disp_nb1)/(disp_nb1+mean_nb1_sampled)
            #disp_nb2 = (disp_nb2)/(disp_nb2+mean_nb2_sampled)
#            print(disp_nb1)
#            print(disp_nb2)
#            enzyme_activity = enzyme_activity.expand(x.shape[1], x.shape[0]).t()
#            pi_reconstructed = pi_reconstructed*(1-enzyme_activity)
#            pi_reconstructed = pi_reconstructed>0.5
            dist_res = dist.MaskedMixture(pi_reconstructed,
                          dist.NegativeBinomial(mean_nb1_sampled, disp_nb1),
                          dist.NegativeBinomial(mean_nb2_sampled, (disp_nb1*alpha_sampled)+self.eps)).to_event(1)
            pyro.sample("obs", 
                       dist_res,
                       obs=x)

    def guide(self, dat):
        x = dat['X']
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            mean_nb1, alpha_min, alpha_max, disp_nb1, disp_nb2  = torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.zeros(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1]))), torch.ones(torch.Size((x.shape[0], x.shape[1])))
            gauss_mean, gauss_scale, latent = self.encoder(x)
            pyro.param("mean_nb1", self.mean_nb1)
            pyro.param("mean_nb2", self.mean_nb2)
            pyro.sample("alpha", dist.Uniform(alpha_min,alpha_max))
#            pyro.param("enzyme_activity", torch.nn.Parameter(torch.randn(x.shape[0])), constraint=constraints.interval(0,1))
#            pyro.param("alpha", self.alpha, constraint=constraints.interval(0,1))
            pyro.param("disp_nb1", self.disp_nb1)
#            pyro.param("disp_nb2", self.disp_nb2)
#            pyro.sample("mean_nb", dist.HalfNormal(mean_nb1).to_event(1))
#            pyro.sample("alpha", dist.Uniform(alpha_min, alpha_max).to_event(1))
#            pyro.sample("disp_nb1", dist.HalfNormal(disp_nb1).to_event(1))
#            pyro.sample("disp_nb2", dist.HalfNormal(disp_nb2).to_event(1))
            pi = torch.nn.Parameter(torch.randn(x.shape[0]))
            pi = pi.expand(x.shape[1], x.shape[0]).t()
            pi = torch.exp(pi)/(1+torch.exp(pi))
            pyro.param("pi",dist.Bernoulli(pi).to_event(1))>0.5
            pyro.sample("bottleneck", dist.Normal(gauss_mean, gauss_scale).to_event(1))
            
    def generate_counts(self, x):
        gauss_mean, gauss_scale, latent = self.encoder(x)
        latent = dist.Normal(gauss_mean, gauss_scale).sample()
        pi_reconstructed = self.decoder(latent)
        return pi_reconstructed