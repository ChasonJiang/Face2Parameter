import math
import os
from typing import List
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super(Encoder, self).__init__()
        self.downconv_blocks = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), # 224*224 -> 112*112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ),
            nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2), # 112*112 -> 56*56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ),
            nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), # 56*56 -> 28*28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ),
            nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1), # 28*28 -> 14*14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ),
            nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1), # 14*14 -> 7*7
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ),
            nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1), # 7*7 -> 4*4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ),
            # nn.Sequential(
            # nn.Conv2d(512, 1024, 4), # 4*4 -> 1*1
            # nn.BatchNorm2d(1024),
            # nn.ReLU(),
            # ),

        ])
        
        self.resconv_blocks = nn.ModuleList([
            # 64*112*112
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), 
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            # 128*56*56
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1), 
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            # 256*28*28
            nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1), 
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            # 512*14*14
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            # 512*7*7
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            # 512*4*4
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            # # 1024*1*1
            # nn.Sequential(
            #     nn.Conv2d(1024, 1024, 3, 1, 1), 
            #     nn.BatchNorm2d(1024),
            #     nn.ReLU(),
            # ),
        ])
        self.final_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 2*hidden_dim),
        )

        # self.mu_layer = nn.Linear(hidden_dim, hidden_dim)
        # self.logvar_layer = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x:torch.Tensor):
        for downconv_block, resconv_block in zip(self.downconv_blocks, self.resconv_blocks):
            x = downconv_block(x)
            res = x
            x = resconv_block(x) + res
        x = self.final_layer(x)
        mu, logvar = x.split(self.hidden_dim, dim=-1)
        # mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super(Decoder, self).__init__()
        self.upconv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 512*4*4),
                nn.Unflatten(1, (512, 4, 4)), # 4*4
                nn.ReLU(),
            ),
            # nn.Sequential(
            #     nn.ConvTranspose2d(512, 512, 4, 2, 1), # 2*2 -> 4*4
            #     nn.BatchNorm2d(512),    
            #     nn.ReLU(),
            # ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, 2, 1), # 4*4 -> 7*7
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1), # 7*7 -> 14*14
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 4, 2, 1), # 14*14 -> 28*28
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 4, 2, 1), # 28*28 -> 56*56
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1), # 56*56 -> 112*112
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1), # 112*112 -> 224*224
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        ])



        self.resconv_blocks = nn.ModuleList([
            # # 2*2
            # nn.Sequential(
            #     nn.Conv2d(512, 512, 1, 1, 0), 
            #     nn.BatchNorm2d(512),
            #     nn.ReLU(),
            # ),
            # 4*4
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1), 
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            # 7*7
            nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1), 
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            # 14*14
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1), 
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            # 28*28
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1), 
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            # 56*56
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1), 
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
            # 112*112
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1), 
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            # 224*224
            nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1), 
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),

        ])


        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), # 224*224 -> 224*224
            # nn.PixelShuffle(2), # 224*224 -> 448*448
            # nn.Conv2d(8, 8, 3, 2, 1), # 448*448 -> 224*224
            nn.ReLU(),
            nn.Conv2d(32, 3, 1, 1, 0), # 224*224 -> 224*224
            nn.Tanh(),
        )

    def forward(self, x:torch.Tensor):
        for upconv_block, resconv_block in zip(self.upconv_blocks, self.resconv_blocks):
            x = upconv_block(x)
            res = x
            x = resconv_block(x) + res
        x = self.final_conv(x)
        return x
    

class VAEModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super(VAEModel, self).__init__()
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x:torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu
    
    def compute_loss(self, x:torch.Tensor, x_hat:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor, beta=1.0):
        # reconstruction_loss = torch.mean((x-x_hat)**2)
        reconstruction_loss = nn.functional.mse_loss(x_hat, x)
        # reconstruction_loss = nn.functional.l1_loss(x_hat, x)
        kl_divergence = beta*torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        return reconstruction_loss, kl_divergence

        
    def sample(self, num_samples=1, device:torch.device='cpu'):
        z = torch.randn(num_samples, self.hidden_dim, device=device)
        x_hat = self.decoder(z)
        return x_hat
    
    def decode(self, z:torch.Tensor):
        x_hat = self.decoder(z)
        return x_hat



