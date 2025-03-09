import math
import os
from typing import List
import torch
import torch.nn as nn
import torchvision

class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        self.blocks = nn.Sequential(
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
                nn.Conv2d(128, 256, 3, 2, 1), # 56*56 -> 28*28
                nn.BatchNorm2d(256),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1), # 28*28 -> 14*14
                nn.BatchNorm2d(512),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 1024, 3, 2, 1), # 14*14 -> 7*7
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 1024, 7,), # 7*7 -> 1*1
                nn.BatchNorm2d(1024),
                nn.ReLU(),
            ),
        )
        self.proj_layer = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor):
        feat = []
        for block in self.blocks:
            x = block(x)
            feat.append(x)

        x = self.proj_layer(x.flatten(1))
        x = self.sigmoid(x)

        return x, feat
    
    def compute_loss(self, pred:torch.Tensor, is_real:bool):
        if is_real:
            loss = torch.mean((1-pred)**2)
        else:
            loss = torch.mean(pred**2)
        # if is_real:
        #     loss = torch.log(pred)
        # else:
        #     loss = torch.log(1-pred)
        return loss
    
    def compute_feat_loss(self, feat_real:List[torch.Tensor], feat_fake:List[torch.Tensor]):
        loss = 0
        for i in range(len(feat_real)):
            loss += torch.mean((feat_real[i]-feat_fake[i])**2)
        return loss


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
    
class MLP(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int=1024):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x:torch.Tensor):
        x = self.layers(x)
        return x
    

class PriorEncoder(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super(PriorEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),            
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, output_dim*2),
        )

    def forward(self, x:torch.Tensor):
        x = self.layers(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return mu, logvar
    
class PriorDecoder(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super(PriorDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x:torch.Tensor):
        x = self.layers(x)
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
    

class VAEWithCondition(nn.Module):
    def __init__(self, hidden_dim=64, cond_input_dim=59, cond_output_dim=512):
        super(VAEWithCondition, self).__init__()
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.prior_encoder = PriorEncoder(cond_input_dim, cond_output_dim)
        self.prior_decoder = PriorDecoder(cond_output_dim, cond_input_dim)
        self.cond_output_dim = cond_output_dim
        self.hidden_dim = hidden_dim

    def forward(self, x:torch.Tensor, cond:torch.Tensor):
        mu, logvar = self.encoder(x)
        mu_p, logvar_p = self.prior_encoder(cond)
        z = self.reparameterize(mu, logvar)
        # z_p = self.reparameterize(mu_p, logvar_p)
        cond_hat = self.prior_decoder(z[:, :self.cond_output_dim])
        x_hat = self.decoder(z)
        return x_hat, cond_hat, mu, logvar, mu_p, logvar_p
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu
    
    def compute_loss(self, x:torch.Tensor, x_hat:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor, cond:torch.Tensor, cond_hat:torch.Tensor, mu_p:torch.Tensor, logvar_p:torch.Tensor, alpha:float=1.0, beta:float=1.0, gamma:float=1.0):
        recon_loss1 = nn.functional.mse_loss(x_hat, x)
        recon_loss2 = nn.functional.mse_loss(cond_hat, cond)
        # reconstruction_loss = nn.functional.l1_loss(x_hat, x)

        mu1 = mu[:, :self.cond_output_dim]
        logvar1 = logvar[:,:self.cond_output_dim]

        mu2 = mu[:, self.cond_output_dim:]
        logvar2 = logvar[:,self.cond_output_dim:]

        kl_loss1 = (logvar_p - logvar1) + ((2*logvar1).exp() + (mu1 - mu_p)**2) / (2 * (2*logvar_p).exp()) - 0.5
        kl_loss1 = torch.mean(kl_loss1)
        kl_loss2 = torch.mean(-0.5 * torch.sum(1 + logvar2 - mu2**2 - logvar2.exp(), dim=1), dim=0)
        return recon_loss1, recon_loss2, kl_loss1, kl_loss2

        
    def sample(self, num_samples=1, device:torch.device='cpu'):
        z = torch.randn(num_samples, self.hidden_dim, device=device)
        x_hat = self.decoder(z)
        return x_hat
    
    def decode(self, z:torch.Tensor):
        x_hat = self.decoder(z)
        return x_hat
