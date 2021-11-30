import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
                
    def forward(self, x):
        output = self.model(x)
        output = output.unsqueeze(1)
        return(output)
        
    def reset_parameters(self) :
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
    
        self.model = nn.Sequential(
            nn.Linear(latent_dim,64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, z):
        N = z.size(0)
        output = self.model(z)
        output = output.view(N, 20, 6)
        return(output)
        
    def reset_parameters(self) :
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        output = self.model(z)
        return(output)
        
    def reset_parameters(self) :
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
