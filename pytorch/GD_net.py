import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module): # Leaky ReLU
    def __init__(self, input_size, hidden_size):
        pass

    def forward(self, x):
        return x
class Generator(nn.Module): # ReLU everywhere, tanh at last layer (output of G -> input of D)
    def __init__(self, input_size, hidden_size):
        pass

    def forward(self, x):
        return x