
import torch

import torch.nn as nn

class Gatelayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Gatelayer, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        gates = torch.nn.Softmax(dim=-1)(self.W(x))
        return gates