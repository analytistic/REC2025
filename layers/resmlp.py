import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.dropout(self.fc2(self.act(self.fc1(x)))) + x
        return x
        


class RESMLP(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout=0.1, num_blocks=1):
        super(RESMLP, self).__init__()

        self.blocks = nn.ModuleList(
            [
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            ]
        ) + nn.ModuleList([
            MLP(hid_dim, hid_dim, dropout)
            for i in range(num_blocks)
        ])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

