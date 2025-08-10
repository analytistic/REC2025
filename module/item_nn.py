import torch


class ItemDnn(torch.nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(ItemDnn, self).__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_units)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.act(self.linear(x))
        return x


