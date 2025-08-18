import torch
import torch.nn.functional as F


class GLUFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, out_units, droupout_rate):
        super(GLUFeedForward, self).__init__()
        self.W_u = torch.nn.Linear(hidden_units, hidden_units)
        self.W_v = torch.nn.Linear(hidden_units, hidden_units)
        self.W_o = torch.nn.Linear(hidden_units, out_units)
        self.act_u = torch.nn.SiLU()
        self.dropout1 = torch.nn.Dropout(p=droupout_rate)
        self.dropout2 = torch.nn.Dropout(p=droupout_rate)
        self.dropout = torch.nn.Dropout(p=droupout_rate)

    def forward(self, inputs):
        u = self.act_u(self.W_u(inputs))
        v = self.W_v(inputs)
        outputs = self.W_o(u * v)
        outputs = self.dropout(outputs)
        return outputs
        




class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, out_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.act = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(hidden_units, out_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.act(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs