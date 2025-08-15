from torch import nn
import torch




class EmbeddingFusionGate(nn.Module):
    def __init__(self, cat_emb_dim, fusion_dim):
        super().__init__()
        self.gate = nn.Linear(cat_emb_dim, fusion_dim)  


    def forward(self, id_emb, feat_emb):

        g = torch.sigmoid(self.gate(torch.cat([id_emb, feat_emb], dim=-1)))
        output = id_emb * g + feat_emb * (1 - g)


        return output
    
class SeNet(nn.Module):
    """
    feats_emb: bs, len, num, dim
    return: bs, len, dim
    
    """
    def __init__(self, in_channels, hidden_dim):
        super(SeNet, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        bs, len, num, dim = x.shape
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

        reweight = self.excitation(self.pool(x).squeeze(-1))
        return torch.sum(x * reweight.unsqueeze(-1), dim=-2).reshape(bs, len, dim)

