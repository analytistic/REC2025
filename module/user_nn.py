import torch
from .emb_fusion import EmbeddingFusionGate, SeNet


class UserDnn(torch.nn.Module):
    """
    查看了一下gate， 发现更多让id——emb流过，feat——emb可能太杂乱，所以尝试在feat——emb上使用senet。
    - id_emb: bs, len, id_emb_dim
    - feats_emb: bs, len, feat_num, emb_dim

    """
    def __init__(self, id_emb_dim, feats_emb_dim, hidden_units):
        super(UserDnn, self).__init__()
        self.feats_fusion_linear = torch.nn.Linear(feats_emb_dim, hidden_units)
        self.feats_fusion_layer = SeNet(in_channels= feats_emb_dim//hidden_units, hidden_dim=2*hidden_units)
        self.gate_fusion = EmbeddingFusionGate(id_emb_dim+hidden_units, hidden_units)


    def forward(self, id_emb, feats_emb):
        feat_emb = self.feats_fusion_layer(feats_emb)
        x = self.gate_fusion(id_emb, feat_emb)

        
        return x
    


