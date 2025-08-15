import torch
from .emb_fusion import EmbeddingFusionGate
from .emb_fusion import SeNet

class ItemDnn(torch.nn.Module):
    def __init__(self, id_emb_dim, feats_emb_dim, hidden_units):
        super(ItemDnn, self).__init__()
        self.feats_fusion_linear = torch.nn.Linear(feats_emb_dim, hidden_units)
        self.feats_fusion_layer = SeNet(in_channels= feats_emb_dim//hidden_units, hidden_dim=2*hidden_units)

        self.feats_act = torch.nn.GELU()
        self.gate_fusion = EmbeddingFusionGate(id_emb_dim+hidden_units, hidden_units)


    def forward(self, id_emb, feats_emb):

        # feat_emb = self.feats_fusion_linear(feats_emb)
        feat_emb = self.feats_fusion_layer(feats_emb)
        x = self.gate_fusion(id_emb, feat_emb)

        
        return x
    

