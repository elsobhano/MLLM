import torch
import torch.nn as nn
# from models.model_utils.masked_norm import MaskedNorm


class Model(nn.Module):
    def __init__(self, d_model=None, out_dim=None):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.mapper = nn.Linear(d_model, out_dim)

    def forward(self, x, mask):
        return {"x": self.mapper(x), "mask": mask}
