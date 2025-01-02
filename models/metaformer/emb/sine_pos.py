import torch
import torch.nn as nn
import math
# from transformers import PositionalEncoding

class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_len=5000
        d_model = config["dim_model"]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Model(nn.Module):
    def __init__(self, in_dim, d_model, pos_config, num_embed_tokens = None):
        super().__init__()
        if in_dim != d_model:
            self.linear = nn.Linear(in_dim, d_model)
        else:
            self.linear = nn.Identity()
        self.pos_emb = PositionalEncoding(pos_config)
        self.num_embed_tokens = num_embed_tokens
        if num_embed_tokens:
            self.embed_tokens = torch.nn.Embedding(num_embed_tokens, d_model)
            self.embed_tokens.weight.data.normal_(mean=0,std=0.7)
        else:
            self.embed_tokens = None


    def forward(self, x, mask):
        if self.embed_tokens is not None:
            embeds = self.embed_tokens.weight
            # ADDED [:,:x.shape[1]] to keep it at 2** shape, shouldn't impact results as padding is large
            mask = torch.cat([torch.ones(mask.shape[0],self.num_embed_tokens).bool().to(mask.device), mask],dim=1)[:,:x.shape[1]]
            x = torch.cat([embeds.unsqueeze(0).expand(mask.shape[0], *embeds.shape), x], dim=1)[:,:x.shape[1],:]
        
        x = self.linear(x)
        x = self.pos_emb(x)
        return x, mask
