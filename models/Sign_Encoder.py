import torch.nn as nn
import torch.nn.functional as F

from models.metaformer.meta_model import MetaFormer
from configs.standards.standard_meta_model_zero_config import get_sign_encoder
from models.spatial_models.frame_models.dino_adaptor_model import Model

class SignEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model_name, sign_model_params, dim_model = get_sign_encoder()
        spatial_params = sign_model_params['spatial_params']
        self.spatial_model = Model(**spatial_params, out_dim=dim_model)

        encoder_name = sign_model_params['encoder_name']
        encoder_params = sign_model_params['encoder_params']
        self.encoder = MetaFormer(**encoder_params)

    def forward(self, x):
        y, mask,_ = self.spatial_model(x)
        outputs = self.encoder(y, mask)
        return outputs