import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration
from models.spatial_models.frame_models.dino_adaptor_model import Model
from timm.models.layers import DropPath
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn.init import constant_
from models.utils import local_1d_pattern
import numpy as np
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torchvision
import math
from psp_head import HeadModel
PAD_IDX = 1

def make_resnet(name='resnet18', resnet_path=None):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)


    state_dict = torch.load(resnet_path, map_location='cpu', weights_only=True)

    # Load the weights into the model
    model.load_state_dict(state_dict)
    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class dino(nn.Module):
    def __init__(self, dino_path):
        super().__init__()
        # self.resnet = make_resnet(name='resnet18', resnet_path=resnet_path)
        dino_params = {
            "ckpt_dir": dino_path,
            "trainable_names": [],
            "adaptor_layers": list(np.arange(9, 12, 1)),
            "adapt_params": {
                "w_lora": True,
                "w_lora_ff": True,
                "lora_rank": 4,
                "lora_drop": 0.1,
                "lora_a": 4.0,
                "rng_init": False,
            },
            "out_dim": 512,
        }
        self.dino = Model(**dino_params)

    def forward(self, x, lengths):
        y, mask, _ = self.dino(x, lengths)
        return y, mask

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.adinoppend(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        self.dim = dim
        # Create inverse frequency bands for half the dimension
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        sinusoid = torch.einsum("i,j->ij", position, inv_freq)
        self.register_buffer("sin", sinusoid.sin().unsqueeze(1).unsqueeze(1))  # [seq_len, 1, 1, dim//2]
        self.register_buffer("cos", sinusoid.cos().unsqueeze(1).unsqueeze(1))  # [seq_len, 1, 1, dim//2]

    def forward(self, q, k, seq_len):
        # q, k shape: [batch, heads, seq_len, head_dim]
        
        # Take required sequence length of position encodings
        sin = self.sin[:seq_len]  # [seq_len, 1, 1, dim//2]
        cos = self.cos[:seq_len]  # [seq_len, 1, 1, dim//2]
        
        # First, reshape q and k to split last dimension in half
        q_split = q.reshape(*q.shape[:-1], -1, 2)  # [batch, heads, seq_len, dim//2, 2]
        k_split = k.reshape(*k.shape[:-1], -1, 2)  # [batch, heads, seq_len, dim//2, 2]
        
        # Handle dimensions for broadcasting
        sin = sin.expand(seq_len, q.size(0), q.size(1), q.size(-1)//2)  # [seq_len, batch, heads, dim//2]
        cos = cos.expand(seq_len, q.size(0), q.size(1), q.size(-1)//2)  # [seq_len, batch, heads, dim//2]
        
        # Permute dimensions for proper broadcasting
        sin = sin.permute(1, 2, 0, 3)  # [batch, heads, seq_len, dim//2]
        cos = cos.permute(1, 2, 0, 3)  # [batch, heads, seq_len, dim//2]
        
        # Apply rotary embeddings
        q_rot = torch.cat([
            q_split[..., 0] * cos - q_split[..., 1] * sin,
            q_split[..., 1] * cos + q_split[..., 0] * sin
        ], dim=-1)
        
        k_rot = torch.cat([
            k_split[..., 0] * cos - k_split[..., 1] * sin,
            k_split[..., 1] * cos + k_split[..., 0] * sin
        ], dim=-1)
        
        return q_rot, k_rot

# The rest of LightweightTemporalTransformer remains the same
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=300, windfow_size=7,
                use_layer_scale=True, layer_scale_init_value=1e-5,
                ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = windfow_size

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Linear layers for Q, K, V
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        if isinstance(self.out_proj, nn.Linear) and self.out_proj.bias is not None:
            constant_(self.out_proj.bias, 0.0)
        # Rotary positional embeddings
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(d_model), requires_grad=True)


    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, embed_dim = query.shape

        # Linear projections
        query = self.query_proj(query)  # (batch_size, seq_len, embed_dim)
        key = self.key_proj(key)        # (batch_size, seq_len, embed_dim)
        value = self.value_proj(value)  # (batch_size, seq_len, embed_dim)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation: (batch_size, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply rotary positional embeddings
        query, key = self.rotary_emb(query, key, seq_len)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)

        local_mask = local_1d_pattern(seq_len, self.window_size).to(scores.device)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
        # local_mask = torch.ones_like(local_mask)

        # Apply mask (if provided)
        if mask is not None:
            # Reshape mask to (batch_size, 1, 1, seq_len) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            combined_mask = local_mask * mask
            # Apply mask to scores
        else:
            combined_mask = local_mask
        # Ensure each token attends to itself
        diag_mask = torch.eye(seq_len, dtype=torch.bool, device=scores.device).unsqueeze(0).unsqueeze(0)
        combined_mask = combined_mask | diag_mask
        
        scores = scores.masked_fill(combined_mask == 0, float('-inf'))  # Replace masked positions with -inf
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

        # Final linear projection
        out = self.dropout(self.out_proj(out))

        # Apply layer scaling if enabled
        if self.use_layer_scale:
            out = out * self.layer_scale

        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, drop_path_rate):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
        self.ffn = FeedForward(d_model, d_model*4)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Drop Path (Stochastic Depth)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = x + self.drop_path(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.drop_path(ffn_output)
        x = self.norm2(x)

        # x = self.output_proj(x)

        return x, mask

class Downsampler(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, mask):
        x = torch.nn.functional.avg_pool1d(
            x.permute(0, 2, 1), 3, stride=2, padding=1
        ).permute(0, 2, 1)
        with torch.no_grad():
            mask = torch.nn.functional.avg_pool1d(
                mask.float(), 3, stride=2, padding=1, count_include_pad=False
            )
            # Ensure that there is atleast something on the encoder side.
            mask[:, 0] = 1.0
        return (x, (mask > 0.0).bool())

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # Create positional encodings
        position = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))  # (d_model // 2,)
        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.register_buffer('pe', pe)  # Register as a buffer (not a parameter)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]  # Add positional embeddings to the input
    
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, layers, drop_path_rate=0.1, d_llm=1024):
        super().__init__()
        self.pos_embedding = SinusoidalPositionalEmbedding(d_model, max_seq_len=300)
        self.layers = nn.ModuleList()
        self.num_layers = sum(layers)
        for index, num_blocks in enumerate(layers):
            for block_idx in range(num_blocks):
                # Calculate drop path rate for this block
                block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
                # Add a transformer block with the calculated drop path rate
                self.layers.append(TransformerBlock(d_model, num_heads, block_dpr))
            if index != len(layers) - 1:
                self.layers.append(Downsampler())

        self.output_proj = nn.Linear(d_model, d_llm)
        self.inits = 'standard'
        self.apply(self._init_weights)
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            if self.inits == 'standard':
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif self.inits =='xavier':
                torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if "attention.out_proj" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if self.inits == 'standard':
                    p.data.normal_(
                        mean=0.0,
                        std=(0.02 / math.sqrt(2 * self.num_layers)),
                    )
    def forward(self, x, mask=None):
        x = self.pos_embedding(x)
        for layer in self.layers:
            x, mask = layer(x, mask)
        x = self.output_proj(x)
        return x, mask

def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()
    
class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False, dino_path=None):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = dino(dino_path) # InceptionI3d()
        # self.conv_1d = LightweightTemporalTransformer(input_dim=512, hidden_dim=1024, num_heads=8, dropout=0.1, max_seq_len=300)
        # self.conv_1d = TwoLayerTransformerBlock(d_model=512, num_heads=8, d_llm=1024)
        self.conv_1d = Transformer(d_model=512, num_heads=8, layers=[2,2])

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch,
                mask,
                ):
        # src shape: (all_frames_in_batch, 3, 224, 224)
        src, mask = self.conv_2d(src, src_length_batch) #(batch_size, seq_len, dim=512)
        src, mask = self.conv_1d(src, mask) #(batch_size, new_seq_len, new_dim=1024)

        return src, mask

class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()
        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder() 

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'], attention_mask=tgt_input['attention_mask'])[0]
        txt_logits = txt_logits.mean(dim=1)
        return txt_logits

class ImageCLIP(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear') :
        super(ImageCLIP, self).__init__()
        self.config = config
        self.model =  FeatureExtracter(dino_path=config['model']['dino'])
        post_params = {
        "in_dim": 1024,
        "hidden_dim": 300,
        "num_classes": 2300,
        "dropout": 0.2,
        "class_temperature": 0.1,
        "time_temperature": 0.1,
        "dynamic_time_temperatures": False,
        "dynamic_class_temperatures": False,
        "emb_lang": "de",
        "emb_pkl_dir": f"data/processed_words.phx_pkl",
        "trainable_emb": True,
    }
        self.head_model = HeadModel(**post_params)

        trans_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder()
        lora_config = LoraConfig(
            inference_mode=False,          # Enable training
            r=16,                          # Rank of the update matrices
            lora_alpha=32,                 # LoRA scaling factor
            lora_dropout=0.1,               # Dropout probability
            target_modules=["q_proj", "v_proj"]
        )

        self.trans_encoder = get_peft_model(trans_encoder, lora_config)
        for param in self.trans_encoder.parameters():
            param.requires_grad = False
        # Only unfreeze LoRA parameters
        for name, param in self.trans_encoder.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        param_after_lora = sum(p.numel() for p in self.trans_encoder.parameters() if p.requires_grad)
    
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))

    def forward(self, src_input):
        x, attention_mask = self.model(src_input['input_ids'], src_input['src_length_batch'], src_input['attention_mask']) # [b, n, c]
        # attention_mask = src_input['attention_mask']
        psp_logits = self.head_model(x, attention_mask)['logits']

        B, N, C = x.shape
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        img_logits = last_hidden_state.mean(dim=1)
        return img_logits, psp_logits

class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024):
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        for param in self.model_txt.parameters():
            param.requires_grad = False 
        trainable_params_model_texts = sum(p.numel() for p in self.model_txt.parameters() if p.requires_grad)

        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)
        trainable_params_model_images = sum(p.numel() for p in self.model_images.parameters() if p.requires_grad)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def compute_text_similarities_static(self, text_features):
        # Clone text features and disable gradient tracking
        text_features_static = text_features.clone().detach()
        text_features_static.requires_grad = False  # Ensure no gradients are tracked

        # Compute cosine similarities
        text_sim_matrix = torch.matmul(text_features_static, text_features_static.T)
        return text_sim_matrix

    def create_soft_target_matrix_with_gradients(self, batch_size, text_sim_matrix, scale_off_diag=0.1):
        # Create an identity matrix for the diagonal
        target_matrix = torch.eye(batch_size, device=text_sim_matrix.device, requires_grad=False)
        # Add off-diagonal text similarities
        off_diag_matrix = scale_off_diag * text_sim_matrix * (1 - torch.eye(batch_size, device=text_sim_matrix.device, requires_grad=False))
        target_matrix += off_diag_matrix
        return target_matrix

    def forward(self, src_input, tgt_input):
        image_features, psp_logits = self.model_images(src_input)
        text_features = self.model_txt(tgt_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_sim_matrix = self.compute_text_similarities_static(text_features)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)
        ground_truth = self.create_soft_target_matrix_with_gradients(
            batch_size=logits_per_image.shape[0],
            text_sim_matrix=text_sim_matrix,
            scale_off_diag=0.0,
        )
        return logits_per_image, logits_per_text, ground_truth, psp_logits

def config_decoder(config):
    from transformers import AutoConfig
    decoder_type = 'LLMD'
    if decoder_type == 'LD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes = True, 
                                                            config = AutoConfig.from_pretrained(config['model']['visual_encoder']+'/config.json'))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, 
                                                            config = AutoConfig.from_pretrained(config['model']['transformer']+'/config.json'))
class V_encoder(nn.Module):
    def __init__(self,
                emb_size,
                feature_size,
                ):
        super(V_encoder, self).__init__()

        # Linear layer to project input features to embedding size
        self.src_emb = nn.Linear(feature_size, emb_size)

        # Replace BatchNorm1d with LayerNorm and ReLU with GELU
        modules = []
        modules.append(nn.LayerNorm(emb_size))  # LayerNorm operates on the last dimension
        modules.append(nn.GELU())  # Use GELU instead of ReLU
        self.bn_ac = nn.Sequential(*modules)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, src):
        # Project input features to embedding size
        src = self.src_emb(src)  # Shape: (batch_size, seq_length, emb_size)
        # Apply LayerNorm and GELU
        src = self.bn_ac(src)  # No need to permute dimensions for LayerNorm
        return src  # Shape: (batch_size, seq_length, emb_size)

class gloss_free_model(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(frozen=False, dino_path=self.config['model']['dino'])
        # self.mbart = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])
        self.mbart = config_decoder(config)

        lora_config = LoraConfig(
            inference_mode=False,          # Enable training
            r=16,                          # Rank of the update matrices
            lora_alpha=32,                 # LoRA scaling factor
            lora_dropout=0.1,               # Dropout probability
            target_modules=["q_proj", "v_proj"]
        )
        self.mbart = get_peft_model(self.mbart, lora_config)
        for param in self.mbart.parameters():
            param.requires_grad = False
        # Only unfreeze LoRA parameters
        for name, param in self.mbart.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim,feature_size=embed_dim)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

    def share_forward(self, src_input):
        
        frames_feature, attention_mask = self.backbone(src_input['input_ids'], src_input['src_length_batch'], src_input['attention_mask'])
        # attention_mask = src_input['attention_mask']

        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds

        return inputs_embeds, attention_mask

    def forward(self,src_input, tgt_input):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),
                    # decoder_input_ids = tgt_input['input_ids'].cuda(),
                    labels = tgt_input['input_ids'].cuda(),
                    decoder_attention_mask = tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        return out['logits']
    def generate(self, src_input, max_new_tokens, num_beams, decoder_start_token_id ):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(inputs_embeds = inputs_embeds,
                                attention_mask = attention_mask, max_new_tokens=max_new_tokens, 
                                num_beams = num_beams,
                                decoder_start_token_id=decoder_start_token_id
                            )
        return out