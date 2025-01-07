import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType
from models.video_clip import video_header
import numpy as np
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torchvision
import math
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

class resnet(nn.Module):
    def __init__(self, resnet_path):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18', resnet_path=resnet_path)

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)
        return x

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
                modules.append(
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

class MoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([FeedForward(d_model, d_model*2) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # Gating mechanism
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_probs = torch.softmax(gate_logits, dim=-1)
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)

        # Sparse combination of experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_mask = (top_k_indices[..., i] == i).unsqueeze(-1)
            expert_mask = expert_mask.expand(-1, -1, self.d_model)
            expert_output = self.experts[i](x)
            # print(expert_output.shape, expert_mask.shape, top_k_gate_probs[..., i:i+1].shape)
            output += expert_mask.float() * expert_output * top_k_gate_probs[..., i:i+1]
        
        # Compute load balancing loss
        load_balancing_loss = self.compute_load_balancing_loss(gate_probs, top_k_indices)

        return output, load_balancing_loss
    def compute_load_balancing_loss(self, gate_probs, top_k_indices):
        # Compute expert usage (how often each expert is in the top-k)
        # Flatten the top_k_indices to count expert occurrences
        top_k_indices_flat = top_k_indices.view(-1)  # [batch_size * seq_len * top_k]

        expert_usage = torch.zeros(self.num_experts, device=gate_probs.device)  # [num_experts]
        for i in range(self.top_k):
            expert_usage += (top_k_indices_flat == i).float().sum()  # Sum over batch and sequence

        # Normalize expert usage
        expert_usage = expert_usage / expert_usage.sum()

        # Compute load balancing loss
        mean_usage = expert_usage.mean()
        std_usage = expert_usage.std()
        load_balancing_loss = (std_usage / mean_usage) ** 2

        return load_balancing_loss

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=300):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Linear layers for Q, K, V
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Rotary positional embeddings
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)


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

        # Apply mask (if provided)
        if mask is not None:
            # Reshape mask to (batch_size, 1, 1, seq_len) for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            # Apply mask to scores
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Replace masked positions with -inf

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

        return attn_output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_llm):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, d_llm, num_heads)
        self.ffn = FeedForward(d_model, d_model*2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_llm)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = x + attn_output
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)

        x = self.output_proj(x)

        return x

class TransformerBlockWithMoE(nn.Module):
    def __init__(self, d_model, num_heads, d_llm, num_experts, top_k):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)  # Custom multi-head attention
        self.moe = MoE(d_model, num_experts, top_k)        # MoE layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_llm)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)  # Use custom multi-head attention
        x = x + attn_output
        x = self.norm1(x)

        # MoE feedforward
        moe_output, load_balancing_loss = self.moe(x)
        x = x + moe_output
        x = self.norm2(x)

        x = self.output_proj(x)

        return x, load_balancing_loss

class LightweightTemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=2, dropout=0.1, max_seq_len=1000):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer components
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # FFN
        # self.ffn = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 2, hidden_dim)
        # )
        self.moe = MoE(hidden_dim, num_experts=4, top_k=2)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Attention block
        residual = x
        x = self.norm1(x)
        
        # QKV projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply RoPE to queries and keys
        q_rope, k_rope = self.rope(q, k, seq_len)
        
        # Compute attention
        attn = torch.matmul(q_rope, k_rope.transpose(-2, -1)) * self.scale
        
        # mask shape: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        mask = mask.unsqueeze(1).unsqueeze(-1)
        broadcasted_mask = mask.expand(-1, self.num_heads, -1, seq_len)
        # # The mask will broadcast from [batch_size, 1, 1, seq_len] to [batch_size, num_heads, seq_len, seq_len]
        attn = attn.masked_fill(broadcasted_mask == 0, float('-1e4'))
        valid_positions = broadcasted_mask.sum(dim=-1, keepdim=True)  # Count valid positions
        all_masked = valid_positions == 0
        if all_masked.any():
                # Create uniform attention for rows where all positions are masked
                uniform_attention = torch.zeros_like(attn)
                uniform_attention = uniform_attention.masked_fill(broadcasted_mask.bool(), 1.0)
                uniform_attention = uniform_attention / (valid_positions.clamp(min=1))
                attn = torch.where(all_masked, uniform_attention, attn)
        # attn = attn.to(torch.float32)
        # attn = torch.where(torch.isnan(attn), torch.full_like(attn, float('-1e3')), attn)
        # attn = attn.to(torch.float16)
        # Apply softmax
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.o_proj(out)
        
        x = residual + self.dropout(out)
        
        # FFN block
        residual = x
        x = self.norm2(x)
        # x = self.ffn(x)
        moe_output = self.moe(x)
        x = residual + x
        
        return x

def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()
    
class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False, resent_path=None):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet(resnet_path=resent_path) # InceptionI3d()
        # self.conv_1d = LightweightTemporalTransformer(input_dim=512, hidden_dim=1024, num_heads=8, dropout=0.1, max_seq_len=300)
        # self.conv_1d = TransformerBlockWithMoE(d_model=512, num_heads=8, d_llm=1024, num_experts=4, top_k=2)
        self.conv_1d = video_header(
            vid_head = "Transf",
            interaction = "DP",
            temporal_layer=1,
            clip_state_dict=None,
            spe_cls_feature=400,)
        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch,
                mask,
                ):
        # src shape: (all_frames_in_batch, 3, 224, 224)
        src = self.conv_2d(src, src_length_batch) #(batch_size, seq_len, dim=512)
        src = self.conv_1d(src, mask) #(batch_size, new_seq_len, new_dim=1024)

        return src 

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
        self.model =  FeatureExtracter(resent_path=config['model']['resnet'])

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

        # self.lm_head = make_head(inplanes, planes, head_type)
        
    def forward(self, src_input):
        x = self.model(src_input['input_ids'], src_input['src_length_batch'], src_input['attention_mask']) # [b, n, c]
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask, return_dict=True)
        output = outs['last_hidden_state']
        # output = self.lm_head(last_hidden_state[:, 0, :])
        return output.mean(dim=1)

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
        image_features = self.model_images(src_input)
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
            scale_off_diag=0.0
        )
        return logits_per_image, logits_per_text, ground_truth 

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
                config,
                ):
        super(V_encoder, self).__init__()
        
        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                src,
                ):
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src

class gloss_free_model(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(frozen=False, resent_path=self.config['model']['resnet'])
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
            self.sign_emb = V_encoder(emb_size=embed_dim,feature_size=embed_dim, config = config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

    def share_forward(self, src_input):
        
        frames_feature = self.backbone(src_input['input_ids'], src_input['src_length_batch'], src_input['attention_mask'])
        attention_mask = src_input['attention_mask']

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