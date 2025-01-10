import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration
from models.i3d import InceptionI3d
from peft import get_peft_model, LoraConfig, TaskType
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
        self.mapper = nn.Linear(1024, 512)

    def forward(self, x, d ,lengths):
        x = self.resnet(x)
        d = self.mapper(d)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(torch.cat([x[start:end], d[start:end]], dim=1))
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
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

def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()
    
class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False, resent_path=None):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet(resnet_path=resent_path) # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=1024, hidden_size=1024, conv_type=2)
    # 1536
        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self,
                image_batch,
                desc_batch,
                src_length_batch,
                ):
        src = self.conv_2d(image_batch, desc_batch, src_length_batch)
        src = self.conv_1d(src)

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
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear'):
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

        self.lm_head = make_head(inplanes, planes, head_type)
        
    def forward(self, src_input):
        x = self.model(src_input['input_ids'], src_input['desc_feats'], src_input['src_length_batch']) # [b, n, c]
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        output = self.lm_head(last_hidden_state[:, 0, :])
        return output

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
            scale_off_diag=0.0,
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
        
        frames_feature = self.backbone(src_input['input_ids'], src_input['desc_feats'] ,src_input['src_length_batch'])
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