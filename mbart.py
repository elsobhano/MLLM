from transformers import MBartForConditionalGeneration
from pathlib import Path
from transformers import AutoConfig
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

visual_encoder_path = '/home/sobhan/Documents/Code/GFSLT-VLP/pretrain_models/mytran'
transformer_path = '/home/sobhan/Documents/Code/GFSLT-VLP/pretrain_models/MBart_trimmed'
decoder_type = 'LLMD'
if decoder_type == 'LD':
    mbart = MBartForConditionalGeneration.from_pretrained(visual_encoder_path, ignore_mismatched_sizes = True, 
                                                            config = AutoConfig.from_pretrained(f'{visual_encoder_path}/config.json'))
elif decoder_type == 'LLMD':
    mbart = MBartForConditionalGeneration.from_pretrained(transformer_path, ignore_mismatched_sizes = True, 
                                                        config = AutoConfig.from_pretrained(f'{transformer_path}/config.json'))#AutoConfig.from_pretrained(Path(config['model']['transformer'])/'LLMD_config.json'))
    
# mbart.model.encoder = nn.Identity()
# print(mbart)
param_before_lora = sum(p.numel() for p in mbart.parameters() if p.requires_grad)
print(f'params before lora: {param_before_lora}')
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # For causal language modeling tasks
    inference_mode=False,          # Enable training
    r=16,                          # Rank of the update matrices
    lora_alpha=32,                 # LoRA scaling factor
    lora_dropout=0.1,               # Dropout probability
    target_modules=["q_proj", "v_proj"]
)
mbart = get_peft_model(mbart, lora_config)
for param in mbart.parameters():
    param.requires_grad = False
        # Only unfreeze LoRA parameters
for name, param in mbart.named_parameters():
    if "lora" in name:
        # print(name)
        param.requires_grad = True
param_after_lora = sum(p.numel() for p in mbart.parameters() if p.requires_grad)
print(f'params after lora: {param_after_lora}')