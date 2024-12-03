import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import XGLMTokenizer
from sacrebleu.metrics import BLEU
import pandas as pd
import os
import wandb
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType
from transformers import MBartConfig, MBartModel





class PreTrain(pl.LightningModule):
    def __init__(self, 
                path_1="path_1", 
                path_2="path_2",
                lr=3e-4, 
                encoder_ckpt=None,
                eval_freq=10,
                ):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_hyperparameters()

        ################Set the visual Encoder####################
        self.visual_encoder = models.resnet18(pretrained=True) # ResNet18 pre-trained on ImageNet
        self.visual_encoder.fc = nn.Identity()
        ################Set the description Encoder####################
        self.disc_encoder = None #  a pre-trained, frozen 12-layer BERT
        ##################Set the Description Mapper####################
        self.disc_mapper = DescriptionMapper(768, 512, 768) #  a simple two-layer MLP structure that predicts Dj(all frames descriptions of the video)
        # from Vj(all the features of the frames of the video)
        ##################Set the Modality Adapter###################
        self.modality_adapter = ModalityAdapter(512, 512, 3, 1024, 512) # consisting of a 1D convolution, a maxpooling layer, and a two-layer MLP
        #################Set the Mutlimodal Enocer with LoRA##########
        lora_config = LoraConfig(
                r=8,  # LoRA rank
                lora_alpha=32,  # Scaling factor
                target_modules=["q_proj", "v_proj"],  # Apply LoRA to query and value projections
                lora_dropout=0.1,  # Dropout for LoRA layers
                task_type="SEQ_2_SEQ_LM"  # Task type: sequence-to-sequence
            )
        self.multimodal_encoder = MBartEncoderWithLoRA(lora_config=lora_config) #  mBART encoder ,which consists of 12 layers and is initialized with parameters pre-trained on a large corpus
        # Add LoRA to the model
        #################Set the text encoder###################
        self.text_encoder = MBartEncoder() # Frozen mBART encoder ,which consists of 12 layers and is initialized with parameters pre-trained on a large corpus
        #################Set the Optimizer####################
        self.lr = lr
        #################Set the loss settings###############
        self.lmbda = 1
        self.loss_img = KLLoss()
        self.loss_text = KLLoss()
        
    def forward(self, batch):
        images, discs, texts = batch
        visual_features = self.visual_encoder(images) # V_{j}
        visual_mapped = self.disc_mapper(visual_features) # D_{j}-Predicted
        SE_j = self.modality_adapter(torch.cat(visual_mapped, discs, dim=1))
        M_predicted = self.multimodal_encoder(SE_j)
        return (M_predicted, texts), (visual_mapped, discs)
    
    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss_total = self.mmlp_loss(self.forward(batch))
        return loss_total

    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass
    
    def align_loss(self, visual_features, text_features):
        # visual features and tesxt features should be normalized 
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * visual_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ visual_features.t()
        
        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)
        
        images_loss = self.loss_img(logits_per_image, ground_truth)
        texts_loss = self.loss_text(logits_per_text, ground_truth)
        
        loss_align = (images_loss + texts_loss) / 2
        return loss_align
    
    def dm_loss(self, predicted_discs, target_discs):
        """
        Calculate the normalized L2 loss (Euclidean distance) between two tensors.
        
        Args:
            tensor1 (Tensor): First tensor with shape (B, T).
            tensor2 (Tensor): Second tensor with shape (B, T).
        
        Returns:
            Tensor: The normalized L2 loss as a scalar.
        """
        # Compute the squared difference
        loss = torch.sum((predicted_discs - target_discs) ** 2)
        
        # Normalize the loss by dividing by the number of elements (B * T)
        B, T = predicted_discs.size()
        normalized_loss = loss / (B * T)
        
        return normalized_loss

    def mmlp_loss(self, features, discs):
        visual_features, text_features = features
        predicted_discs, target_discs = discs

        loss_align = self.align_loss(visual_features, text_features)
        loss_dm = self.dm_loss(predicted_discs, target_discs)
        loss_mmlp = loss_align + self.lmbda * loss_dm
        return loss_mmlp
    
    def add_weight_decay(self, weight_decay, skip_list=()):
        """Custom method to create parameter groups with/without weight decay."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Ignore frozen parameters
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': weight_decay}
        ]
    
    def configure_optimizers(self):

        print(f'lr: {self.lr}')
        optimizer = torch.optim.AdamW(self.add_weight_decay(weight_decay=0.001), lr=self.lr)
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,  # 5% of total steps for warmup
                anneal_strategy='cos'
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return [optimizer], [scheduler]
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Implement your own custom logic to clip gradients
        # You can call `self.clip_gradients` with your settings:
        self.clip_gradients(
        optimizer,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value",
        )
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

class DescriptionMapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the description mapper ψₘ.

        Args:
            input_dim (int): Dimension of the input vector (V).
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output vector (D).
        """
        super(DescriptionMapper, self).__init__()
        
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First layer
            nn.ReLU(),                         # Activation
            nn.Linear(hidden_dim, output_dim)  # Second layer
        )

    def forward(self, x):
        """
        Forward pass for the description mapper.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.mapper(x)
    

class ModalityAdapter(nn.Module):
    def __init__(self, input_dim, conv_out_channels, kernel_size, mlp_hidden_dim, output_dim):
        """
        Initializes the modality adapter ψₘₐ.

        Args:
            input_dim (int): Number of input channels (dimensionality of Vj or D̂j features).
            conv_out_channels (int): Number of output channels for the 1D convolution.
            kernel_size (int): Kernel size for the 1D convolution.
            mlp_hidden_dim (int): Hidden dimension for the MLP layers.
            output_dim (int): Output dimension for the modality adapter.
        """
        super(ModalityAdapter, self).__init__()
        
        # 1D convolution for temporal modeling
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=conv_out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2  # Padding to preserve temporal dimensions
        )
        
        # Max-pooling layer
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass for the modality adapter.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim, seq_len),
                        where seq_len is the temporal dimension.

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Apply 1D convolution
        x = self.conv1d(x)  # (batch_size, conv_out_channels, seq_len)
        
        # Apply max-pooling
        x = self.maxpool(x)  # (batch_size, conv_out_channels, seq_len//2)
        
        # Global average pooling over the temporal dimension
        x = torch.mean(x, dim=-1)  # (batch_size, conv_out_channels)
        
        # Apply MLP
        x = self.mlp(x)  # (batch_size, output_dim)
        
        return x
    

class MBartEncoder(nn.Module):
    def __init__(self, pretrained_model_name="facebook/mbart-large-cc25", num_layers=12):
        """
        Initializes the mBART encoder with a subset of layers.

        Args:
            pretrained_model_name (str): Name of the pre-trained mBART model from Hugging Face.
            num_layers (int): Number of encoder layers to use.
        """
        super(MBartEncoder, self).__init__()
        
        # Load the pre-trained mBART model
        self.mbart = MBartModel.from_pretrained(pretrained_model_name)
        for param in self.mbart.parameters():
            param.requires_grad = False  # Freeze the pre-trained weights

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for the mBART encoder.

        Args:
            input_ids (Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (Tensor, optional): Attention mask (batch_size, seq_len).

        Returns:
            Tensor: Encoded outputs (batch_size, seq_len, hidden_dim).
        """
        # Pass through the mBART encoder
        encoder_outputs = self.mbart.encoder(input_ids, attention_mask=attention_mask)
        return encoder_outputs.last_hidden_state

class MBartEncoderWithLoRA(nn.Module):
    def __init__(self, pretrained_model_name="facebook/mbart-large-cc25", num_layers=12, lora_config=None):
        """
        mBART encoder with LoRA adapters.

        Args:
            pretrained_model_name (str): Name of the pre-trained mBART model.
            num_layers (int): Number of encoder layers to use.
            lora_config (LoraConfig): LoRA configuration for adding adapters.
        """
        super(MBartEncoderWithLoRA, self).__init__()
        
        # Load pre-trained mBART model
        self.mbart = MBartModel.from_pretrained(pretrained_model_name)
        
        # Apply LoRA to the encoder
        if lora_config is not None:
            self.mbart.encoder = get_peft_model(self.mbart.encoder, lora_config)
        
        for param in self.mbart.parameters():
            param.requires_grad = False

        # Only unfreeze LoRA parameters
        for name, param in self.mbart.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for the mBART encoder with LoRA.

        Args:
            input_ids (Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (Tensor, optional): Attention mask (batch_size, seq_len).

        Returns:
            Tensor: Encoded outputs (batch_size, seq_len, hidden_dim).
        """
        encoder_outputs = self.mbart.encoder(input_ids, attention_mask=attention_mask)
        return encoder_outputs.last_hidden_state

class KLLoss(torch.nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=torch.nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss