import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import wandb
from collections import OrderedDict
from models.utils import KLLoss
from models.clip_models import SLRCLIP
import yaml

class PreTrainModel(pl.LightningModule):
    def __init__(self,
                config="configs/config.yaml",
                lr=3e-4, 
                ):
        super().__init__()
        self.save_hyperparameters()
        #################Load the Config file####################
        with open(config, 'r') as file:
            self.config = yaml.safe_load(file)
        ################Set the SLRCLIP ####################
        self.model = SLRCLIP(self.config)
        #################Set the Optimizer####################
        self.lr = lr
        criterion = KLLoss()
        self.loss_img = criterion
        self.loss_txt = criterion
        ######################Prompts#######################
    
    def forward(self, samples):
        src_input, tgt_input = samples
        return self.model(src_input, tgt_input)

    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth = self(batch)
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        total_loss = (loss_imgs + loss_texts)/2.0

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

        return total_loss

    def validation_step(self, batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth = self(batch)
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        total_loss = (loss_imgs + loss_texts)/2.0
        
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss
    
    def test_step(self, batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth = self(batch)
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        total_loss = (loss_imgs + loss_texts)/2.0
        
        self.log("test_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return total_loss

    def add_weight_decay(self, weight_decay, skip_list=()):
        """Custom method to create parameter groups with/without weight decay."""
        decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Ignore frozen parameters
            else:
                decay.append(param)
        return [
            {'params': decay, 'weight_decay': weight_decay}
        ]

    def configure_optimizers(self):

        print(f'lr: {self.lr}')
        optimizer = torch.optim.AdamW(self.add_weight_decay(weight_decay=0.01), lr=self.lr)
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,  # 5% of total steps for warmup
                anneal_strategy='cos',
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return [optimizer], [scheduler]

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )
        # Implement your own custom logic to clip gradients
        # You can call `self.clip_gradients` with your settings:
        
        # total_grad_norm_before = torch.sqrt(
        #     sum(
        #         (p.grad.norm(2) ** 2) for p in self.parameters() if p.grad is not None
        #     )
        # )
        
        # self.log("grad_norm_before", total_grad_norm_before, prog_bar=True, on_step=True, on_epoch=False)
        
        # self.clip_gradients(
        # optimizer,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="value",
        # )
        # total_grad_norm = torch.sqrt(
        #     sum(
        #         (p.grad.norm(2) ** 2) for p in self.parameters() if p.grad is not None
        #     )
        # )
        # Log the gradient norm to the progress bar
        # self.log("grad_norm_after", total_grad_norm, prog_bar=True, on_step=True, on_epoch=False)
