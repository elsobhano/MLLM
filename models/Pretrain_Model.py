import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import wandb
from collections import OrderedDict
from models.utils import KLLoss, PG_Loss
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
        pg_criterion = PG_Loss()
        self.loss_img = criterion
        self.loss_txt = criterion
        self.loss_pg = pg_criterion
        ######################Prompts#######################
        self.landa = 1.0
    def forward(self, samples):
        src_input, tgt_input = samples
        return self.model(src_input, tgt_input)

    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, input_batch, batch_idx):
        batch, psp_ground_truth = input_batch[:-1], input_batch[-1]
        logits_per_image, logits_per_text, clip_ground_truth, psp_logits = self(batch)
        loss_imgs = self.loss_img(logits_per_image, clip_ground_truth)
        loss_texts = self.loss_txt(logits_per_text, clip_ground_truth)
        train_clip_loss = (loss_imgs + loss_texts)/2.0
        train_psp_loss = self.loss_pg(psp_logits, psp_ground_truth)
        train_total_loss = train_clip_loss + self.landa*train_psp_loss
        
        self.log("train_clip_loss", train_clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_psp_loss", train_psp_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_total_loss", train_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log gradient norm
        # total_norm = 0.0
        # for p in self.parameters():
        #     if p.grad is not None:
        #         total_norm += p.grad.norm(2).item()
        # self.log("grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return train_total_loss

    def validation_step(self, input_batch, batch_idx):
        batch, psp_ground_truth = input_batch[:-1], input_batch[-1]
        logits_per_image, logits_per_text, clip_ground_truth, psp_logits = self(batch)
        loss_imgs = self.loss_img(logits_per_image, clip_ground_truth)
        loss_texts = self.loss_txt(logits_per_text, clip_ground_truth)
        val_clip_loss = (loss_imgs + loss_texts)/2.0
        val_psp_loss = self.loss_pg(psp_logits, psp_ground_truth)
        val_total_loss = val_clip_loss + self.landa*val_psp_loss
        
        self.log("val_clip_loss", val_clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_psp_loss", val_psp_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_total_loss", val_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_total_loss
    
    def test_step(self, input_batch, batch_idx):
        batch, psp_ground_truth = input_batch[:-1], input_batch[-1]
        logits_per_image, logits_per_text, clip_ground_truth, psp_logits = self(batch)
        loss_imgs = self.loss_img(logits_per_image, clip_ground_truth)
        loss_texts = self.loss_txt(logits_per_text, clip_ground_truth)
        test_clip_loss = (loss_imgs + loss_texts)/2.0
        test_psp_loss = self.loss_pg(psp_logits, psp_ground_truth)
        test_total_loss = test_clip_loss + self.landa*test_psp_loss
        
        self.log("test_clip_loss", test_clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_psp_loss", test_psp_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_total_loss", test_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return test_total_loss

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
