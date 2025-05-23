import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.utils import KLLoss
from models.clip_models import SLRCLIP
import yaml

class PreTrainModel(pl.LightningModule):
    def __init__(self,
                config="configs/config.yaml",
                lr=3e-4,
                landa_desc=1.0,
                landa_hamer=1.0,
                ):
        super().__init__()
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
        self.save_hyperparameters()
    def forward(self, samples):
        src_input, tgt_input = samples
        return self.model(src_input, tgt_input)

    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, input_batch, batch_idx):
        hamer_feats, hamer_mask = input_batch
        logits_per_image, logits_per_text, ground_truth = self(input_batch)
        
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        train_clip_loss = (loss_imgs + loss_texts)/2.0
        
        train_total_loss = train_clip_loss
        
        self.log("train_clip_loss", train_clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_total_loss", train_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return train_total_loss

    def validation_step(self, input_batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth= self(input_batch)
        
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        val_clip_loss = (loss_imgs + loss_texts)/2.0
        
        val_total_loss = val_clip_loss
        
        self.log("val_clip_loss", val_clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_total_loss", val_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return val_total_loss
    
    def test_step(self, input_batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth = self(input_batch)
        
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        test_clip_loss = (loss_imgs + loss_texts)/2.0
        
        test_total_loss = test_clip_loss
        
        self.log("test_clip_loss", test_clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
    def masked_l2_loss(self, pred, target, mask):
        """
        Computes the masked L2 loss (Mean Squared Error) between predictions and targets.

        Args:
            pred (torch.Tensor): Predictions of shape (Batch_size, Sequence_length, dim).
            target (torch.Tensor): Ground truth of shape (Batch_size, Sequence_length, dim).
            mask (torch.Tensor): Binary mask of shape (Batch_size, Sequence_length), where 1 indicates valid values and 0 indicates padding.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        B, T, D = pred.shape
        # Compute squared error
        squared_norm = torch.norm(pred - target, p=2, dim=-1) ** 2  # Shape: (Batch_size, Sequence_length, dim)
        
        masked_loss = squared_norm * mask
        
        # Compute mean over valid elements
        loss = masked_loss.sum() / (mask.sum() * D)
        
        return loss
