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
                landa_clip=1.0,
                landa_lm=1.0,
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
        SI_IDX,PAD_IDX,UNK_IDX,BOS_IDX, EOS_IDX = 0 ,1 ,2 ,3 ,4
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)
        ######################Prompts#######################
        self.landa_clip = landa_clip
        self.landa_lm = landa_lm
        self.save_hyperparameters()
    def forward(self, samples):
        src_input, tgt_input = samples
        return self.model(src_input, tgt_input)

    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, input_batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth, lm_logits = self(input_batch)
        
        # Contrastive loss
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        clip_loss = (loss_imgs + loss_texts)/2.0
        
        # Language model loss
        loss_lm = self.calc_loss(lm_logits, input_batch[-1]['input_ids'])
        
        total_loss = self.landa_clip*clip_loss + self.landa_lm*loss_lm 
        
        self.log("train_clip_loss", clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_lm_loss", loss_lm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, input_batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth, lm_logits = self(input_batch)
        
        # Contrastive loss
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        clip_loss = (loss_imgs + loss_texts)/2.0
        
        # Language model loss
        loss_lm = self.calc_loss(lm_logits, input_batch[-1]['input_ids'])
        
        total_loss = self.landa_clip*clip_loss + self.landa_lm*loss_lm 
        
        self.log("val_clip_loss", clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_lm_loss", loss_lm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return total_loss
    
    def test_step(self, input_batch, batch_idx):
        logits_per_image, logits_per_text, ground_truth, lm_logits = self(input_batch)
        
        # Contrastive loss
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        clip_loss = (loss_imgs + loss_texts)/2.0
        
        # Language model loss
        loss_lm = self.calc_loss(lm_logits, input_batch[-1]['input_ids'])
        
        total_loss = self.landa_clip*clip_loss + self.landa_lm*loss_lm 
        
        self.log("test_clip_loss", clip_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_lm_loss", loss_lm, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
    def calc_loss(self, outputs, targets):
        # outputs = logit[:, :-1, :]
        # targets = y[:, 1:]
        vocab_siz =  outputs.size(-1)
        return self.criterion(outputs.reshape(-1, vocab_siz), targets.reshape(-1))
