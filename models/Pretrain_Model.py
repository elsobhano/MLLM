import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import wandb
from collections import OrderedDict
from models.utils import extract_layers_by_prefix, KLLoss
from models.clip_models import SLRCLIP, Text_Decoder
import yaml
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
PAD_IDX = 1
class PreTrainModel(pl.LightningModule):
    def __init__(self,
                config="configs/config.yaml",
                args=None, 
                ):
        super().__init__()
        self.save_hyperparameters()
        #################Load the Config file####################
        with open(config, 'r') as file:
            self.config = yaml.safe_load(file)
        
        ################Set the Text Decoder####################
        self.txt_decoder = Text_Decoder(self.config)
        ################Set the SLRCLIP ####################
        self.model = SLRCLIP(self.config)
        #################Set the Optimizer####################
        self.args = args
        criterion = KLLoss()
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
        self.loss_img = criterion
        self.loss_txt = criterion
        ######################Prompts#######################
        self.step_counter = 0
        self.automatic_optimization = False  # Enable manual optimization
    def forward(self, src_input, tgt_input):
        # src_input, tgt_input = samples
        return self.pretrain_model(src_input, tgt_input)

    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        mask_optimizer = self.trainer.optimizers[1]
        mask_lr = mask_optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mask_learning_rate', mask_lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        optimizer, mask_optimizer = self.optimizers()
        self.step_counter += 1
        src_input, tgt_input, masked_tgt_input = batch
        logits_per_image, logits_per_text, ground_truth, _ = self(src_input, tgt_input)
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        total_loss = (loss_imgs + loss_texts)/2.0
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if self.step_counter % 5 == 0:
            # print(tgt_input['input_ids'].shape, masked_tgt_input['input_ids'].shape, encoder_hidden_states.shape)
            _, encoder_hidden_states_deocder = self.pretrain_model.model_txt(masked_tgt_input)
            encoder_hidden_states_deocder = encoder_hidden_states_deocder.detach().clone()
            lm_logits = self.txt_decoder(tgt_input, masked_tgt_input, encoder_hidden_states_deocder)
            masked_lm_loss = self.loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].view(-1))
            self.log("train_masked_loss", masked_lm_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            mask_optimizer.zero_grad()
            self.manual_backward(masked_lm_loss)
            mask_optimizer.step()

        return total_loss
    def on_train_epoch_end(self):
        
        schedulers = self.lr_schedulers()
        schedulers[0].step(self.current_epoch)  # Step the first scheduler
        schedulers[1].step(self.current_epoch)  # Step the second scheduler
    

    def validation_step(self, batch, batch_idx):
        src_input, tgt_input, masked_tgt_input = batch
        logits_per_image, logits_per_text, ground_truth, _ = self(src_input, tgt_input)
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        total_loss = (loss_imgs + loss_texts)/2.0
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # print(tgt_input['input_ids'].shape, masked_tgt_input['input_ids'].shape, encoder_hidden_states.shape)
        _, encoder_hidden_states_deocder = self.pretrain_model.model_txt(masked_tgt_input)
        encoder_hidden_states_deocder = encoder_hidden_states_deocder.detach().clone()
        lm_logits = self.txt_decoder(tgt_input, masked_tgt_input, encoder_hidden_states_deocder)
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].view(-1))
        self.log("val_masked_loss", masked_lm_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        src_input, tgt_input, masked_tgt_input = batch
        logits_per_image, logits_per_text, ground_truth, _ = self(src_input, tgt_input)
        loss_imgs = self.loss_img(logits_per_image, ground_truth)
        loss_texts = self.loss_txt(logits_per_text, ground_truth)
        total_loss = (loss_imgs + loss_texts)/2.0
        self.log("test_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # print(tgt_input['input_ids'].shape, masked_tgt_input['input_ids'].shape, encoder_hidden_states.shape)
        _, encoder_hidden_states_deocder = self.pretrain_model.model_txt(masked_tgt_input)
        encoder_hidden_states_deocder = encoder_hidden_states_deocder.detach().clone()
        lm_logits = self.txt_decoder(tgt_input, masked_tgt_input, encoder_hidden_states_deocder)
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].view(-1))
        self.log("test_masked_loss", masked_lm_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    # def add_weight_decay(self, weight_decay, skip_list=()):
    #     """Custom method to create parameter groups with/without weight decay."""
    #     decay = []
    #     no_decay = []
    #     for name, param in self.pretrain_model.named_parameters():
    #         if not param.requires_grad:
    #             continue  # Ignore frozen parameters
    #         # if 'gate' in name:
    #         #     no_decay.append(param)
    #         else:
    #             decay.append(param)
    #     return [
    #         {'params': no_decay, 'weight_decay': 0.0},
    #         {'params': decay, 'weight_decay': weight_decay}
    #     ]

    def configure_optimizers(self):

        optimizer = create_optimizer(self.args, self.pretrain_model)
        scheduler = {"scheduler": create_scheduler(self.args, optimizer)[0], 
                    "interval": "epoch", 
                    "frequency": 1}
        
        mask_optimizer = torch.optim.AdamW(self.txt_decoder.parameters(), lr=5e-4, weight_decay=0.01, betas=(0.9, 0.98))
        mask_scheduler = {"scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer=mask_optimizer,
                            eta_min=1e-8,
                            T_max=self.trainer.max_epochs,
                            ),
                          "interval": "epoch",
                          "frequency": 1 
                        }
        return [optimizer, mask_optimizer], [scheduler, mask_scheduler]

    # def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    #     # Implement your own custom logic to clip gradients
    #     # You can call `self.clip_gradients` with your settings:
    #     self.clip_gradients(
    #     optimizer,
    #     gradient_clip_val=1.0,
    #     gradient_clip_algorithm="value",
    #     )
    #     self.clip_gradients(
    #         optimizer,
    #         gradient_clip_val=1.0,
    #         gradient_clip_algorithm="norm",
    #     )
