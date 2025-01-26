import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from sacrebleu.metrics import BLEU
import pandas as pd
import os
import wandb
from collections import OrderedDict
from models.utils import extract_layers_by_prefix
from models.clip_models import gloss_free_model
from pathlib import Path
import yaml

import math

SI_IDX,PAD_IDX,UNK_IDX,BOS_IDX, EOS_IDX = 0 ,1 ,2 ,3 ,4

class FineTuneModel(pl.LightningModule):
    def __init__(self,
                config="configs/config.yaml",
                args=None,
                eval_freq=10,
                csv_dire=None
                ):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_hyperparameters()
        #################Load the Config file####################
        with open(config, 'r') as file:
            self.config = yaml.safe_load(file)
        self.args = args
        ################Set the Sign Encoder####################
        self.model = gloss_free_model(self.config, self.args)
        # for k, v in self.model.named_parameters():
        #     print(k)
        print('***********************************')
        print('Load parameters from Pretrained...')
        print('***********************************')
        state_dict = torch.load(args.model_ckpt, map_location='cpu')['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # k = '.'.join(k.split('.')[1:])
            # new_state_dict[k] = v
            if 'conv_2d' in k or 'conv_1d' in k:
                k = 'backbone.'+'.'.join(k.split('.')[3:])
                new_state_dict[k] = v
            if 'trans_encoder' in k:
                k = 'mbart.base_model.model.model.encoder.'+'.'.join(k.split('.')[5:])
                new_state_dict[k] = v

        ret = self.model.load_state_dict(new_state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
        # for name, param in self.model.named_parameters():
        #     if "backbone" in name or "mbart.base_model.model.model.encoder" in name:
        #         param.requires_grad = False
        #         print(f"Froze layer: {name}")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Unfroze layer: {name}")
        #################Initialize the tokenizer####################
        self.tokenizer = MBartTokenizer.from_pretrained(self.config['model']['tokenizer'], src_lang = 'de_DE', tgt_lang = 'de_DE')
        lang_code_to_id = self.tokenizer.lang_code_to_id['de_DE']
        self.end_sym = ' .'
        self.max_txt_len = 64
        #################Set the Projection####################
        # self.proj_visual = nn.Linear(1024, 1024)
        # self.proj_visual.load_state_dict(proj_state_dict, strict=True)
        #################Set the Optimizer####################
        self.lr = self.args.lr
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

        self.csv_dire = csv_dire
        
        self.train_decoded_teacher = []
        self.train_step_outputs = []

        self.validation_decoded = []
        self.validation_decoded_teacher = []
        self.validation_step_outputs = []
        
        
        self.test_decoded = []
        self.test_step_outputs = []
        ######################Prompts#######################
    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        src_input, tgt_input, _ = batch
        outputs = self.model(src_input, tgt_input)
        loss = self.calc_loss(outputs, tgt_input['input_ids'])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_input, tgt_input, _ = batch
        outputs = self.model(src_input, tgt_input)
        loss = self.calc_loss(outputs, tgt_input['input_ids'])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if (self.current_epoch + 1) % self.eval_freq == 0 and self.current_epoch != 0:
            self.validation_decoded_teacher.extend(self.teacher_forcing_generate(outputs))
            self.validation_decoded.extend(self.generate(src_input))
            self.validation_step_outputs.extend(self.tokenizer.batch_decode(tgt_input['input_ids'], skip_special_tokens=True))
        return loss

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.eval_freq == 0 and self.current_epoch != 0:
            tgt_refs = [item for item in self.validation_step_outputs]
            hypotheses = [item for item in self.validation_decoded]
            hypotheses_teacher = [item for item in self.validation_decoded_teacher]
            new_data = {"hypotheses": hypotheses, "hypotheses_teacher": hypotheses_teacher, "targets": tgt_refs}
            file_path = self.csv_dire + f"val_outputs_{self.current_epoch+1}_{self.trainer.global_rank}.csv"
            self.add_data_to_csv(file_path, new_data, columns=["hypotheses", "hypotheses_teacher", "targets"])

            self.validation_decoded = []
            self.validation_decoded_teacher = []
            self.validation_step_outputs = []

        # elif (self.current_epoch) % self.eval_freq == 0 and self.current_epoch != 0:
            hypotheses = []
            hypotheses_teacher = []
            tgt_refs = []
            for idx in range(self.trainer.world_size):
                df = pd.read_csv(self.csv_dire + f"val_outputs_{self.current_epoch+1}_{idx}.csv", sep='|')
                hypotheses.extend([str(item) + ' .' for item in df['hypotheses'].tolist()]) # df['hypotheses'].tolist()
                hypotheses_teacher.extend([str(item) + ' .' for item in df['hypotheses_teacher'].tolist()]) # df['hypotheses_teacher'].tolist()
                tgt_refs.extend([str(item) + ' .' for item in df['targets'].tolist()]) # df['targets'].tolist()
            
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_text("hypotheses_teacher", "\n".join(hypotheses_teacher[:5]), self.current_epoch)
                self.logger.experiment.add_text("hypotheses", "\n".join(hypotheses[:5]), self.current_epoch)
                self.logger.experiment.add_text("targets", "\n".join(tgt_refs[:5]), self.current_epoch)
            
            if isinstance(self.logger, WandbLogger):
                self._log_to_wandb(tgt_refs[:5], hypotheses[:5], hypotheses_teacher[:5], split="val", epoch=self.current_epoch)
            
            print(len(tgt_refs), len(hypotheses))
            # print(hypotheses)
            # print(tgt_refs)
            bleu = BLEU()
            bleu_s = bleu.corpus_score(hypotheses, [tgt_refs]).score
            self.log("val_bleu", bleu_s ,prog_bar=True, sync_dist=True)
            print(bleu_s)
            print('*'*50)
            print(len(tgt_refs), len(hypotheses_teacher))
            # print(hypotheses_teacher)
            # print(tgt_refs)
            bleu = BLEU()
            bleu_s = bleu.corpus_score(hypotheses_teacher, [tgt_refs]).score
            print(bleu_s)
            self.log("val_teacher_bleu", bleu_s ,prog_bar=True, sync_dist=True)

    def _log_to_wandb(self, targets, hypotheses, hypotheses_teacher, split: str, epoch: int):
        """Log text examples to Weights & Biases."""
        # Create a table
        columns = ["Target", "hypotheses", "hypotheses_teacher"]
        data = [
            [t, h, h_teacher]
            for t, h, h_teacher in zip(targets, hypotheses, hypotheses_teacher)
        ]
        table = wandb.Table(columns=columns, data=data)
        self.logger.experiment.log({f"{split}_outputs_{epoch}": table})
    
    def test_step(self, batch, batch_idx):
        src_input, tgt_input, _ = batch
        outputs = self.model(src_input, tgt_input)
        loss = self.calc_loss(outputs, tgt_input['input_ids'])

        self.log("test_loss", loss, sync_dist=True)

        self.test_decoded = self.generate(src_input)
        self.test_step_outputs = self.tokenizer.batch_decode(tgt_input['input_ids'], skip_special_tokens=True)
        tgt_refs = [str(item) + ' .' for item in self.test_step_outputs]
        hypotheses = [str(item) + ' .' for item in self.test_decoded]
        new_data = {"hypotheses": hypotheses, "targets": tgt_refs}
        file_path = self.csv_dire + f"test_outputs_{self.trainer.global_rank}.csv"
        self.add_data_to_csv(file_path, new_data, columns=["hypotheses", "targets"])
        
        return loss

    def on_test_epoch_end(self):
        hypotheses = []
        tgt_refs = []
        for idx in range(self.trainer.world_size):
            df = pd.read_csv(self.csv_dire + f"test_outputs_{idx}.csv", sep='|')
            hypotheses.extend(df['hypotheses'].tolist())
            tgt_refs.extend(df['targets'].tolist())
            
        print(len(tgt_refs), len(hypotheses))
        bleu = BLEU()
        bleu_s = bleu.corpus_score(hypotheses, [tgt_refs]).score
        
        self.log("test_bleu", bleu_s ,prog_bar=True, sync_dist=True)
        self.test_decoded = []
        self.test_step_outputs = []
    
    def teacher_forcing_generate(self, logits):
        predicted = torch.argmax(logits, dim=-1)
        generated_texts = self.tokenizer.batch_decode(predicted, skip_special_tokens=True)
        
        return generated_texts

    def generate(self, src_input):
        max_new_tokens, num_beams, decoder_start_token_id  = 150, 4, self.tokenizer.lang_code_to_id['de_DE']
        out = self.model.generate(
                            src_input,
                            max_new_tokens, 
                            num_beams,
                            decoder_start_token_id
                            )

        generated_texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        return generated_texts

    def calc_loss(self, outputs, targets):
        # outputs = logit[:, :-1, :]
        # targets = y[:, 1:]
        vocab_siz =  outputs.size(-1)
        return self.criterion(outputs.reshape(-1, vocab_siz), targets.reshape(-1))

    def add_weight_decay(self, weight_decay, skip_list=()):
        """Custom method to create parameter groups with/without weight decay."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Ignore frozen parameters
            # if 'gate' in name:
            #     no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.0},
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
            gradient_clip_algorithm="norm",
        )
        # self.clip_gradients(
        # optimizer,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="value",
        # )

    def add_data_to_csv(self,file_path, new_data, columns):
        """
        Add data to a CSV file. If the file doesn't exist, create it with the given headers.
        Args:
            file_path (str): Path to the CSV file.
            new_data (list of dict): List of data to append.
            columns (list of str): List of column headers.
        """
        # Check if the file exists
        file_exists = os.path.exists(file_path)

        # Convert new data to a DataFrame
        df = pd.DataFrame(new_data, columns=columns)

        # Write or append the data
        if file_exists:
            df.to_csv(file_path, mode='a', index=False, header=False, sep='|')  # Append without header
            print(f"Data appended to {file_path}.")
        else:
            df.to_csv(file_path, mode='w', index=False, header=True, sep='|')  # Write with header
            print(f"New file created with data: {file_path}.")