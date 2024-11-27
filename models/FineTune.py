import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import XGLMTokenizer
from sacrebleu.metrics import BLEU
import pandas as pd
import os
import wandb
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType
class FineTuneModel(pl.LightningModule):
    def __init__(self, 
                path_1="path_1", 
                path_2="path_2",
                lr=3e-4, 
                encoder_ckpt=None,
                eval_freq=10,
                csv_dire=None
                ):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_hyperparameters()

        ################Set the Sign Encoder####################
        
        #################Set the Optimizer####################
        self.lr = lr
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

        self.csv_dire = csv_dire
        
        self.train_decoded_teacher = []
        self.train_step_outputs = []

        self.validation_decoded = []
        self.validation_decoded_teacher = []
        self.validation_step_outputs = []
        
        
        self.test_decoded = []
        self.test_step_outputs = []
        ######################Prompts#######################
        prompt = "Translate the following sign language video into German:\n"
        self.promts_ids = self.tokenizer(prompt, truncation=False, add_special_tokens=False ,return_tensors='pt')['input_ids']
        self.prompt_length = self.promts_ids.shape[1]
    
    def transfer_specific_features(self, old_checkpoint_path, prefixes):
        """
        Transfer specific feature layers from checkpoint to new model.
        
        Args:
            old_checkpoint_path (str): Path to the checkpoint file
            feature_prefixes (list): List of feature prefixes to transfer
        """
        # Get matched layers
        matched_layers, filtered_state_dict = extract_layers_by_prefix(
            old_checkpoint_path, 
            prefixes
        )
        
        # Load matched layers into new model
        self.load_state_dict(filtered_state_dict, strict=False)
        
        # Clean up
        del filtered_state_dict
        del matched_layers
        torch.cuda.empty_cache()

    def forward(self, samples):
        video_clips, atts_clips = samples['video_clips'], samples['atts_clips']
        bsz, clip_len, num_channels, num_frames, height, width = samples['video_clips'].shape

        clips_embeds = self.sign_encoder(video_clips.view(-1, num_channels, num_frames, height, width))['embds'].squeeze()
        clips_embeds = clips_embeds.view(bsz, clip_len, -1)
        clips_embeds = self.proj_visual(clips_embeds)
        clips_embeds, atts_clips = self.prompt_wrap(clips_embeds, atts_clips, self.prompt)
        self.tokenizer.padding_side = 'right'
        text = [t + self.end_sym for t in samples["text_input"]]
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        )
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([atts_clips.shape[0], atts_clips.shape[1]+1],
                    dtype=torch.long).fill_(-100))  # plus one for bos
        targets = torch.cat([empty_targets, targets], dim=1).to('cuda')
        
        batch_size = clips_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device='cuda') * self.tokenizer.bos_token_id
        
        bos_embeds = self.xglm.base_model.model.model.embed_tokens(bos)
        atts_bos = atts_clips[:, :1]

        to_regress_embeds = self.xglm.base_model.model.model.embed_tokens(to_regress_tokens.input_ids.to('cuda'))
        inputs_embeds = torch.cat([bos_embeds, clips_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_clips, to_regress_tokens.attention_mask.to('cuda')], dim=1)
        outputs = self.xglm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        logits = outputs.logits
        return logits, targets
    def on_train_epoch_start(self):
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        outputs, labels = self(batch)
        loss = self.calc_loss(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, labels = self(batch)
        
        loss = self.calc_loss(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if (self.current_epoch + 1) % self.eval_freq == 0 and self.current_epoch != 0:
            nn_idx = (labels[0] != -100).nonzero(as_tuple=True)[0]
            f_nn_idx = nn_idx[0].item()
            self.validation_decoded_teacher.extend(self.teacher_forcing_generate(outputs, idx_start_resp=f_nn_idx))
            self.validation_decoded.extend(self.generate(batch))
            self.validation_step_outputs.extend([text + self.end_sym for text in batch['text_input']])
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

        elif (self.current_epoch) % self.eval_freq == 0 and self.current_epoch != 0:

            hypotheses = []
            hypotheses_teacher = []
            tgt_refs = []
            for idx in range(self.trainer.world_size):
                df = pd.read_csv(self.csv_dire + f"val_outputs_{self.current_epoch}_{idx}.csv", sep='|')
                hypotheses.extend([str(item) for item in df['hypotheses'].tolist()]) # df['hypotheses'].tolist()
                hypotheses_teacher.extend([str(item) for item in df['hypotheses_teacher'].tolist()]) # df['hypotheses_teacher'].tolist()
                tgt_refs.extend([str(item) for item in df['targets'].tolist()]) # df['targets'].tolist()
            
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_text("hypotheses_teacher", "\n".join(hypotheses_teacher[:5]), self.global_step)
                self.logger.experiment.add_text("hypotheses", "\n".join(hypotheses[:5]), self.global_step)
                self.logger.experiment.add_text("targets", "\n".join(tgt_refs[:5]), self.global_step)
            
            if isinstance(self.logger, WandbLogger):
                self._log_to_wandb(tgt_refs[:5], hypotheses[:5], hypotheses_teacher[:5], split="val", epoch=self.current_epoch)
            
            print(len(tgt_refs), len(hypotheses))
            bleu = BLEU()
            bleu_s = bleu.corpus_score(hypotheses, [tgt_refs]).score
            self.log("val_bleu", bleu_s ,prog_bar=True)
            
            print(len(tgt_refs), len(hypotheses_teacher))
            bleu = BLEU()
            bleu_s = bleu.corpus_score(hypotheses_teacher, [tgt_refs]).score
            self.log("val_teacher_bleu", bleu_s ,prog_bar=True)

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
        outputs, labels = self(batch)
        
        loss = self.calc_loss(outputs, labels)

        self.log("test_loss", loss, sync_dist=True)

        self.test_decoded = self.generate(batch)
        self.test_step_outputs = [text + self.end_sym for text in batch['text_input']]
        tgt_refs = [str(item) for item in self.test_step_outputs]
        hypotheses = [str(item) for item in self.test_decoded]
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
    
    def teacher_forcing_generate(self, logits, idx_start_resp):
        predicted = torch.argmax(logits, dim=-1)
        generated_texts = [self.tokenizer.decode(seq[idx_start_resp:], skip_special_tokens=True) for seq in predicted]
        for idx,text in enumerate(generated_texts): # generated_texts:
            if '.' in text:
                dot_index = text.index('.')
                generated_texts[idx] = text[:dot_index + 1]
        return generated_texts
    
    def generate(self, samples):
        video_clips, atts_clips = samples['video_clips'], samples['atts_clips']
        bsz, clip_len, num_channels, num_frames, height, width = samples['video_clips'].shape

        clips_embeds = self.sign_encoder(video_clips.view(-1, num_channels, num_frames, height, width))['embds'].squeeze()
        clips_embeds = clips_embeds.view(bsz, clip_len, -1)
        clips_embeds = self.proj_visual(clips_embeds)
        clips_embeds, atts_clips = self.prompt_wrap(clips_embeds, atts_clips, self.prompt)
        
        bos = torch.ones([bsz, 1],
                        dtype=torch.long,
                        device=self.device) * self.tokenizer.bos_token_id
        
        bos_embeds = self.xglm.base_model.model.model.embed_tokens(bos)
        atts_bos = atts_clips[:, :1]

        inputs_embeds = torch.cat([bos_embeds, clips_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_clips], dim=1)
    
        gen_params = {"max_length": 64, "temperature": 1.0, "num_beams": 4}
        
        outputs = self.xglm.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            do_sample=False,
                            use_cache=False,
                            return_dict_in_generate=True,
                            output_scores=False,
                            **gen_params,
                            )
        
        generated_tokens = outputs["sequences"][:, inputs_embeds.shape[1]:]
        generated_texts = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_tokens]
        for idx,text in enumerate(generated_texts): # generated_texts:
            if '.' in text:
                dot_index = text.index('.')
                generated_texts[idx] = text[:dot_index + 1]
        return generated_texts
    
    def calc_loss(self, logit, y):
        outputs = logit[:, :-1, :]
        targets = y[:, 1:]
        vocab_siz =  outputs.size(-1)
        return self.criterion(outputs.reshape(-1, vocab_siz), targets.reshape(-1))
    
    def add_weight_decay(self, weight_decay, skip_list=()):
        """Custom method to create parameter groups with/without weight decay."""
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Ignore frozen parameters
            if 'gate' in name:
                no_decay.append(param)
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

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<VideoHere>')
            p_before_tokens = self.tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            
            p_after_tokens = self.tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            
            p_before_embeds = self.xglm.base_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.xglm.base_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)

            p_before_len = p_before_embeds.shape[1]
            p_after_len = p_after_embeds.shape[1]
            img_len = img_embeds.shape[1]

            wrapped_atts_img = torch.cat([
            torch.ones(batch_size, p_before_len, device=atts_img.device),  # p_before
            atts_img,  # img_embeds
            torch.ones(batch_size, p_after_len, device=atts_img.device)  # p_after
            ], dim=1)            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img