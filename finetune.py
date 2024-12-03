import torch
import torch.backends.cudnn as cudnn
from models.Finetune_Model import FineTuneModel

from models.utils import manage_directory
from dataset.slt_dataset import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml

import os
import argparse
from pathlib import Path
from datetime import datetime

torch.set_float32_matmul_precision("medium")

def get_args_parser():
    parser = argparse.ArgumentParser('Sign2GPT', add_help=False)
    
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--num_gpus', default=1, type=int, metavar='N', help='number of gpus per node')
    parser.add_argument('--eval_freq', default=10, type=int, metavar='N', 
                        help='The frequency of metric evaluation, e.g Bleu score')
    ##################Transformer and Encoder Params####################################    
    parser.add_argument('--mbart_path', type=str, default="/home/sobhan/Documents/Code/GFSLT-VLP/pretrain_models/MBart_trimmed",
                        help='Path to the MBart model.')
    parser.add_argument('--tokenizer_path', type=str, default="/home/sobhan/Documents/Code/GFSLT-VLP/pretrain_models/MBart_trimmed",
                        help='Path to the MBart tokenizer.')
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to the encoder checkpoint.')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to the model checkpoint.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    ##################Data Params##########################################################
    parser.add_argument('--text_path', type=str, default="/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/labels", 
                        help='Path to the text data.')
    parser.add_argument('--qa_csv_path', type=str, default=None,
                        help='Path to the csv file.')
    parser.add_argument('--data_config', type=str, default='configs/config.yaml',
                        help='Path to the data config file.')  
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    parser.add_argument('--data_ver', type=int, default=0, help='Data version.')
    
    parser.add_argument('--logger', type=str, default='tensorboard', help='Logger type.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory.')
    parser.add_argument('--log_dir', type=str, default="output", help='Output directory.')
    parser.add_argument('--save_csv', type=str, default="csv_outputs/", help='Output directory.')
    return parser

WANDB_CONFIG = {"WANDB_API_KEY": "1af8cc2a4ed95f2ba66c31d193caf3dd61c3a41f", "WANDB_IGNORE_GLOBS":"*.patch", 
                "WANDB_DISABLE_CODE": "true", "TOKENIZERS_PARALLELISM": "false"}
def setupWandB(storage=None):
    os.environ.update(WANDB_CONFIG)
    if storage is not None:
        os.environ['WANDB_CACHE_DIR'] = storage+'/wandb/cache'
        os.environ['WANDB_CONFIG_DIR'] = storage+'/wandb/config'

def main(args):
    pl.seed_everything(args.seed)
    # fix the seed for reproducibility
    cudnn.benchmark = True

    # set logger
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.logger == 'wandb':
        save_dir=f'{args.log_dir}/log_{current_time}'
        setupWandB(storage=save_dir)
        logger = WandbLogger(project="Shit-Test", config=vars(args))
    else:
        logger = TensorBoardLogger(save_dir=f'{args.log_dir}/log_{current_time}', name="Sign2GPT")
    dirpath = f'{args.output_dir}/run_{current_time}'
    print("Current Time = {}".format(current_time)) 
    
    # set callbacks
    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    save_last=True,
    monitor="val_bleu",
    every_n_epochs=args.eval_freq + 1,
    mode="max",
    dirpath=dirpath,
    filename="best-{epoch:03d}-{val_loss:.3f}-{val_bleu:.3f}",
    )
    early_stop = EarlyStopping("val_loss", patience=args.epochs, mode="min", verbose=True)
    callbacks = [checkpoint_callback]
    manage_directory(args.save_csv)
    model = FineTuneModel(
                config=args.data_config,
                lr=args.lr, 
                encoder_ckpt=args.encoder_ckpt,
                eval_freq=args.eval_freq,
                csv_dire=args.save_csv)
    with open(args.data_config, 'r') as file:
            config = yaml.safe_load(file) 
    # model.transfer_specific_features('best-epoch=088-val_loss=0.071-train_loss=0.010.ckpt',['sign_encoder','proj_visual'])
    checkpoint = torch.load(config['training']['ckpt_path'], map_location='cpu')

    new_state_dict = {}
    # Inspect parameter names in the checkpoint
    # print("Checkpoint parameters:")
    for name, key in checkpoint['state_dict'].items():
        if 'pretrain_model.model_images.model' in name:
            name = name.replace('pretrain_model.model_images.model', 'sign_encoder')
            new_state_dict[name] = key
        if 'pretrain_model.model_images.trans_encoder' in name:
            name = name.replace('pretrain_model.model_images.trans_encoder', 'mbart')
            if '.model.' in name:
                name = name.replace('.model.', '.model.model.encoder.')
            new_state_dict[name] = key
        # print(name, key.shape)
        # if 'final_logits_bias' in name or 'shared' in name or 'lm_head' in name:
            # print(name)

    # *replace the word embedding
    model_dict = torch.load(config['model']['transformer']+'/pytorch_model.bin', map_location='cpu')
    for k, v in model_dict.items():
        if 'decoder.embed_tokens.weight' in k:
            k = 'mbart.base_model.model.' + k
            new_state_dict[k] = v
        if 'decoder.embed_positions.weight' in k:
            k = 'mbart.base_model.model.' + k
            new_state_dict[k] = v
    
    # ret = model.load_state_dict(new_state_dict, strict=False)
    # print('Missing keys: \n', '\n'.join(ret.missing_keys))
    # print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    data_module = DataModule(
                root_text_path=args.text_path, 
                data_config=args.data_config,
                qa_csv_path=args.qa_csv_path,
                tokenizer_path=args.tokenizer_path,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                data_ver=args.data_ver)
    
    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true",
        sync_batchnorm=True,
        logger=logger,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices="auto",
        min_epochs=1,
        max_epochs=args.epochs,
        precision=16,
        callbacks=callbacks,
    )
    if args.model_ckpt is None:
        trainer.fit(model, data_module)
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        best_model = FineTuneModel.load_from_checkpoint(best_model_path)
        trainer.validate(best_model, data_module)
        trainer.test(best_model, data_module)
    else:
        best_model = FineTuneModel.load_from_checkpoint(args.model_ckpt)
        trainer.test(best_model, data_module)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
