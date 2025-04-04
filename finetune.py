import torch
import torch.backends.cudnn as cudnn
from models.Finetune_Model import FineTuneModel
from collections import OrderedDict

from models.utils import manage_directory, SaveBestModelOnNEpochs
from dataset.slt_dataset import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from transformers import MBartTokenizer
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
    parser.add_argument('--eval_freq', default=2, type=int, metavar='N', 
                        help='The frequency of metric evaluation, e.g Bleu score')
    ##################Transformer and Encoder Params####################################   
    parser.add_argument('--tokenizer_path', type=str, default="pretrain_models/MBart_trimmed",
                        help='Path to the MBart tokenizer.')
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to the encoder checkpoint.')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to the model checkpoint.')
    ##################Data Params##########################################################
    parser.add_argument('--text_path', type=str, default="data/labels", 
                        help='Path to the text data.')
    parser.add_argument('--qa_csv_path', type=str, default=None,
                        help='Path to the csv file.')
    parser.add_argument('--data_config', type=str, default='configs/csldaily-config.yaml',
                        help='Path to the data config file.')
    parser.add_argument('--secret_config', type=str, default='configs/secret.yaml',
                        help='Path to the secret config file.')
    
    parser.add_argument('--label_smoothing', type=float, default=0.2)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--data_ver', type=int, default=0, help='Data version.')
    parser.add_argument('--run_ver', type=int, default=0, help='Data version.')
    parser.add_argument('--warmup', type=float, default=0.05, help='Warmup')
    
    parser.add_argument('--logger', type=str, default='tensorboard', help='Logger type.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_dir', type=str, default="finetune_new", help='Output directory.')
    parser.add_argument('--log_dir', type=str, default="finetune_new", help='Output directory.')
    parser.add_argument('--save_csv', type=str, default="csv_outputs/", help='Output directory.')

    # * Learning rate schedule parameters
    parser.add_argument('--lr', type=float, default=6e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    return parser

def setupWandB(wandb_configs, storage=None):
    os.environ.update(wandb_configs)
    if storage is not None:
        os.environ['WANDB_CACHE_DIR'] = storage+'/wandb/cache'
        os.environ['WANDB_CONFIG_DIR'] = storage+'/wandb/config'


def main(args):
    pl.seed_everything(args.seed)
    # fix the seed for reproducibility
    cudnn.benchmark = True

    with open(args.data_config, 'r') as file:
            config = yaml.safe_load(file)
    with open(args.secret_config, 'r') as file:
            secret_config = yaml.safe_load(file)

    args.text_path = config['data']['labels']
    args.tokenizer_path = config['model']['tokenizer']
    args.output_dir = config['save']['output']
    args.log_dir = config['save']['output']
    args.save_csv = config['save']['csv']
    args.save_csv = args.save_csv.split("/")[0] + str(args.run_ver) + "/"
    if args.model_ckpt is None:
        args.model_ckpt = config['training']['ckpt_path']
    print(f'Loaded checkpoint is : {args.model_ckpt}')
    # set logger
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.logger == 'wandb':
        save_dir=f'{args.log_dir}/log_{current_time}_{args.run_ver}'
        setupWandB(secret_config["WANDB_CONFIG"],storage=save_dir)
        logger = WandbLogger(project="dino-test", config=vars(args))
    else:
        logger = TensorBoardLogger(save_dir=f'{args.log_dir}/log_{current_time}', name="Sign2GPT")
    dirpath = f'{args.output_dir}/run_{current_time}_{args.run_ver}'
    print("Current Time = {}".format(current_time)) 
    
    # set callbacks
    # checkpoint_callback = ModelCheckpoint(
    # save_top_k=1,
    # save_last=True,
    # monitor="val_bleu",
    # every_n_epochs=args.eval_freq + 1,
    # mode="max",
    # dirpath=dirpath,
    # filename="best-{epoch:03d}-{val_loss:.3f}-{val_bleu:.3f}",
    # )
    checkpoint_callback = SaveBestModelOnNEpochs(
        save_every_n_epochs=args.eval_freq, 
        monitor="val_bleu", mode="max", 
        dirpath=dirpath)
    early_stop = EarlyStopping("val_loss", patience=args.epochs, mode="min", verbose=True)
    callbacks = [checkpoint_callback]
    manage_directory(args.save_csv)
    model = FineTuneModel(
                config=args.data_config,
                args=args,
                eval_freq=args.eval_freq,
                csv_dire=args.save_csv,
                label_smoothing=args.label_smoothing,
                num_beams=args.num_beams,
                warmup=args.warmup,)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'], src_lang = 'zh_CN', tgt_lang = 'zh_CN')
    data_module = DataModule(
                root_text_path=args.text_path, 
                data_config=args.data_config,
                qa_csv_path=args.qa_csv_path,
                tokenizer=tokenizer,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                data_ver=args.data_ver)

    trainer = pl.Trainer(
        strategy="ddp",
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
    trainer.fit(model, data_module)
    best_model_path = checkpoint_callback.last_best_checkpoint_path
    print(f"Best model path: {best_model_path}")
    best_model = FineTuneModel.load_from_checkpoint(best_model_path)
    trainer.validate(best_model, data_module)
    trainer.test(best_model, data_module)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
