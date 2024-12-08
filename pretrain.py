import torch
import torch.backends.cudnn as cudnn
from models.Pretrain_Model import PreTrainModel

from models.utils import manage_directory
from dataset.slt_dataset import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import os
import argparse
from pathlib import Path
from datetime import datetime

torch.set_float32_matmul_precision("medium")

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP', add_help=False)
    
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--num_gpus', default=1, type=int, metavar='N', help='number of gpus per node')
    parser.add_argument('--eval_freq', default=10, type=int, metavar='N', 
                        help='The frequency of metric evaluation, e.g Bleu score')
    ##################Transformer and Encoder Params####################################    
    parser.add_argument('--mbart_path', type=str, default="pretrain_models/MBart_trimmed",
                        help='Path to the MBart model.')
    parser.add_argument('--tokenizer_path', type=str, default="pretrain_models/MBart_trimmed",
                        help='Path to the MBart tokenizer.')
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to the encoder checkpoint.')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Path to the model checkpoint.')
    # parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    ##################Data Params##########################################################
    parser.add_argument('--text_path', type=str, default="data/labels", 
                        help='Path to the text data.')
    parser.add_argument('--qa_csv_path', type=str, default=None,
                        help='Path to the csv file.')
    parser.add_argument('--data_config', type=str, default='configs/config.yaml',
                        help='Path to the data config file.')  
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--data_ver', type=int, default=0, help='Data version.')
    
    parser.add_argument('--logger', type=str, default='tensorboard', help='Logger type.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_dir', type=str, default="output_new", help='Output directory.')
    parser.add_argument('--log_dir', type=str, default="_new", help='Output directory.')
    parser.add_argument('--save_csv', type=str, default="csv_outputs/", help='Output directory.')
    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
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
        logger = WandbLogger(project="GFSLT", config=vars(args))
    else:
        logger = TensorBoardLogger(save_dir=f'{args.log_dir}/log_{current_time}', name="CLIP")
    dirpath = f'{args.output_dir}/run_{current_time}'
    print("Current Time = {}".format(current_time)) 
    
    # set callbacks
    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    save_last=True,
    monitor="val_loss",
    mode="min",
    dirpath=dirpath,
    filename="best-{epoch:03d}-{val_loss:.3f}",
    )
    
    early_stop = EarlyStopping("val_loss", patience=args.epochs, mode="min", verbose=True)
    callbacks = [checkpoint_callback]
    # manage_directory(args.save_csv)
    model = PreTrainModel(
                config=args.data_config,
                args=args, 
                )
    # model.transfer_specific_features('best-epoch=088-val_loss=0.071-train_loss=0.010.ckpt',['sign_encoder','proj_visual'])

    data_module = DataModule(
                root_text_path=args.text_path, 
                data_config=args.data_config,
                qa_csv_path=args.qa_csv_path,
                tokenizer_path=args.tokenizer_path,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                data_ver=args.data_ver,
                training_Refurbish=True)

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
        best_model = PreTrainModel.load_from_checkpoint(best_model_path)
        trainer.validate(best_model, data_module)
        trainer.test(best_model, data_module)
    else:
        best_model = PreTrainModel.load_from_checkpoint(args.model_ckpt)
        trainer.test(best_model, data_module)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
