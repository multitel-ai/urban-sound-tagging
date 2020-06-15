from argparse import ArgumentParser
import os
# FILTER WARNINGS
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss

from optimizer.lookahead import Lookahead
from optimizer.ralamb import Ralamb

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from albumentations import Compose, ShiftScaleRotate, GridDistortion, Cutout
from albumentations.pytorch import ToTensor

from utils.metrics import *
import pandas as pd
from utils.metrics_dcase import evaluate, micro_averaged_auprc, macro_averaged_auprc
from prepare_data.sonycust import SONYCUST_TALNet
from models.TALNet import TALNetV2_meta, TALNetV2
from losses.DCASEmaskedLoss import *
import config

def mixup_data(x, meta, y, alpha=1.0, use_cuda=True):
   '''Returns mixed inputs, pairs of targets, and lambda'''
   if alpha > 0:
      lam = np.random.beta(alpha, alpha)
   else:
      lam = 1
   batch_size = x.size()[0]
   index = torch.randperm(batch_size)
   mixed_x = lam * x + (1 - lam) * x[index, :]
   mixed_meta = lam * meta + (1 - lam) * meta[index, :]
   y_a, y_b = y, y[index]
   return mixed_x, mixed_meta, y_a, y_b, lam

class DCASETALNetClassifier(LightningModule):
   def __init__(self, hparams):
      super().__init__()
      # Save hparams for later
      self.hparams=hparams
      
      self.num_classes_dict = {'coarse':8,
      'fine':23,
      'both':31
      }
      # To avoid calling prepare_data 2 times
      self.data_prepared = False
      self.prepare_data()
      num_meta = len(self.dataset[0]['metadata'])

      # Model parameters
      model_param = {
         'n_conv_layers':hparams.n_conv_layers,
         'kernel_size':(3,3),
         'n_pool_layers':hparams.n_pool_layers,
         'embedding_size':hparams.embedding_size,
         'norm':'GN',
         'conv_pool_strat':'max',
         'conv_activation':hparams.conv_activation,
         'pooling':hparams.pooling,
         'dropout':hparams.dropout,
         'n_head':hparams.n_head,
         'd_kv':hparams.d_kv,
         'dropout_transfo':hparams.dropout_transfo,
         'num_meta':num_meta,
         'meta_emb':hparams.meta_embedding_size,
         'num_mels':hparams.num_mels,
         'num_classes':self.num_classes_dict[self.hparams.output_mode]
      }
      self.model = TALNetV2_meta(**model_param)
      self.loss_c = BCELoss(reduction='none')
      self.loss_f = Masked_loss(BCELoss(reduction='none'))

   @staticmethod
   def add_model_specific_args(parent_parser):
      parser = ArgumentParser(parents=[parent_parser], add_help=False)
      parser.add_argument('--output_mode', type=str, default='both', choices=['coarse', 'fine', 'both'])
      parser.add_argument('--dropout', type = float, default = 0.0)
      parser.add_argument('--pooling', type = str, default = 'att', choices = ['max', 'ave', 'lin', 'exp', 'att', 'auto'])
      parser.add_argument('--n_conv_layers', type=int, default=10)
      parser.add_argument('--kernel_size', type = str, default = '3')
      parser.add_argument('--n_pool_layers', type=int, default=5)
      parser.add_argument('--embedding_size', type = int, default = 1024)
      parser.add_argument('--meta_embedding_size', type = int, default = 128)
      parser.add_argument('--batch_norm', type = bool, default = True)
      parser.add_argument('--conv_activation', type = str, default = 'mish', choices = ['relu', 'prelu', 'leaky_relu', 'mish'])
      parser.add_argument('--n_head', type = int, default = 8)
      parser.add_argument('--d_kv', type = int, default = 128)
      parser.add_argument('--dropout_transfo', type = float, default = 0.0)

      parser.add_argument('--batch_size', type=int, default=64)
      parser.add_argument('--init_lr', type=float, default=1e-3)
      parser.add_argument('--weight_decay', type=float, default=1e-5)
      parser.add_argument('--alpha', type = float, default = 0.5)
      parser.add_argument('--start_mixup', type=int, default=-1)

      parser.add_argument('--num_mels', type = int, default = 64)
      parser.add_argument('--path_to_SONYCUST', type=str, default=config.path_to_SONYCUST)
      parser.add_argument('--metadata', nargs='+', default=["latitude","longitude","week","day","hour"])
      parser.add_argument('--consensus_threshold', type=float, default=0.0)
      parser.add_argument('--one_hot_time', type=bool, default=False)

      return parser
   
   def forward(self, x, meta):
      x = self.model(x, meta)
      return x
   
   def prepare_data(self):
      # To avoid calling prepare_data 2 times
      if self.data_prepared:
         return True

      transformation_list = [
         ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
         GridDistortion(),
         Cutout()
      ]
      albumentations_transform = Compose(transformation_list)

      # Dataset parameters
      data_param={
         'mode':self.hparams.output_mode, 
         'transform':albumentations_transform,
         'metadata':self.hparams.metadata,
         'one_hot_time':self.hparams.one_hot_time,
         'consensus_threshold':self.hparams.consensus_threshold,
         'cleaning_strat':'Relabeled', 
         'relabeled_name':'best2.csv'
         }
      
      # Creating dataset
      self.dataset = SONYCUST_TALNet(self.hparams.path_to_SONYCUST, **data_param)
      self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.train_validation_test_split()

      self.data_prepared = True

   def train_dataloader(self):
      return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

   def val_dataloader(self):
      return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)

   def configure_optimizers(self):
      base_optim_param = {'lr':self.hparams.init_lr}
      base_optim = Ralamb(self.model.parameters(), **base_optim_param)
      optim_param = {'k':5,
                      'alpha':0.5}
      optimizer = Lookahead(base_optim, **optim_param)
      return optimizer
   
   def training_step(self, batch, batch_idx):
      # Get input vector and labels, do not forget float()
      data, meta, target_c, target_f = batch['input_vector'].float(), batch['metadata'].float(), batch['label']['coarse'].float(), batch['label']['full_fine'].float()
      target = torch.cat([target_c, target_f],1)

      # Do mixup if current epoch >= start mixup
      if self.current_epoch >= self.hparams.start_mixup:
         data, meta, target_a, target_b, lam = mixup_data(data, meta, target, self.hparams.alpha)
         target_a_c, target_a_f = torch.split(target_a, [8,29],1)
         target_b_c, target_b_f = torch.split(target_b, [8,29],1)

      # Forward pass
      output, frame_prob = self.forward(data, meta)
      outputs_c, outputs_f = torch.split(output, [8,23],1)

      # Compute loss of the batch
      if self.current_epoch >= self.hparams.start_mixup:
         loss = lam * torch.cat([self.loss_c(outputs_c, target_a_c).mean(0), self.loss_f(outputs_f, target_a_f)], 0).mean() + \
            (1 - lam) * torch.cat([self.loss_c(outputs_c, target_b_c).mean(0), self.loss_f(outputs_f, target_b_f)], 0).mean()
      else:
         loss = torch.cat([self.loss_c(outputs_c, target_c).mean(0), self.loss_f(outputs_f, target_f)], 0).mean()
      return {'loss':loss, 'log': {'1_loss/train_loss': loss}}
   
   def validation_step(self, batch, batch_idx):
      # Get input vector and labels, do not forget float()
      filename, data, meta, target_c, target_f = batch['file_name'], batch['input_vector'].float(), batch['metadata'].float(), batch['label']['coarse'].float(), batch['label']['full_fine'].float()
      target = torch.cat([target_c, target_f],1)
      # Forward pass
      output = self.forward(data, meta)[0]
      outputs_c, outputs_f = torch.split(output, [8,23],1)
      # Compute loss of the batch
      loss = torch.cat([self.loss_c(outputs_c, target_c).mean(0), self.loss_f(outputs_f, target_f)], 0)
      return {'val_loss':loss, 'output':output, 'target':target, 'filename': filename}
   
   def validation_epoch_end(self, outputs):
      val_loss = torch.cat([o['val_loss'] for o in outputs], 0).mean()
      all_outputs = torch.cat([o['output'] for o in outputs], 0)
      all_outputs_c, all_outputs_f = torch.split(all_outputs, [8,23],1) 
      all_outputs_c, all_outputs_f = all_outputs_c.cpu().numpy(), all_outputs_f.cpu().numpy()

      all_targets = torch.cat([o['target'] for o in outputs], 0)
      all_targets_c, all_targets_f = torch.split(all_targets, [8,29],1)
      all_targets_c, all_targets_f = all_targets_c.cpu().numpy(), all_targets_f.cpu().numpy()

      filename_array = [f for o in outputs for f in o['filename']]

      # Trick to use DCASE metrics : we save a csv file
      pred_df = pd.DataFrame(columns=['audio_filename']+self.dataset.idlabel_dict['coarse']+self.dataset.idlabel_dict['full_fine'])
      pred_df['audio_filename'] = filename_array
      pred_df[self.dataset.idlabel_dict['coarse']] = all_outputs_c
      pred_df[self.dataset.idlabel_dict['fine']] = all_outputs_f
      pred_df.to_csv(os.path.join(config.path_to_SONYCUST, "temp.csv"), index = False, header=True)
      
      log = {'1_loss/val_loss': val_loss}
      for mode in ['coarse', 'fine']:
         df_dict = evaluate(os.path.join(config.path_to_SONYCUST, "temp.csv"),
                           config.path_to_annotation,
                           config.path_to_taxonomy,
                           mode)
         if mode == 'coarse':
            auprc_micro_c, eval_df = micro_averaged_auprc(df_dict, return_df=True)
            auprc_macro_c, auprc_classes = macro_averaged_auprc(df_dict, return_classwise=True)
            thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).to_numpy().nonzero()[0][0]
            F1_micro_c = eval_df["F"][thresh_0pt5_idx]
            log_temp = {'2_valid_coarse/1_auprc_macro':auprc_macro_c,
            '2_valid_coarse/2_auprc_micro':auprc_micro_c,
            '2_valid_coarse/3_F1_micro':F1_micro_c}
         else:
            auprc_micro_f, eval_df = micro_averaged_auprc(df_dict, return_df=True)
            auprc_macro_f, auprc_classes = macro_averaged_auprc(df_dict, return_classwise=True)
            thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).to_numpy().nonzero()[0][0]
            F1_micro_f = eval_df["F"][thresh_0pt5_idx]
            log_temp = {'2_valid_fine/1_auprc_macro':auprc_macro_f,
            '2_valid_fine/2_auprc_micro':auprc_micro_f,
            '2_valid_fine/3_F1_micro':F1_micro_f
            }
         log.update(log_temp)
      
      tqdm_dict = {'auprc_macro_c':auprc_macro_c, 'auprc_macro_f':auprc_macro_f, 
         'val_loss': val_loss}
      return {'progress_bar':tqdm_dict, 'log': log}

def main(hparams):
   seed_everything(hparams.seed)
   MAIN_DIR = os.path.join(config.path_to_summaries,'TALNetV2_LOSS_TRYHARD/')
   model = DCASETALNetClassifier(hparams)
   early_stop_callback = EarlyStopping(
      monitor='auprc_macro',
      mode='max',
      patience=25,
      strict=False,
      verbose=False)

   trainer = Trainer.from_argparse_args(hparams,
      early_stop_callback=early_stop_callback,
      checkpoint_callback=False,
      default_root_dir=MAIN_DIR,
      logger=True,
      #fast_dev_run=True
      #train_percent_check=0.001,
      num_sanity_val_steps=0,
      )
   # Have to declare here checkpoint_callback because of a bug
   checkpoint_callback_1 = ModelCheckpoint(
      monitor='auprc_macro_c',
      mode='max',
      filepath=os.path.join(MAIN_DIR,'checkpoints/{epoch}-{auprc_macro_c:.3f}')
      )
   trainer.checkpoint_callback = checkpoint_callback_1
   trainer.fit(model)

if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('--seed', type = int, default = 42)
   parser = DCASETALNetClassifier.add_model_specific_args(parser)
   parser = Trainer.add_argparse_args(parser)
   hparams = parser.parse_args()
   main(hparams)
   # tensorboard --host localhost --logdir /Users/augustinarnault/Downloads/lightning_logs