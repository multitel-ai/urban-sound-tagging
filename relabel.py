import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from training_system2 import DCASETALNetClassifier

from tqdm import tqdm
import config
from prepare_data.sonycust import SONYCUST_TALNet

# Arg Parser (will be moved ?)
parser = argparse.ArgumentParser()
parser.add_argument('--output_mode', type=str, default='both', choices=['coarse', 'fine', 'both'])
parser.add_argument('--path_to_SONYCUST', type=str, default=config.path_to_SONYCUST)
parser.add_argument('--path_to_ckpt', type=str, default=os.path.join(config.path_to_summaries, 'TALNetV3_TO23_1/checkpoints/epoch=15-auprc_macro_c=0.765.ckpt'))
parser.add_argument('--path_to_yaml', type=str, default=os.path.join(config.path_to_summaries, 'TALNetV3_TO23_1/lightning_logs/version_13/hparams_wo_es.yaml'))
parser.add_argument('--metadata', nargs='+', default=["latitude","longitude","week","day","hour"])
parser.add_argument('--consensus_threshold', type=float, default=0.0)
parser.add_argument('--one_hot_time', type=bool, default=False)
args = parser.parse_args()

# Dataset parameters
data_param={'mode':args.output_mode,
    'metadata':args.metadata,
    'one_hot_time':args.one_hot_time,
    'consensus_threshold':args.consensus_threshold, 
    'cleaning_strat':'All_unique'}
      
# Creating dataset
dataset = SONYCUST_TALNet(args.path_to_SONYCUST, **data_param)

test_dataloader = DataLoader(dataset, batch_size=64, num_workers=4)

# Creating model
model = DCASETALNetClassifier.load_from_checkpoint(args.path_to_ckpt, hparams_file=args.path_to_yaml)
model.freeze()
model.to('cuda:0')

print("Computing new labels...")
for i_batch, sample_batched in enumerate(tqdm(test_dataloader), 1):

    filenames = sample_batched['file_name']
    inputs = sample_batched['input_vector'].float().cuda()
    metas = sample_batched['metadata'].float().cuda()
    # forward + eval
    outputs = model(inputs, metas)[0]
    if i_batch == 1:
        filename_array = [] + filenames
        output_array = np.array(outputs.cpu())
    else:
        filename_array += filenames
        output_array = np.vstack((output_array, outputs.cpu().numpy()))

print("Done")

pred_df = pd.DataFrame(columns=['audio_filename']+dataset.idlabel_dict['coarse']+dataset.idlabel_dict['fine'])
pred_df['audio_filename'] = filename_array
pred_df[dataset.idlabel_dict['coarse']+dataset.idlabel_dict['fine']] = output_array
pred_df.to_csv(os.path.join(config.path_to_SONYCUST, "relabel.csv"), index = False, header=True)
# "best2.csv"

print("Saved")