import os
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from training_system3 import DCASETALNetClassifier

from tqdm import tqdm
import config
from prepare_data.sonycust import SONYCUST_TALNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Arg Parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_mode', type=str, default='both', choices=['coarse', 'fine', 'both'])
parser.add_argument('--path_to_SONYCUST', type=str, default=config.path_to_SONYCUST)
parser.add_argument('--path_to_ckpt', type=str, default=os.path.join(config.path_to_summaries, 'TALNetV3_TO23_2/checkpoints/epoch=17-auprc_macro_c=0.811.ckpt'))
parser.add_argument('--path_to_yaml', type=str, default=os.path.join(config.path_to_summaries, 'TALNetV3_TO23_2/lightning_logs/version_0/hparams_wo_es.yaml'))
parser.add_argument('--metadata', nargs='+', default=["latitude","longitude","week","day","hour"])
parser.add_argument('--consensus_threshold', type=float, default=0.0)
parser.add_argument('--one_hot_time', type=bool, default=False)
args = parser.parse_args()

# Dataset parameters
data_param={'mode':args.output_mode,
    'metadata':args.metadata,
    'one_hot_time':args.one_hot_time, 
    'consensus_threshold':args.consensus_threshold}
      
# Creating dataset
dataset = SONYCUST_TALNet(args.path_to_SONYCUST, **data_param)
train_dataset, valid_dataset, test_dataset = dataset.train_validation_test_split()

test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=4)

# Creating model
model = DCASETALNetClassifier.load_from_checkpoint(args.path_to_ckpt, hparams_file=args.path_to_yaml)
model.freeze()
model.to('cuda:0')
print("Number of parameters : ", count_parameters(model))

print("Computing outputs...")
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

X_mask = ~np.array([0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0], dtype=bool)
coarse, fine = output_array[:,0:8], output_array[:,8:37]
fine = fine[:,X_mask]
output_array = np.concatenate((coarse, fine),1)

pred_df = pd.DataFrame(columns=['audio_filename']+dataset.idlabel_dict['coarse']+dataset.idlabel_dict['fine'])
pred_df['audio_filename'] = filename_array
pred_df[dataset.idlabel_dict['coarse']+dataset.idlabel_dict['fine']] = output_array
pred_df.to_csv(os.path.join(config.path_to_SONYCUST, "Arnault_MULT_task5_3.output.csv"), index = False, header=True)
# pred_test_TALNETV3_4.csv

print("Saved")