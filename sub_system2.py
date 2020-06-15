import sys; sys.argv=['']; del sys
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Arg Parser (will be moved ?)
parser = argparse.ArgumentParser()
parser.add_argument('--output_mode', type=str, default='both', choices=['coarse', 'fine', 'both'])
parser.add_argument('--path_to_SONYCUST', type=str, default=config.path_to_SONYCUST)
parser.add_argument('--metadata', nargs='+', default=["latitude","longitude","week","day","hour"])
parser.add_argument('--consensus_threshold', type=float, default=0.0)
parser.add_argument('--one_hot_time', type=bool, default=False)
args = parser.parse_args()

# Dataset parameters
data_param={'mode':args.output_mode}
      
# Creating dataset
dataset = SONYCUST_TALNet(args.path_to_SONYCUST, 
    metadata=args.metadata,one_hot_time=args.one_hot_time,consensus_threshold=args.consensus_threshold, **data_param) #, cleaning_strat='All_unique'
train_dataset, valid_dataset, test_dataset = dataset.train_validation_test_split()

test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=4)
#valid_dataloader = DataLoader(valid_dataset, batch_size=64, num_workers=4)

PATH = os.path.join(config.path_to_summaries, 'TALNetV3_TO23_1/checkpoints/epoch=19-auprc_macro_c=0.801.ckpt')
hparams_file = os.path.join(config.path_to_summaries, 'TALNetV3_TO23_1/lightning_logs/version_14/hparams_wo_es.yaml')
# Creating model
model = DCASETALNetClassifier.load_from_checkpoint(PATH, hparams_file=hparams_file)
print(count_parameters(model))
model.freeze()
model.to('cuda:0')

print("computing")
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

print("done")

pred_df = pd.DataFrame(columns=['audio_filename']+dataset.idlabel_dict['coarse']+dataset.idlabel_dict['fine'])
pred_df['audio_filename'] = filename_array
pred_df[dataset.idlabel_dict['coarse']+dataset.idlabel_dict['fine']] = output_array
#exp_path = os.path.join(args.path_to_summaries, 'experiment1')
pred_df.to_csv(os.path.join(config.path_to_SONYCUST, "pred_test_TALNETV3.csv"), index = False, header=True)