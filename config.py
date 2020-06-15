import os

PATH ="../data/"

# General directories
path_to_SONYCUST = PATH # os.path.join(PATH,"SONYC-UST")
path_to_summaries = os.path.join(PATH,"summaries")

# Pretrained weights
audioset = os.path.join(path_to_SONYCUST, "model/TALNet.pt")
audiosetCNN = os.path.join(path_to_SONYCUST, "model/Cnn14_mAP=0.431.pth")

# General things for SONYC-UST
path_to_annotation = os.path.join(path_to_SONYCUST, "annotations.csv")
path_to_taxonomy = os.path.join(path_to_SONYCUST, "dcase-ust-taxonomy.yaml")
wav_dir = os.path.join(path_to_SONYCUST, "audio")

# DCASE Baseline
emb_dir = os.path.join(path_to_SONYCUST, "embedding")

# TALNet
mel_dir = os.path.join(path_to_SONYCUST, "melTALNet")
