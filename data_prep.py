import argparse
from prepare_data.sonycust import SONYCUST
import config

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--download", help="download dataset",
      action="store_true")
   parser.add_argument("--mel", help="precompute mels for TALNet",
      action="store_true")
   parser.add_argument("--resample", help="pecompute resampled wavs for CNN14",
      action="store_true")
   args = parser.parse_args()

   if args.download:
      print("Downloading Dataset")
      dataset = SONYCUST(config.path_to_SONYCUST, 'coarse', download=True)
   else:
      dataset = SONYCUST(config.path_to_SONYCUST, 'coarse')

   if args.mel:
      print("Computing mel spectrograms for TALNet")
      dataset.compute_melspec()

   if args.resample:
      print("Resampling wavs for CNN14")
      dataset.precompute_wav()