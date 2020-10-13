import os
import shutil
import pickle

import oyaml as yaml
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed

from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets.utils import download_url, download_and_extract_archive

NUM_HOURS = 24
NUM_DAYS = 7
NUM_WEEKS = 52

def one_hot(idx, num_items):
   return [(0.0 if n != idx else 1.0) for n in range(num_items)]

def remove_duplicates(dataframe,idlabel_presence_dict, consensus_threshold=0.01):
   all_idlabel_presence = idlabel_presence_dict['full_fine']+idlabel_presence_dict['coarse']

   annotation_filtered = dataframe.copy()
   # One hot encode presence
   annotation_filtered.loc[:,all_idlabel_presence] = (annotation_filtered[all_idlabel_presence]>0).astype(float)

   # Create a dictionnary for agregation strategie
   # For everything that is not a label presence, take first
   agreg_dict = {label:'first' for label in dataframe.columns.values}
   # Sum on all label presence
   for label in all_idlabel_presence:
      agreg_dict[label] =  'mean'
   # Remove audio_filename as it is the key used by groupby
   del agreg_dict['audio_filename']
   
   # Aggregate according to dict strategie
   annotation_filtered = annotation_filtered.groupby('audio_filename').agg(agreg_dict).reset_index()

   # One hot encode presence based on consensus_treshold
   annotation_filtered.loc[:,all_idlabel_presence] = (annotation_filtered[all_idlabel_presence]>consensus_threshold).astype(float)

   return annotation_filtered

def cleaning_annotation_baseline(dataframe,idlabel_presence_dict, consensus_threshold=0.01):
   """ Removes duplicate and aggregate labels
      Returns:
         annotation_filtered (pd.DataFrame): Annotations cleaned followind DCASE baseline strategie
   """
   all_idlabel_presence = idlabel_presence_dict['full_fine']+idlabel_presence_dict['coarse']
   # Mask using the 2 conditions used by DCASE Baseline
   cond1 = (dataframe["split"]=="train") & (dataframe["annotator_id"]>0)
   cond2 = (dataframe["split"]!="train") & (dataframe["annotator_id"]==0)
   # Filter annotation COND1 + COND2
   annotation_filtered = dataframe[cond1 | cond2].copy()

   # One hot encode presence
   annotation_filtered.loc[:,all_idlabel_presence] = (annotation_filtered[all_idlabel_presence]>0).astype(float)

   # Create a dictionnary for agregation strategie
   # For everything that is not a label presence, take first
   agreg_dict = {label:'first' for label in dataframe.columns.values}
   # Sum on all label presence
   for label in all_idlabel_presence:
      agreg_dict[label] =  'mean'
   # Remove audio_filename as it is the key used by groupby
   del agreg_dict['audio_filename']
   
   # Aggregate according to dict strategie
   annotation_filtered = annotation_filtered.groupby('audio_filename').agg(agreg_dict).reset_index()

   # One hot encode presence based on consensus_treshold
   annotation_filtered.loc[:,all_idlabel_presence] = (annotation_filtered[all_idlabel_presence]>consensus_threshold).astype(float)

   return annotation_filtered

def clean_annotation_and_use_relabel(dataframe,idlabel_presence_dict, idlabel_dict, relabel_dir, consensus_threshold=0.01):
   """ Removes duplicate and aggregate labels
      Returns:
         annotation_filtered (pd.DataFrame): Annotations cleaned followind DCASE baseline strategie
   """
   all_idlabel_presence = idlabel_presence_dict['full_fine']+idlabel_presence_dict['coarse']
   # Mask using the 2 conditions used by DCASE Baseline
   cond1 = (dataframe["split"]=="train")
   cond2 = (dataframe["split"]!="train") & (dataframe["annotator_id"]==0)
   valid_exclusion = dataframe.loc[cond2, 'audio_filename']
   cond3 = (dataframe["split"]=="validate") & (dataframe["annotator_id"]!=0) & ~dataframe["audio_filename"].isin(valid_exclusion)
   dataframe.loc[cond3, 'split']= 'train'
   # Filter annotation COND1 + COND2
   annotation_filtered = dataframe[cond1 | cond2 | cond3].copy()

   # One hot encode presence
   annotation_filtered.loc[:,all_idlabel_presence] = (annotation_filtered[all_idlabel_presence]>0).astype(float)

   # Create a dictionnary for agregation strategie
   # For everything that is not a label presence, take first
   agreg_dict = {label:'first' for label in dataframe.columns.values}
   # Sum on all label presence
   for label in all_idlabel_presence:
      agreg_dict[label] =  'mean'
   # Remove audio_filename as it is the key used by groupby
   del agreg_dict['audio_filename']
   
   # Aggregate according to dict strategie
   annotation_filtered = annotation_filtered.groupby('audio_filename').agg(agreg_dict)#.reset_index()

   relabeled_df = pd.read_csv(relabel_dir).set_index('audio_filename')
   relabeled_df[idlabel_dict['coarse']+idlabel_dict['fine']] = (relabeled_df[idlabel_dict['coarse']+idlabel_dict['fine']]>0.7).astype(float)
   rename_column = dict(zip(idlabel_dict['coarse']+idlabel_dict['fine'], idlabel_presence_dict['coarse']+idlabel_presence_dict['fine']))
   relabeled_df.rename(columns=rename_column,inplace=True)
   #annotation_filtered.loc[annotation_filtered["split"]=="train", idlabel_presence_dict['full_fine']] = 0.0
   annotation_filtered[annotation_filtered["split"]=="train"].update(relabeled_df)
   annotation_filtered.reset_index(inplace=True)
   # One hot encode presence based on consensus_treshold
   annotation_filtered.loc[:,all_idlabel_presence] = (annotation_filtered[all_idlabel_presence]>consensus_threshold).astype(float)

   return annotation_filtered

class SONYCUST(Dataset):
   base_folder = 'SONYC-UST'
   resources = [
      ("https://zenodo.org/record/3873076/files/audio.tar.gz", "audio", "e709fbbeade2e45d1fbc755fda688a04"),
      ("https://zenodo.org/record/3873076/files/audio-eval-0.tar.gz", "audio-eval-0", "f92136a4a77c626a0900c18217c908bf"),
      ("https://zenodo.org/record/3873076/files/audio-eval-1.tar.gz", "audio-eval-1", "b9a727afcdb794572bf3d60b879e30a8"),
      ("https://zenodo.org/record/3873076/files/audio-eval-2.tar.gz", "audio-eval-2", "05b06fd8e274e6b8ae9202bd09fddc56"),
      ("https://zenodo.org/record/3873076/files/annotations.csv", "annotations.csv", "5292e1eed610775a4460103989db2d69"),
      ("https://zenodo.org/record/3873076/files/dcase-ust-taxonomy.yaml", "dcase-ust-taxonomy.yaml", "6c1cca1c4c383a6ebb0cb71cb74fe3a9")
   ]
   all_metadata_labels = ["sensor_id","annotator_id","borough","block",
               "latitude","longitude","year","week","day","hour"]
   
   def __init__(self, sonycust_folder, mode, metadata=None, transform=None, download=False):
      super().__init__()
      # Here we declare every useful path
      self.sonycust_folder = sonycust_folder
      self.file_path_dict = {name:os.path.join(self.sonycust_folder, name)
         for _,name,_ in self.resources}
      self.file_path_dict.update({'embedding':os.path.join(self.sonycust_folder, 'embedding')})
      self.file_path_dict.update({'melTALNet':os.path.join(self.sonycust_folder, 'melTALNet')})
      self.file_path_dict.update({'audio_PANN_32000':os.path.join(self.sonycust_folder, 'audio_PANN_32000')})

      # Downloading and extracting if needed
      if download:
         self.download()
      
      # Checking if the dataset exist at specified location
      if not os.path.exists(self.sonycust_folder):
         raise RuntimeError('Dataset not found.' +
                            ' You can use download=True to download it')
      
      if mode not in ['coarse','fine','full_fine','both']:
         raise ValueError("mode must be 'coarse','fine', 'both' or 'full_fine'")
      
      self.mode = mode
      self.transform = transform
      self.idlabel_dict, self.idlabel_presence_dict, self.idlabel_proximity_dict = self.parse_taxonomy()

      self.metadata_labels = self.all_metadata_labels if metadata == None else metadata

      self.raw_annotation_df = pd.read_csv(self.file_path_dict['annotations.csv'])
      self.annotation_df = self.raw_annotation_df.copy()

   def download(self):

      #if os.path.exists(self.sonycust_folder):
      #   return

      os.makedirs(self.sonycust_folder, exist_ok=True)
      
      # Download files
      print("Downloading files")
      for url, filename, md5 in self.resources[4:]:
         filename = url.rpartition('/')[2]
         download_url(url, root=self.sonycust_folder, filename=filename, md5=md5)
      
      for url, filename, md5 in self.resources[0:4]:
         download_and_extract_archive(url, download_root=self.sonycust_folder, filename=filename+".tar.gz", md5=md5, remove_finished=True)

      # Moving evaluation files to audio directory
      print("Moving files from eval to audio")
      for eval_num in range(3):
         for f in os.listdir(self.file_path_dict['audio-eval-'+str(eval_num)]):
            if f.endswith(".wav"):
               shutil.move(os.path.join(self.file_path_dict['audio-eval-'+str(eval_num)], f), self.file_path_dict['audio'])

   def compute_melspec(self):
      """Configuration for filterbank feature extraction: 
      The waveform is downsampled to 16 kHz; 
      frames of 1,024 samples (64 ms) are taken with a hop of 400 samples (25 ms);
      each frame is Hanning windowed and padded to 4,096 samples before taking the Fourier transform; 
      the filterbank of 64 triangle filters spans a frequency range from 0 Hz to 8 kHz.
      """

      if ~os.path.exists(self.file_path_dict['melTALNet']):
         os.makedirs(self.file_path_dict['melTALNet'], exist_ok=True)
         
      audio_list_path = [os.path.join(self.file_path_dict["audio"], x) 
                        for x in list(pd.unique(self.raw_annotation_df["audio_filename"]))]

      def compute_one_mel(filename):
         wav = librosa.load(filename, sr=44100)[0]
         melspec = librosa.feature.melspectrogram(
                     wav,
                     sr=44100,
                     n_fft=2822,
                     hop_length=1103,
                     n_mels=64,
                     fmin=0,
                     fmax=8000)
         logmel = librosa.core.power_to_db(melspec)
         np.save(os.path.join(self.file_path_dict['melTALNet'], os.path.basename(filename)[:-3] + 'npy'), logmel)
      
      _ = Parallel(n_jobs=-2)(delayed(lambda x: compute_one_mel(x))(x)
		for x in tqdm(audio_list_path))

   def parse_taxonomy(self):
      """ Opens taxononmy.yaml and return id+label list at different level
         Returns:
            full_fine_idlabel_list (list) : Full list of fine label with their id (X included)
            fine_idlabel_list (list) : List of fine label with their id (X excluded)
            coarse_idlabel_list (list) : List of coarse label with their id
      """
      # Loading taxonomy
      with open(self.file_path_dict['dcase-ust-taxonomy.yaml'], "r") as f:
         taxonomy = yaml.load(f, Loader=yaml.Loader)

      # Write labels inside lists according to the grain (fine/coarse)
      full_fine_idlabel_list = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                                 for coarse_id, fine_dict in taxonomy['fine'].items()
                                 for fine_id, fine_label in fine_dict.items()]
      fine_idlabel_list = [x for x in full_fine_idlabel_list
                              if x.split('_')[0].split('-')[1] != 'X']
      coarse_idlabel_list = ["_".join([str(k), v])
                              for k,v in taxonomy['coarse'].items()]
      
      idlabel_dict = {'full_fine':full_fine_idlabel_list,
         'fine':fine_idlabel_list,
         'coarse':coarse_idlabel_list}
      
      full_fine_idlabel_presence_list = [x+"_presence" for x in full_fine_idlabel_list]
      fine_idlabel_presence_list = [x+"_presence" for x in fine_idlabel_list]
      coarse_idlabel_presence_list = [x+"_presence" for x in coarse_idlabel_list]
      
      idlabel_presence_dict = {'full_fine':full_fine_idlabel_presence_list,
         'fine':fine_idlabel_presence_list,
         'coarse':coarse_idlabel_presence_list}
      
      full_fine_idlabel_proximity_list = [x+"_proximity" for x in full_fine_idlabel_list]
      fine_idlabel_proximity_list = [x+"_proximity" for x in fine_idlabel_list]
      coarse_idlabel_proximity_list = [x+"_proximity" for x in coarse_idlabel_list]
      
      idlabel_proximity_dict = {'full_fine':full_fine_idlabel_presence_list,
         'fine':fine_idlabel_presence_list,
         'coarse':coarse_idlabel_presence_list}

      return idlabel_dict, idlabel_presence_dict, idlabel_proximity_dict

   def train_validation_test_split(self):
      """ 
      """
      train_idx = list(self.annotation_df[self.annotation_df["split"]=="train"].index)
      val_idx = list(self.annotation_df[self.annotation_df["split"]=="validate"].index)
      test_idx = list(self.annotation_df[self.annotation_df["split"]=="test"].index)

      train_set = Subset(self, train_idx)
      val_set = Subset(self, val_idx)
      val_set.transform = None
      test_set = Subset(self, test_idx)
      test_set.transform = None

      return train_set, val_set, test_set

   def __len__(self):
      """ Gives the length of annotation.csv (no processing on it)
         Returns:
            len(self.raw_annotation_df) (int) : The length of df
      """
      return len(self.annotation_df)

   def __getitem__(self, index):
      file_name = self.annotation_df["audio_filename"].iloc[index]
      file_path = os.path.join(self.file_path_dict["audio"], file_name)

      wav = librosa.load(file_path, sr=44100)[0]

      metadata = np.array(self.annotation_df[self.metadata_labels].iloc[index])

      proximity = np.array(self.annotation_df[self.idlabel_proximity_dict[self.mode]].iloc[index])

      label = np.array(self.annotation_df[self.idlabel_presence_dict[self.mode]].iloc[index])

      return {'file_name':file_name, 'audio':wav, 'metadata':metadata, 
         'proximity':proximity,'label': label}

class SONYCUST_TALNet(SONYCUST):
   def __init__(self, sonycust_folder, mode, cleaning_strat='DCASE', consensus_threshold=0.01,relabeled_name=None,metadata=None, one_hot_time=False, transform=None, download=False):
      super().__init__(sonycust_folder, mode, metadata=metadata, transform=transform, download=download)

      if relabeled_name:
         self.file_path_dict.update({'relabeled_df':os.path.join(self.sonycust_folder, relabeled_name)})

      if cleaning_strat=='DCASE':
         self.annotation_df = cleaning_annotation_baseline(self.raw_annotation_df, self.idlabel_presence_dict,consensus_threshold=consensus_threshold)
      elif cleaning_strat=='Relabeled':
         self.annotation_df = clean_annotation_and_use_relabel(self.raw_annotation_df,self.idlabel_presence_dict,
            self.idlabel_dict, self.file_path_dict['relabeled_df'])
      elif cleaning_strat=='All_unique':
         self.annotation_df = remove_duplicates(self.raw_annotation_df, self.idlabel_presence_dict,consensus_threshold=consensus_threshold)

      #self.count_per_class = self.annotation_df[self.annotation_df['split']=='train'][self.idlabel_presence_dict[self.mode]].sum()
      #self.loss_weights = (1 - self.count_per_class/len(self.annotation_df[self.annotation_df['split']=='train'])).to_numpy()

      self.one_hot_time = one_hot_time
      if one_hot_time:
         self.to_one_hot = []
         for temp in ("week","day","hour"):
            if temp in self.metadata_labels:
               self.metadata_labels.remove(temp)
               self.to_one_hot.append(temp)

         self.one_hot_dict={"week":NUM_WEEKS,"day":NUM_DAYS,"hour":NUM_HOURS}

   def __getitem__(self, index):
      """ Gives the input vector and the associated one hot encoded label
         Args:
            index (int) : the index
         Returns:
            Dictionnary containing input embeddings, spatial_context, 
            temporal_context (one hot encoded), labels (one hot encoded)
      """

      file_name = self.annotation_df["audio_filename"].iloc[index]
      mel = np.load(os.path.join(self.file_path_dict['melTALNet'],file_name[:-3]+"npy")).transpose()

      if self.transform and self.annotation_df["split"].iloc[index]=='train':
         mel = self.transform(image=mel)['image']

      metadata = np.array(self.annotation_df[self.metadata_labels].iloc[index])

      if self.one_hot_time and self.to_one_hot != []:
         for features in self.to_one_hot:
            one_hot_enc = np.array(one_hot(self.annotation_df[features].iloc[index]-1, self.one_hot_dict[features])) 
            metadata = np.concatenate((metadata, one_hot_enc), axis=-1)

      if self.mode=='both':
         proximity ={'full_fine':np.array(self.annotation_df[self.idlabel_proximity_dict['full_fine']].iloc[index]),
            'fine':np.array(self.annotation_df[self.idlabel_proximity_dict['fine']].iloc[index]),
            'coarse':np.array(self.annotation_df[self.idlabel_proximity_dict['coarse']].iloc[index])}
         label = {'full_fine':np.array(self.annotation_df[self.idlabel_presence_dict['full_fine']].iloc[index]),
            'fine':np.array(self.annotation_df[self.idlabel_presence_dict['fine']].iloc[index]),
            'coarse':np.array(self.annotation_df[self.idlabel_presence_dict['coarse']].iloc[index])}
      else:
         proximity = np.array(self.annotation_df[self.idlabel_proximity_dict[self.mode]].iloc[index])
         label = np.array(self.annotation_df[self.idlabel_presence_dict[self.mode]].iloc[index])

      base_dict = {'file_name':file_name, "input_vector":mel, 'metadata':metadata,
         'proximity':proximity,'label': label}

      return base_dict
