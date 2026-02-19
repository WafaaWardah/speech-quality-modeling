import os
import numpy as np
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor
import utils
import torch


class AST_SpeechQualityDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 48000  # Set to 48 kHz for your 48k audio
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # AST features (48 kHz) preprocessed normalized tensor files  -------------------------------------
        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        self.feature_extractor.mean = self.df['db_mean'].iloc[index]  #-4.2677393      
        self.feature_extractor.std = self.df['db_std'].iloc[index]  #4.5689974        

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()
        
        
        # target y values ----------------------------------------------------------------------------
        y_mos = self.df['mos_file'].iloc[index].reshape(-1).astype('float32')
        y_noi = self.df['noi_file'].iloc[index].reshape(-1).astype('float32')
        y_dis = self.df['dis_file'].iloc[index].reshape(-1).astype('float32')
        y_col = self.df['col_file'].iloc[index].reshape(-1).astype('float32')
        y_loud = self.df['loud_file'].iloc[index].reshape(-1).astype('float32')
        y = np.concatenate(((y_mos, y_noi, y_dis, y_col, y_loud)), axis=0)
        y = (y - 1) / 4

        return index, features, y
    
class SQA_Test(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 48000  # Set to 48 kHz for your 48k audio
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # AST features (48 kHz) preprocessed normalized tensor files  -------------------------------------
        file_name = os.path.join(self.data_dir, db, self.df['file_path'].iloc[index])

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        self.feature_extractor.mean = -10.25446422 # self.df['db_mean'].iloc[index]  #-4.2677393      
        self.feature_extractor.std = 4.205750774 # self.df['db_std'].iloc[index]  #4.5689974        

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()

        return index, features
    