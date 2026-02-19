import os
import numpy as np
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor, Wav2Vec2FeatureExtractor, WhisperFeatureExtractor
import utils
import torch

######################################## Dataset class for checking through wav file accessibilty###############
class AST_SpeechQualityDataset_Check(Dataset):
    def __init__(self, df, data_dir, dim):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # wav files -------------------------------------
        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])
        waveform, sample_rate = utils.process_audio_file(file_name)

        return index, db, file_name
    
########################################
class AST_Dataset(Dataset):
    def __init__(self, df, data_dir, dim):
        self.df = df
        self.data_dir = data_dir
        self.dim = dim + '_file'

        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 16000  # Set to 16 kHz for zero-shot
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # wav files -------------------------------------
        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        self.feature_extractor.mean = self.df['db_mean'].iloc[index]  #-4.2677393      # Replace if recalculated mean for 48 kHz data
        self.feature_extractor.std = self.df['db_std'].iloc[index]  #4.5689974        # Replace if recalculated std for 48 kHz data

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()
        
        # target y value 
        y = self.df[self.dim].iloc[index].astype('float32')
        y = (y - 1) / 4

        return index, features, y
    
########################################
class AST_Test(Dataset):
    def __init__(self, df, data_dir, dim):
        self.df = df
        self.data_dir = data_dir
        self.dim = dim + '_file'

        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 16000  # Set to 16 kHz for zero-shot
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # wav files -------------------------------------
        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index]) # from the Interspeech 2022 labels csv

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        self.feature_extractor.mean = -4.2677393  #self.df['db_mean'].iloc[index]  #-4.2677393      # Replace if recalculated mean for 48 kHz data
        self.feature_extractor.std = 4.5689974 #self.df['db_std'].iloc[index]  #4.5689974        # Replace if recalculated std for 48 kHz data

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()
        
        # target y value 
        y = self.df[self.dim].iloc[index].astype('float32')
        y = (y - 1) / 4

        return index, features, y
    
########################################
class W2V2_Dataset(Dataset):
    def __init__(self, df, data_dir, dim):
        self.df = df
        self.data_dir = data_dir
        self.dim = dim + "_file"

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self.feature_extractor.sampling_rate = 16000   # IMPORTANT
        self.feature_extractor.return_attention_mask = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = os.path.join(self.data_dir, self.df["file_path"].iloc[index])

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        # Ideally: ensure process_audio_file returns 16k.
        # If not, you must resample either in process_audio_file or here.

        features = waveform.squeeze(0).detach().float()

        y = self.df[self.dim].iloc[index].astype("float32")
        y = (y - 1) / 4

        return index, features, y
    
    
########################################
import os
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

class Whisper_Dataset(Dataset):
    def __init__(self, df, data_dir, dim):
        self.df = df
        self.data_dir = data_dir
        self.dim = dim + "_file"

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
        self.feature_extractor.sampling_rate = 16000  # consistent with your pipeline

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        file_name = os.path.join(self.data_dir, self.df["file_path"].iloc[index])

        waveform, sr = utils.process_audio_file(file_name)  # your util enforces 16k
        waveform = waveform.squeeze(0).detach().float()     # [T]

        feats = self.feature_extractor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt"
        )

        # Whisper expects input_features, but we expose it as "input_values" for your unified loop.
        input_values = feats["input_features"].squeeze(0)  # [80, T']

        # attention_mask may or may not exist depending on transformers version; safe to ignore if missing
        attention_mask = feats.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)

        y = self.df[self.dim].iloc[index].astype("float32")
        y = (y - 1) / 4

        batch_features = {"input_values": input_values}
        if attention_mask is not None:
            batch_features["attention_mask"] = attention_mask

        return index, batch_features, y
    
class ASTVal(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir
        
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 16000  # Set to 16 kHz for 16 kHz zero-shot
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        db = self.df['db'].iloc[index]

        # wav files -------------------------------------
        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])

        waveform, sample_rate = utils.process_audio_file(file_name)
        waveform = waveform.squeeze()

        self.feature_extractor.mean = self.df['db_mean'].iloc[index]  #-4.2677393      # Replace if recalculated mean for 48 kHz data
        self.feature_extractor.std = self.df['db_std'].iloc[index]  #4.5689974        # Replace if recalculated std for 48 kHz data

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()

        return index, features

#########################################

def w2v2_collate_fn(batch, feature_extractor: Wav2Vec2FeatureExtractor):
    indices = torch.tensor([b[0] for b in batch], dtype=torch.long)
    wavs = [b[1].cpu().numpy() for b in batch]  # list of 1D arrays
    ys = torch.tensor([b[2] for b in batch], dtype=torch.float32)

    padded = feature_extractor(
        wavs,
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    # padded["input_values"] -> [B, T]
    # padded["attention_mask"] -> [B, T]
    return indices, padded, ys