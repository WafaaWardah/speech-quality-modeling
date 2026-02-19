# Contains all helpder functions (deasigned during training, perhaps can be narrowed for user's usage)

import sys
import os
import yaml
import argparse
import soundfile as sf
import torchaudio
import torch
import numpy as np
import pandas as pd
from datetime import datetime


def process_audio_file(file_path, control_sampling_rate=16000): # FOR ZERO-SHOT ONLY - 16 kHz
  
    file_ext = file_path.split('.')[-1].lower()
    waveform, sample_rate = torchaudio.load(file_path)

    # Check for 16 kHz sampling rate - FOR ZERO-SHOT ONLY
    if sample_rate != control_sampling_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=control_sampling_rate)(waveform)
        sample_rate = control_sampling_rate  

    return waveform, sample_rate



################################################################################################################

def get_args():
    parser = argparse.ArgumentParser(description="Train Speech Quality Prediction Model")
    parser.add_argument('--yaml', required=True, type=str, help="YAML file with training configurations")
    return parser.parse_args()

def load_config(yaml_file):
    if not os.path.isfile(yaml_file):
        print(f"Error: {yaml_file} does not exist.")
        sys.exit(1)

    with open(yaml_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)

    return config

def create_output_directory(runname, base_output_directory):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_directory, f"{runname}_{current_time}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created new output directory: {output_dir}")
    else:
        print(f"Output directory {output_dir} already exists.")
    return output_dir


def load_train_val_df(dev_labels_path, tr_list, val_list):

    df_file = pd.read_csv(dev_labels_path, low_memory=False)

    if not set(tr_list + val_list).issubset(df_file.db.unique().tolist()):
        missing_dbs = set(tr_list + val_list).difference(df_file.db.unique().tolist())
        raise ValueError(f"csv file is missing some dbs {missing_dbs}")
        sys.exit(1)
    
    df_train = df_file[df_file.db.isin(tr_list)].reset_index()
    df_val = df_file[df_file.db.isin(val_list)].reset_index()
    print(f"Training size: {len(df_train)}, Validation size: {len(df_val)}")
    return df_train, df_val


def model_size_in_MB(model):
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, trainable_params, (num_params * 4 / (1024 ** 2))  # Convert bytes to MB


def masked_mse_loss(preds, targets):
    # Compute the element-wise error
    err = (preds - targets).view(-1)
    
    # Mask out NaN values
    idx_not_nan = ~torch.isnan(err)
    
    # Filter out NaNs and compute the mean squared error on valid values
    nan_err = err[idx_not_nan]
    return torch.mean(nan_err ** 2) if nan_err.numel() > 0 else torch.tensor(0.0, requires_grad=True, device=preds.device)

def make_ep_met(tr_avg_metrics, val_avg_metrics):
    # Validation PCC averages for each dimension
    met = {'val_pcc_mos': round(val_avg_metrics['mos_pcc'], 3), 
        'val_pcc_noi': round(val_avg_metrics['noi_pcc'], 3),
        'val_pcc_dis': round(val_avg_metrics['dis_pcc'], 3), 
        'val_pcc_col': round(val_avg_metrics['col_pcc'], 3),
        'val_pcc_loud': round(val_avg_metrics['loud_pcc'], 3),

        # Validation RMSE averages for each dimension
        'val_rmse_mos': round(val_avg_metrics['mos_rmse'], 3), 
        'val_rmse_noi': round(val_avg_metrics['noi_rmse'], 3),
        'val_rmse_dis': round(val_avg_metrics['dis_rmse'], 3), 
        'val_rmse_col': round(val_avg_metrics['col_rmse'], 3),
        'val_rmse_loud': round(val_avg_metrics['loud_rmse'], 3),

        # Validation RMSE* averages
        'val_rmse_star_mos': round(val_avg_metrics['mos_rmse_star'], 3),
        'val_rmse_star_noi': round(val_avg_metrics['noi_rmse_star'], 3),
        'val_rmse_star_dis': round(val_avg_metrics['dis_rmse_star'], 3),
        'val_rmse_star_col': round(val_avg_metrics['col_rmse_star'], 3),
        'val_rmse_star_loud': round(val_avg_metrics['loud_rmse_star'], 3),

        # Validation First-Order Mapped RMSE averages
        'val_rmse_mapped_first_order_mos': round(val_avg_metrics['mos_rmse_mapped_first_order'], 3),
        'val_rmse_mapped_first_order_noi': round(val_avg_metrics['noi_rmse_mapped_first_order'], 3),
        'val_rmse_mapped_first_order_dis': round(val_avg_metrics['dis_rmse_mapped_first_order'], 3),
        'val_rmse_mapped_first_order_col': round(val_avg_metrics['col_rmse_mapped_first_order'], 3),
        'val_rmse_mapped_first_order_loud': round(val_avg_metrics['loud_rmse_mapped_first_order'], 3),

        # Validation First-Order Mapped RMSE* averages
        'val_rmse_mapped_first_order_star_mos': round(val_avg_metrics['mos_rmse_mapped_first_order_star'], 3),
        'val_rmse_mapped_first_order_star_noi': round(val_avg_metrics['noi_rmse_mapped_first_order_star'], 3),
        'val_rmse_mapped_first_order_star_dis': round(val_avg_metrics['dis_rmse_mapped_first_order_star'], 3),
        'val_rmse_mapped_first_order_star_col': round(val_avg_metrics['col_rmse_mapped_first_order_star'], 3),
        'val_rmse_mapped_first_order_star_loud': round(val_avg_metrics['loud_rmse_mapped_first_order_star'], 3),

        # Validation Third-Order Mapped RMSE averages
        'val_rmse_mapped_third_order_mos': round(val_avg_metrics['mos_rmse_mapped_third_order'], 3),
        'val_rmse_mapped_third_order_noi': round(val_avg_metrics['noi_rmse_mapped_third_order'], 3),
        'val_rmse_mapped_third_order_dis': round(val_avg_metrics['dis_rmse_mapped_third_order'], 3),
        'val_rmse_mapped_third_order_col': round(val_avg_metrics['col_rmse_mapped_third_order'], 3),
        'val_rmse_mapped_third_order_loud': round(val_avg_metrics['loud_rmse_mapped_third_order'], 3),

        # Validation Third-Order Mapped RMSE* averages
        'val_rmse_mapped_third_order_star_mos': round(val_avg_metrics['mos_rmse_mapped_third_order_star'], 3),
        'val_rmse_mapped_third_order_star_noi': round(val_avg_metrics['noi_rmse_mapped_third_order_star'], 3),
        'val_rmse_mapped_third_order_star_dis': round(val_avg_metrics['dis_rmse_mapped_third_order_star'], 3),
        'val_rmse_mapped_third_order_star_col': round(val_avg_metrics['col_rmse_mapped_third_order_star'], 3),
        'val_rmse_mapped_third_order_star_loud': round(val_avg_metrics['loud_rmse_mapped_third_order_star'], 3),

        # Training metrics, similar structure as validation
        'tr_pcc_mos': round(tr_avg_metrics['mos_pcc'], 3), 
        'tr_pcc_noi': round(tr_avg_metrics['noi_pcc'], 3),
        'tr_pcc_dis': round(tr_avg_metrics['dis_pcc'], 3), 
        'tr_pcc_col': round(tr_avg_metrics['col_pcc'], 3),
        'tr_pcc_loud': round(tr_avg_metrics['loud_pcc'], 3),

        'tr_rmse_mos': round(tr_avg_metrics['mos_rmse'], 3), 
        'tr_rmse_noi': round(tr_avg_metrics['noi_rmse'], 3),
        'tr_rmse_dis': round(tr_avg_metrics['dis_rmse'], 3), 
        'tr_rmse_col': round(tr_avg_metrics['col_rmse'], 3),
        'tr_rmse_loud': round(tr_avg_metrics['loud_rmse'], 3),

        'tr_rmse_star_mos': round(tr_avg_metrics['mos_rmse_star'], 3),
        'tr_rmse_star_noi': round(tr_avg_metrics['noi_rmse_star'], 3),
        'tr_rmse_star_dis': round(tr_avg_metrics['dis_rmse_star'], 3),
        'tr_rmse_star_col': round(tr_avg_metrics['col_rmse_star'], 3),
        'tr_rmse_star_loud': round(tr_avg_metrics['loud_rmse_star'], 3),

        'tr_rmse_mapped_first_order_mos': round(tr_avg_metrics['mos_rmse_mapped_first_order'], 3),
        'tr_rmse_mapped_first_order_noi': round(tr_avg_metrics['noi_rmse_mapped_first_order'], 3),
        'tr_rmse_mapped_first_order_dis': round(tr_avg_metrics['dis_rmse_mapped_first_order'], 3),
        'tr_rmse_mapped_first_order_col': round(tr_avg_metrics['col_rmse_mapped_first_order'], 3),
        'tr_rmse_mapped_first_order_loud': round(tr_avg_metrics['loud_rmse_mapped_first_order'], 3),

        'tr_rmse_mapped_first_order_star_mos': round(tr_avg_metrics['mos_rmse_mapped_first_order_star'], 3),
        'tr_rmse_mapped_first_order_star_noi': round(tr_avg_metrics['noi_rmse_mapped_first_order_star'], 3),
        'tr_rmse_mapped_first_order_star_dis': round(tr_avg_metrics['dis_rmse_mapped_first_order_star'], 3),
        'tr_rmse_mapped_first_order_star_col': round(tr_avg_metrics['col_rmse_mapped_first_order_star'], 3),
        'tr_rmse_mapped_first_order_star_loud': round(tr_avg_metrics['loud_rmse_mapped_first_order_star'], 3),

        'tr_rmse_mapped_third_order_mos': round(tr_avg_metrics['mos_rmse_mapped_third_order'], 3),
        'tr_rmse_mapped_third_order_noi': round(tr_avg_metrics['noi_rmse_mapped_third_order'], 3),
        'tr_rmse_mapped_third_order_dis': round(tr_avg_metrics['dis_rmse_mapped_third_order'], 3),
        'tr_rmse_mapped_third_order_col': round(tr_avg_metrics['col_rmse_mapped_third_order'], 3),
        'tr_rmse_mapped_third_order_loud': round(tr_avg_metrics['loud_rmse_mapped_third_order'], 3),

        'tr_rmse_mapped_third_order_star_mos': round(tr_avg_metrics['mos_rmse_mapped_third_order_star'], 3),
        'tr_rmse_mapped_third_order_star_noi': round(tr_avg_metrics['noi_rmse_mapped_third_order_star'], 3),
        'tr_rmse_mapped_third_order_star_col': round(tr_avg_metrics['col_rmse_mapped_third_order_star'], 3),
        'tr_rmse_mapped_third_order_star_loud': round(tr_avg_metrics['loud_rmse_mapped_third_order_star'], 3)
    }
    return met


