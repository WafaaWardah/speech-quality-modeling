# evaluation script for the zero-shot models
# On the HPC, I named this run_train.py to save time redoing the entrypoint command
import yaml
import sys
import os
import time
import pandas as pd
import numpy as np
import torch
from datasets import AST_Dataset, W2V2_Dataset, Whisper_Dataset, w2v2_collate_fn
from models import AST, W2V2, HuBERT, WavLM, Whisper
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import metrics
import utils
import multiprocessing as mp, threading, sys, gc, logging
import warnings

class W2V2Collator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        return w2v2_collate_fn(batch, self.feature_extractor)


torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings(
    "ignore",
    message="Passing `gradient_checkpointing` to a config initialization is deprecated"
)

warnings.filterwarnings(
    "ignore",
    message=r"Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = utils.get_args()
    config = utils.load_config(args.yaml)
    print(f"\nPredicting with {config['training']['runname']} ...")
    print(f"Using device: {device}")
    output_dir = config['training']['output_dir']
    run_output_dir = utils.create_output_directory(config['training']['runname'], output_dir)
    epoch_log_df = pd.DataFrame()

    # Dataset and Dataloader -----------------------------------------------------------------------

    df_tr, df_val = utils.load_train_val_df(config['training']['dev_labels_path'], 
                        config['training']['tr_db_list'],
                        config['training']['val_db_list'])
    
    def load_head_from_dp_checkpoint(w_path, model, device):
        ckpt = torch.load(w_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        # Remove DataParallel prefix
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # Keep only keys that exist in the current model
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state.items() if k in model_keys}

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        model.to(device).eval()

        # Sanity check
        loaded_fc = [k for k in filtered if k.startswith("fc.")]
        print(f"Loaded fc keys: {loaded_fc}")

        if not loaded_fc:
            raise RuntimeError("FC head was not loaded — key mismatch remains.")

        return model


    if 'w2v2' in config['training']['runname']:
        model = W2V2()
        ds_tr = W2V2_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = W2V2_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
        w_path = '/Users/wafaa/Code/psamd-master/zeroshot/hpc_trained/zs-w2v2_20260101_213833/zs-w2v2_20260101_213833mos_.tar'
        load_head_from_dp_checkpoint(w_path, model, device)

    elif 'HuBERT' in config['training']['runname']:
        model = HuBERT()
        ds_tr = W2V2_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = W2V2_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
        w_path = '/Users/wafaa/Code/psamd-master/zeroshot/hpc_trained/zs-HuBERT_20260101_210853/zs-HuBERT_20260101_210853mos_.tar'
        load_head_from_dp_checkpoint(w_path, model, device)

    elif 'WavLM' in config['training']['runname']:
        model = WavLM()
        ds_tr = W2V2_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = W2V2_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
        w_path = '/Users/wafaa/Code/psamd-master/zeroshot/hpc_trained/zs-WavLM_20260101_210904/zs-WavLM_20260101_210904mos_.tar'
        load_head_from_dp_checkpoint(w_path, model, device)

    elif 'Whisper' in config['training']['runname']:
        model = Whisper()
        ds_tr = Whisper_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = Whisper_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
        w_path = '/Users/wafaa/Code/psamd-master/zeroshot/hpc_trained/zs-Whisper_20260101_210917/zs-Whisper_20260101_210917mos_.tar'
        load_head_from_dp_checkpoint(w_path, model, device)

    else:
        model = AST()
        ds_tr = AST_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = AST_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
        w_path = '/Users/wafaa/Code/psamd-master/zeroshot/hpc_trained/zs-AST_20260101_210917/zs-AST_20260101_210917mos_.tar'
        load_head_from_dp_checkpoint(w_path, model, device)
    
    def get_collate_fn(runname: str, ds_tr):
        name = runname.lower()
        if any(k in name for k in ["w2v2", "hubert", "wavlm"]):
            return W2V2Collator(ds_tr.feature_extractor)
        return None

    collate_fn = get_collate_fn(config['training']['runname'], ds_tr)

    dl_tr = DataLoader(
        dataset=ds_tr,
        batch_size=config['training']['bs'],
        shuffle=config['training']['train_shuffle'],
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=config['training']['bs'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    model = nn.DataParallel(model)

    num_params, trainable_params, model_size = utils.model_size_in_MB(model)
    print(f"\nModel: {model.module.__class__.__name__}")
    print(f"Number of parameters: {num_params:,}")
    print(f"Model size: approximately {model_size:.2f} MB")
    print(f"Number of trainable parameters: {trainable_params:,}")

    
    model.to(device)

    best_global_pcc = float('-inf')

    dim_pred = config['dimension'] + "_pred"

    model.eval()
    y_hat_val = torch.full((len(ds_val),), -0.25, device='cpu') # Stores the validation outputs, later filled into ds_val df

    with torch.no_grad():  # Disable gradient tracking for validation
        for b, (index, batch_features, batch_labels) in enumerate(
            tqdm(dl_val, desc="Evaluating", leave=False)
            ):

            #batch_features, batch_labels = batch_features.float().to(device), batch_labels.to(device)
            # Move features to device (works for Tensor or dict/BatchFeature)
            if isinstance(batch_features, dict) or hasattr(batch_features, "keys"):
                batch_features = {k: v.to(device) for k, v in batch_features.items()}

                if "attention_mask" in batch_features and batch_features["attention_mask"] is not None:
                    batch_features["attention_mask"] = batch_features["attention_mask"].to(torch.bool)
            else:
                batch_features = batch_features.float().to(device)

            batch_labels = batch_labels.to(device)

            # Forward pass ---------------------------------------
            #pred = model(batch_features)
            if isinstance(batch_features, dict):
                pred = model(batch_features["input_values"], attention_mask=batch_features.get("attention_mask", None))
            else:
                pred = model(batch_features)
                            
            # Stack predictions for each dimension
            y_hat_batch = pred.to('cpu') # On CPU
            y_hat_val[index] = y_hat_batch # On CPU


    # Scale predictions once all batches are processed
    y_hat_val_descaled = y_hat_val * 4 + 1 # On CPU
    y_hat_val_descaled = y_hat_val_descaled.detach().numpy() # On CPU

    # Convert predictions into DataFrame columns on CPU
    ds_val.df[dim_pred] = y_hat_val_descaled

    filtered_val_df = ds_val.df.loc[ds_val.df[dim_pred] != 0.0]

    val_per_file_metrics_df = metrics.calc_metrics_db(filtered_val_df, dim=config['dimension'])

    filtered_val_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + f"_{config['training']['runname']}_preds_per_file.csv"), index=False)
    val_per_file_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + f"_{config['training']['runname']}_metrics_per_file.csv"), index=False)

        
    with open(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_config_dump.yaml'), 'w') as yaml_file:
              yaml.dump(config, yaml_file, default_flow_style=False)

    print('\nPred/Eval complete.')

