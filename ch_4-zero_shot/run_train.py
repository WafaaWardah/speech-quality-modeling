# Script for zero-shot evaluation.
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
    print(f"\nTraining {config['training']['runname']} begins...")
    print(f"Using device: {device}")
    output_dir = config['training']['output_dir']
    run_output_dir = utils.create_output_directory(config['training']['runname'], output_dir)
    epoch_log_df = pd.DataFrame()

    # Dataset and Dataloader -----------------------------------------------------------------------

    df_tr, df_val = utils.load_train_val_df(config['training']['dev_labels_path'], 
                        config['training']['tr_db_list'],
                        config['training']['val_db_list'])

    if 'w2v2' in config['training']['runname']:
        model = W2V2()
        ds_tr = W2V2_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = W2V2_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
    elif 'HuBERT' in config['training']['runname']:
        model = HuBERT()
        ds_tr = W2V2_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = W2V2_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
    elif 'WavLM' in config['training']['runname']:
        model = WavLM()
        ds_tr = W2V2_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = W2V2_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
    elif 'Whisper' in config['training']['runname']:
        model = Whisper()
        ds_tr = Whisper_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = Whisper_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
    else:
        model = AST()
        ds_tr = AST_Dataset(df_tr, config['training']['data_dir'], dim=config['dimension'].lower())
        ds_val = AST_Dataset(df_val, config['training']['data_dir'], dim=config['dimension'].lower())
    
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

    # -----------------------------------------------------------------------------------------------
    #opt = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-5)
    opt = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["training"]["lr"],
        weight_decay=1e-5
    )

    # Load pre-trained weights ----------------------------------------------------------------------
    if config['model']['pretrained_path']:
        #checkpoint = torch.load(config['model']['pretrained_path'], weights_only=False)
        #model.load_state_dict(checkpoint['model_state_dict'])
        #opt.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loading weights from {config['model']['pretrained_path']}...")
        checkpoint = torch.load(config['model']['pretrained_path'])

        # Prefix all keys with "module." for DataParallel compatibility
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_state_dict[f"module.{k}"] = v

        try:
            model.load_state_dict(new_state_dict)
            print("Model weights loaded successfully.")
        except RuntimeError as e:
            print("Error loading model weights:")
            print(e)
    
    model.to(device)

    # Initialize scheduler ----------------------------------------------------------------------
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5,
    #            patience=config['training']['tr_lr_patience'], threshold=1e-4, min_lr=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, "min",
        factor=0.5,
        patience=config["training"]["tr_lr_patience"],
        threshold=1e-4,
        min_lr=1e-6
    )

    es_patience = config['training']['es_patience']
    best_loss = float('inf')
    best_global_pcc = float('-inf')

    # Train epochs ----------------------------------------------------------------------------------
    for epoch in tqdm(range(1, config['training']['max_epochs'] + 1)):
        print(f'\nEpoch {epoch} Training')

        epoch_tic = time.time()
        batch_count = 0
        epoch_train_loss = 0.0
        y_hat_train = torch.full((len(ds_tr),), -0.25, device='cpu') # On CPU. Stores the training outputs, later filled into ds_train df

        model.train()

        # Train batches -----------------------------------------------------------------------------
        for b, (index, batch_features, batch_labels) in enumerate(dl_tr):
            torch.autograd.set_detect_anomaly(True)

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
            
            y_hat_batch = pred.to('cpu') # On CPU
            y_hat_train[index] = y_hat_batch # On CPU

            # Loss and backprop ------------------------------------
            mask = ~torch.isnan(batch_labels)
            masked_pred = pred[mask].float()
            masked_target = batch_labels[mask].float()

            if masked_pred.numel() > 0:
                loss = F.mse_loss(masked_pred, masked_target)
                loss.backward()
                opt.step()
                opt.zero_grad()
                # Update loss tracking
                epoch_train_loss += loss.item()
                batch_count += 1
            else:
                loss = None
                print(f"Skipped tr batch: {b+1} because loss=None")
   
            

            print(f"Epoch {epoch}, Batch {b+1}, {config['dimension']} loss = {loss:.5f}")

            if batch_count == config['dev_flag']: break # Remove after setting up successfully

        epoch_train_loss = epoch_train_loss/batch_count
        print(f"global train loss = {round(epoch_train_loss, 3)}\n")

        # Scale predictions once all batches are processed
        y_hat_train_descaled = y_hat_train * 4 + 1 # On CPU
        y_hat_train_descaled = y_hat_train_descaled.detach().numpy() # On CPU

        dim_pred = config['dimension'] + "_pred"
        ds_tr.df[dim_pred] = y_hat_train_descaled

        filtered_train_df = ds_tr.df.loc[ds_tr.df[dim_pred] != 0.0]
        tr_per_file_metrics_df = metrics.calc_metrics_db(filtered_train_df, dim=config['dimension'])

        # Validate batches -----------------------------------------------------------------------------
        model.eval()

        print(f'\nEpoch {epoch} Validation')

        # Reset helper variables --------------------------------------------------------------------
        batch_count = 0 # Reset batch_count to 0 at start of epoch validation
        epoch_val_loss = 0.0 # Reset epoch validation loss to 0 at start of epoch
        y_hat_val = torch.full((len(ds_val),), -0.25, device='cpu') # Stores the validation outputs, later filled into ds_val df

        with torch.no_grad():  # Disable gradient tracking for validation
            for b, (index, batch_features, batch_labels) in enumerate(dl_val):

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

                # Loss ------------------------------------
                # MOS  loss 
                mask = ~torch.isnan(batch_labels)
                masked_pred = pred[mask]
                masked_target = batch_labels[mask]

                if masked_pred.numel() > 0:
                    loss = F.mse_loss(masked_pred, masked_target)
                    epoch_val_loss += loss.item()
                    batch_count += 1
                else:
                    loss = None

                print(f"Epoch {epoch}, Batch {b+1}, {config['dimension']} loss = {loss:.5f}")

                if batch_count == config['dev_flag']: break 

        epoch_val_loss = epoch_val_loss/batch_count
        
        # Scale predictions once all batches are processed
        y_hat_val_descaled = y_hat_val * 4 + 1 # On CPU
        y_hat_val_descaled = y_hat_val_descaled.detach().numpy() # On CPU

        # Convert predictions into DataFrame columns on CPU
        ds_val.df[dim_pred] = y_hat_val_descaled

        filtered_val_df = ds_val.df.loc[ds_val.df[dim_pred] != 0.0]

        global_pcc = metrics.calculate_pcc(y_true=filtered_val_df[f"{config['dimension']}_file"], y_pred=filtered_val_df[dim_pred])

        print(f"\nglobal_pcc = {round(global_pcc, 3)} \tbest_global_pcc = {round(best_global_pcc, 3)}")
        print(f"val loss = {round(epoch_val_loss, 3)} \tbest_loss = {round(best_loss, 3)}\n")

        val_per_file_metrics_df = metrics.calc_metrics_db(filtered_val_df, dim=config['dimension'])

        # Early stopping check -----------------------------------------------------------------
        if epoch_val_loss <= best_loss:
            best_loss = epoch_val_loss # Update best loss
            es_patience = config['training']['es_patience'] # Reset early stopping patience
            best_global_pcc = global_pcc # The best is referring to best epoch based on lowest validation loss and not pcc

            # Save model predictions to file - per file and per condition - for training set and validation set
            filtered_train_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_train_preds_per_file.csv'), index=False)
            filtered_val_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_val_preds_per_file.csv'), index=False)

            # Save model performance metrics per DB to file - per file and per condition - for training set and validation set
            tr_per_file_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_train_metrics_per_file.csv'), index=False)
            #tr_per_con_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_train_metrics_per_con.csv'), index=False)

            val_per_file_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_val_metrics_per_file.csv'), index=False)
            #val_per_con_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_val_metrics_per_con.csv'), index=False)

            # Save trained model -------------------------------------------------------------------
            model_checkpoint_name = os.path.basename(run_output_dir) + config['dimension'] + '_.tar'
            model_checkpoint_path = os.path.join(run_output_dir, model_checkpoint_name)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_val_loss
                }, model_checkpoint_path)
        else:
            es_patience -= 1

        scheduler.step(epoch_val_loss)  # Adjust LR for the MOS model

        if device == "cuda": torch.cuda.empty_cache() # To prevent OOM error

        # Epoch logging  -----------------------------------------------------------------------------

        epoch_dict = {
            'epoch': epoch,
            'val_loss': round(epoch_val_loss, 6),
            'tr_loss': round(epoch_train_loss, 6),
            'es_patience': str(es_patience) + '/' + str(config['training']['es_patience']), 
            'duration': round(time.time() - epoch_tic, 4),
            'learning_rate': opt.param_groups[0]['lr']
            }
        
        row_dict = pd.DataFrame([epoch_dict])
        epoch_log_df = pd.concat([epoch_log_df, row_dict], ignore_index=True)
        epoch_log_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_log.csv'), index=False)

        if es_patience == 0:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    config['best_epoch'] =  epoch - config['training']['es_patience'] + 1 # likely not correct, fix later
    with open(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_config_dump.yaml'), 'w') as yaml_file:
              yaml.dump(config, yaml_file, default_flow_style=False)

    print('\nTraining complete.')

        # --- Graceful shutdown ---
    # 1) Close logging and flush stdio
    logging.shutdown()
    sys.stdout.flush(); sys.stderr.flush()

    # 2) Ensure DataLoader workers are gone
    kids = mp.active_children()
    for p in kids:
        p.terminate()
    for p in kids:
        p.join(timeout=1)

    # 3) Join any non-main Python threads briefly
    for t in [t for t in threading.enumerate() if t.name != "MainThread"]:
        t.join(timeout=1)

    # 4) GC to release file descriptors
    gc.collect()

    # 5) Last-resort hard exit if something still blocks (Mac-safe)
    os._exit(0)
