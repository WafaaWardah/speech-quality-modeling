# training script for P.SAMD, 27.10.2024, 27.11.2024, 11.12.2024
import yaml
import os, sys
import time
import pandas as pd
import numpy as np
import torch
from datasets import AST_SpeechQualityDataset
from models import AST_pretrained
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import metrics
import utils
from tabulate import tabulate


torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = utils.get_args()
    config = utils.load_config(args.yaml)
    print("\nTraining begins...")
    print(f"Using device: {device}")
    output_dir = config['training']['output_dir']
    run_output_dir = utils.create_output_directory(config['training']['runname'], output_dir)
    epoch_log_df = pd.DataFrame()

    # Dataset and Dataloader -----------------------------------------------------------------------
    df_tr, df_val = utils.load_train_val_df(config['training']['dev_labels_path'], 
                        config['training']['tr_db_list'],
                        config['training']['val_db_list'])
    
    ds_tr = AST_SpeechQualityDataset(df_tr, config['training']['data_dir'])
    ds_val = AST_SpeechQualityDataset(df_val, config['training']['data_dir'])

    dl_tr = DataLoader(
        dataset=ds_tr,
        batch_size=config['training']['bs'],
        shuffle=config['training']['train_shuffle'],
        num_workers=config['training']['num_workers'],
        drop_last=True
    )

    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=config['training']['bs'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        drop_last=True
    )

    # Initialize model and optimizer ----------------------------------------------------------------
    model = AST_pretrained()
    model = nn.DataParallel(model)

    # Estimate model size
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)

    def model_size_in_MB(model):
        num_params = count_parameters(model)
        return num_params * 4 / (1024 ** 2)  # Convert bytes to MB
    
    num_params, model_size = utils.model_size_in_MB(model)
    print(f"Model has {num_params:,} parameters and model size is approximately {model_size:.2f} MB")
    
    # -----------------------------------------------------------------------------------------------
    opt = optim.RAdam(model.parameters(), lr=config['training']['lr'], weight_decay=1e-5)

    if config['model']['pretrained_path']:
        checkpoint = torch.load(config['model']['pretrained_path'], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    model.to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                'min',
                factor=0.5,
                patience=config['training']['tr_lr_patience'],
                threshold=1e-4,
                min_lr=1e-6)
    
    es_patience = config['training']['es_patience']
    best_loss = 100.0

    # Train epochs ----------------------------------------------------------------------------------
    for epoch in tqdm(range(1, config['training']['max_epochs'] + 1)):
        print(f'\nEpoch {epoch} Training')

        epoch_tic = time.time()
        batch_count = 0
        epoch_train_loss = 0.0
        y_hat_train = torch.full((len(ds_tr), 5), -0.25, device='cpu') # On CPU. Stores the training outputs, later filled into ds_train df

        model.train()
        # Train batches -----------------------------------------------------------------------------
        for b, (index, batch_features, batch_labels) in enumerate(dl_tr):
            torch.autograd.set_detect_anomaly(True)

            batch_features, batch_labels = batch_features.float().to(device), batch_labels.to(device)

            # Forward pass ---------------------------------------
            mos_pred, noi_pred, dis_pred, col_pred, loud_pred = model(batch_features)
            
            # Stack predictions for each dimension
            y_hat_batch = torch.stack([mos_pred, noi_pred, dis_pred, col_pred, loud_pred], dim=1).squeeze().to('cpu') # On CPU
            y_hat_train[index, :] = y_hat_batch # On CPU

            # Zero gradients before accumulation
            opt.zero_grad()

            # MOS dimension loss and backpropagation
            mask_mos = ~torch.isnan(batch_labels[:, 0])
            masked_mos_pred = mos_pred[mask_mos]
            masked_mos_target = batch_labels[:, 0][mask_mos]
            if masked_mos_pred.numel() > 0:
                loss_mos = F.mse_loss(masked_mos_pred, masked_mos_target)
                # Set requires_grad only for the AST backbone and mos_fc layer
                for name, param in model.named_parameters():
                    # Enable gradients for AST backbone and mos_fc only
                    is_mos_fc_param = any(param is fc_param for fc_param in model.module.mos_fc.parameters())
                    if "model" in name or is_mos_fc_param:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False  # Disable gradients for other FC layers
                loss_mos.backward(retain_graph=True)
            else:
                loss_mos = None

            # NOI dimension loss and backpropagation
            mask_noi = ~torch.isnan(batch_labels[:, 1])
            masked_noi_pred = noi_pred[mask_noi]
            masked_noi_target = batch_labels[:, 1][mask_noi]
            if masked_noi_pred.numel() > 0:
                loss_noi = F.mse_loss(masked_noi_pred, masked_noi_target)
                # Set requires_grad only for the AST backbone and noi_fc layer
                for name, param in model.named_parameters():
                    # Enable gradients for AST backbone and noi_fc only
                    is_noi_fc_param = any(param is fc_param for fc_param in model.module.noi_fc.parameters())
                    if "model" in name or is_noi_fc_param:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False  # Disable gradients for other FC layers
                loss_noi.backward(retain_graph=True)
            else:
                loss_noi = None

            # DIS dimension loss and backpropagation
            mask_dis = ~torch.isnan(batch_labels[:, 2])
            masked_dis_pred = dis_pred[mask_dis]
            masked_dis_target = batch_labels[:, 2][mask_dis]
            if masked_dis_pred.numel() > 0:
                loss_dis = F.mse_loss(masked_dis_pred, masked_dis_target)
                # Set requires_grad only for the AST backbone and dis_fc layer
                for name, param in model.named_parameters():
                    # Enable gradients for AST backbone and dis_fc only
                    is_dis_fc_param = any(param is fc_param for fc_param in model.module.dis_fc.parameters())
                    if "model" in name or is_dis_fc_param:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False  # Disable gradients for other FC layers
                loss_dis.backward(retain_graph=True)
            else:
                loss_dis = None

            # COL dimension loss and backpropagation
            mask_col = ~torch.isnan(batch_labels[:, 3])
            masked_col_pred = col_pred[mask_col]
            masked_col_target = batch_labels[:, 3][mask_col]
            if masked_col_pred.numel() > 0:
                loss_col = F.mse_loss(masked_col_pred, masked_col_target)
                # Set requires_grad only for the AST backbone and col_fc layer
                for name, param in model.named_parameters():
                    # Enable gradients for AST backbone and col_fc only
                    is_col_fc_param = any(param is fc_param for fc_param in model.module.col_fc.parameters())
                    if "model" in name or is_col_fc_param:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False  # Disable gradients for other FC layers
                loss_col.backward(retain_graph=True)
            else:
                loss_col = None

            # LOUD dimension loss and backpropagation
            mask_loud = ~torch.isnan(batch_labels[:, 4])
            masked_loud_pred = loud_pred[mask_loud]
            masked_loud_target = batch_labels[:, 4][mask_loud]
            if masked_loud_pred.numel() > 0:
                loss_loud = F.mse_loss(masked_loud_pred, masked_loud_target)
                # Set requires_grad only for the AST backbone and loud_fc layer
                for name, param in model.named_parameters():
                    # Enable gradients for AST backbone and loud_fc only
                    is_loud_fc_param = any(param is fc_param for fc_param in model.module.loud_fc.parameters())
                    if "model" in name or is_loud_fc_param:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False  # Disable gradients for other FC layers
                loss_loud.backward()
            else:
                loss_loud = None

            # After all dimensions have accumulated their gradients, update model parameters once
            opt.step()
            opt.zero_grad()
            # Reset requires_grad to True for all parameters for the next batch
            for param in model.parameters():
                param.requires_grad = True
                
            # Filter out None values
            losses = [loss.item() for loss in [loss_mos, loss_noi, loss_dis, loss_col, loss_loud] if loss is not None]

            # Calculate the mean of available losses
            avg_b_loss = np.mean(losses) if losses else None

            # Update loss tracking
            epoch_train_loss += avg_b_loss
            batch_count += 1

            print(f"Epoch {epoch}, Batch {b+1}, Dimension Losses: MOS={loss_mos:.5f}, NOI={loss_noi:.5f}, DIS={loss_dis:.5f}, "
                  f"COL={loss_col:.5f}, LOUD={loss_loud:.5f}, AVG={avg_b_loss:.5f}")

            if batch_count == config['dev_flag']: break # Remove after setting up successfully

        # Save predicted outputs -------------------------------------------------------------------
        epoch_train_loss = epoch_train_loss/batch_count

        # Scale predictions once all batches are processed
        y_hat_train_descaled = y_hat_train * 4 + 1 # On CPU
        y_hat_train_descaled = y_hat_train_descaled.detach().numpy() # On CPU

        # Convert predictions into DataFrame columns, all on CPU
        ds_tr.df['mos_pred'] = y_hat_train_descaled[:, 0]
        ds_tr.df['noi_pred'] = y_hat_train_descaled[:, 1]
        ds_tr.df['dis_pred'] = y_hat_train_descaled[:, 2]
        ds_tr.df['col_pred'] = y_hat_train_descaled[:, 3]
        ds_tr.df['loud_pred'] = y_hat_train_descaled[:, 4]

        filtered_train_df = ds_tr.df.loc[
            (ds_tr.df['mos_pred'] != 0.0) &
            (ds_tr.df['noi_pred'] != 0.0) &
            (ds_tr.df['dis_pred'] != 0.0) &
            (ds_tr.df['col_pred'] != 0.0) &
            (ds_tr.df['loud_pred'] != 0.0)
        ]

        # Calculate metrics for training predictions --------------------------------------------------
        con_preds_tr_df = metrics.aggr_per_con_db(filtered_train_df) # On CPU

        tr_per_file_metrics_df = metrics.calc_metrics(filtered_train_df)
        tr_per_con_metrics_df = metrics.calc_metrics(con_preds_tr_df)

        # Validate batches -----------------------------------------------------------------------------
        model.eval()
        print(f'\nEpoch {epoch} Validation')

        # Reset helper variables --------------------------------------------------------------------
        batch_count = 0 # Reset batch_count to 0 at start of epoch validation
        epoch_val_loss = 0.0 # Reset epoch validation loss to 0 at start of epoch
        y_hat_val = torch.full((len(ds_val), 5), -0.25, device='cpu') # Stores the validation outputs, later filled into ds_val df

        with torch.no_grad():  # Disable gradient tracking for validation
            for b, (index, batch_features, batch_labels) in enumerate(dl_val):

                batch_features, batch_labels = batch_features.float().to(device), batch_labels.to(device)

                # Forward pass ---------------------------------------
                mos_pred, noi_pred, dis_pred, col_pred, loud_pred = model(batch_features)
                
                # Stack predictions for each dimension
                y_hat_batch = torch.stack([mos_pred, noi_pred, dis_pred, col_pred, loud_pred], dim=1).squeeze().to('cpu')
                y_hat_val[index, :] = y_hat_batch

                # Initialize batch loss for accumulation
                loss_list = []
                val_loss_mos = val_loss_noi = val_loss_dis = val_loss_col = val_loss_loud = None

                # Per-dimension evaluation
                for i, (pred, target) in enumerate(zip(
                        [mos_pred, noi_pred, dis_pred, col_pred, loud_pred],
                        [batch_labels[:, 0], batch_labels[:, 1], batch_labels[:, 2], batch_labels[:, 3], batch_labels[:, 4]])):

                    mask = ~torch.isnan(target)  # Mask for valid targets
                    masked_pred = pred[mask]
                    masked_target = target[mask]

                    if masked_pred.numel() > 0:  # Only if there are valid targets
                        dimension_loss = F.mse_loss(masked_pred, masked_target)
                        loss_list.append(dimension_loss.item())  # Accumulate for logging

                        if i == 0: val_loss_mos = dimension_loss.item()
                        if i == 1: val_loss_noi = dimension_loss.item()
                        if i == 2: val_loss_dis = dimension_loss.item()
                        if i == 3: val_loss_col = dimension_loss.item()
                        if i == 4: val_loss_loud = dimension_loss.item()

                val_b_loss = np.mean(loss_list) if loss_list else None
                epoch_val_loss += val_b_loss
                batch_count += 1

                print(f"Epoch {epoch}, Batch {b+1}, Dimension Losses: MOS={val_loss_mos:.5f}, NOI={val_loss_noi:.5f}, DIS={val_loss_dis:.5f}, "
                  f"COL={val_loss_col:.5f}, LOUD={val_loss_loud:.5f}, AVG={val_b_loss:.5f}")

                if batch_count == config['dev_flag']: break 

        # Save predicted outputs -------------------------------------------------------------------
        epoch_val_loss = epoch_val_loss/batch_count
        
        # Scale predictions once all batches are processed
        y_hat_val_descaled = y_hat_val * 4 + 1 # On CPU
        y_hat_val_descaled = y_hat_val_descaled.detach().numpy() # On CPU

        # Convert predictions into DataFrame columns on CPU
        ds_val.df['mos_pred'] = y_hat_val_descaled[:, 0]
        ds_val.df['noi_pred'] = y_hat_val_descaled[:, 1]
        ds_val.df['dis_pred'] = y_hat_val_descaled[:, 2]
        ds_val.df['col_pred'] = y_hat_val_descaled[:, 3]
        ds_val.df['loud_pred'] = y_hat_val_descaled[:, 4]

        filtered_val_df = ds_val.df.loc[
            (ds_val.df['mos_pred'] != 0.0) &
            (ds_val.df['noi_pred'] != 0.0) &
            (ds_val.df['dis_pred'] != 0.0) &
            (ds_val.df['col_pred'] != 0.0) &
            (ds_val.df['loud_pred'] != 0.0)
        ]

        con_preds_val_df = metrics.aggr_per_con_db(filtered_val_df)

        val_per_file_metrics_df = metrics.calc_metrics(filtered_val_df)
        val_per_con_metrics_df = metrics.calc_metrics(con_preds_val_df)

        columns_to_print = ["db", "mos_pcc", "mos_rmse", "noi_pcc", "noi_rmse", 
                            "dis_pcc", "dis_rmse", "col_pcc", "col_rmse", "loud_pcc", "loud_rmse"]
        selected_val_df = val_per_con_metrics_df[columns_to_print]
        print(tabulate(selected_val_df, headers='keys', tablefmt='psql'))

        # Early stopping check -----------------------------------------------------------------
        if epoch_val_loss <= best_loss:
            best_loss = epoch_val_loss # Update best loss
            es_patience = config['training']['es_patience'] # Reset early stopping patience

            # Save model predictions to file - per file and per condition - for training set and validation set
            filtered_train_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_train_preds_per_file.csv'), index=False)
            con_preds_tr_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_train_preds_per_condition.csv'), index=False)

            filtered_val_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_val_preds_per_file.csv'), index=False)
            con_preds_val_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_val_preds_per_condition.csv'), index=False)

            # Save model performance metrics per DB to file - per file and per condition - for training set and validation set
            tr_per_file_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_train_metrics_per_file.csv'), index=False)
            tr_per_con_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_train_metrics_per_con.csv'), index=False)

            val_per_file_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_val_metrics_per_file.csv'), index=False)
            val_per_con_metrics_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_val_metrics_per_con.csv'), index=False)

            # Save trained model -------------------------------------------------------------------
            model_checkpoint_name = os.path.basename(run_output_dir) + '.tar'
            model_checkpoint_path = os.path.join(run_output_dir, model_checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_val_loss
                }, model_checkpoint_path)
        else:
            es_patience -= 1

        scheduler.step(epoch_val_loss) 

        if device == "cuda": torch.cuda.empty_cache() # To prevent OOM error

        # Epoch logging  -----------------------------------------------------------------------------
        # Calculate the mean for each metric across all databases for training and validation
        #tr_avg_metrics = tr_per_con_metrics_df.select_dtypes(include=[np.number]).mean(skipna=False)
        #val_avg_metrics = val_per_con_metrics_df.select_dtypes(include=[np.number]).mean(skipna=False)
        #epoch_met_dict = utils.make_ep_met(tr_avg_metrics, val_avg_metrics)

        epoch_dict = {
            'epoch': epoch,
            'val_loss': round(epoch_val_loss, 6),
            'tr_loss': round(epoch_train_loss, 6),
            'es_patience': str(es_patience) + '/' + str(config['training']['es_patience']), 
            'duration': round(time.time() - epoch_tic, 4),
            'learning_rate': opt.param_groups[0]['lr']}
        
        #epoch_dict.update(epoch_met_dict)
        
        row_dict = pd.DataFrame([epoch_dict])
        epoch_log_df = pd.concat([epoch_log_df, row_dict], ignore_index=True)
        epoch_log_df.to_csv(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_log.csv'), index=False)

        if es_patience == 0:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    config['best_epoch'] =  epoch - config['training']['es_patience'] + 1
    with open(os.path.join(run_output_dir, os.path.basename(run_output_dir) + '_config_dump.yaml'), 'w') as yaml_file:
              yaml.dump(config, yaml_file, default_flow_style=False)

    print('\nTraining complete.')
