# prediction script for Model-M
import os
import pandas as pd
import torch
from datasets import SQA_Test
from models import AST_pretrained
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import utils
import argparse
from datetime import datetime
from transformers import ASTFeatureExtractor


torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":

    '''output_dir = "/Users/wafaa/Code/psamd-master/multitask/ast_model_M/outputs"
    data_dir = "/Users/wafaa/DataBases/PSAMD_DB"
    data_file = "/Users/wafaa/Code/psamd-master/data/outputs/labels-28-July/label_28_July_stats.csv"'''

    checkpoint_path = "/Users/wafaa/Code/psamd/ast_models/ast_model_M/outputs/train_model_M_20241211_190443/train_model_M_20241211_190443.tar"
    # "/Users/wafaa/Code/psamd/ast_models/ast_model_M/outputs/full_train_20241211_181038/full_train_20241211_181038.tar"

    #----- Evaluating on Interspeech Challenge Test Databases --------------------------------
    output_dir = "/Users/wafaa/TubCloud/phd/writing/results"
    data_dir = "/Users/wafaa/DataBases/IS2022"
    data_file = "/Users/wafaa/DataBases/IS2022/full_eval_data_ci.csv"
    test_dbs = ['TencentCorupsVal'] # 'test', 'TencentCorupsVal'

    df_full = pd.read_csv(data_file)
    df_full.rename(columns={'deg_wav': 'file_path',
                            'mos': 'mos_file'}, inplace=True)

    bs = 12
    num_workers = 4

    #test_dbs = ['NISQA_TEST_NSC', 'TUB_2_LOUD', 'TUB_2_DIS', 'NISQA_TEST_FOR',
    #            'WB_48Hz_NTT_PTEST_2', 'NISQA_TEST_LIVETALK', 'NB_48kHz_NTT_PTEST_1',
    #            'NISQA_TEST_P501', 'GER_Handset', 'SUB_TUB_2', 'Vin726']
    
    #test_dbs = ['NISQA_TEST_NSC', 'NISQA_TEST_FOR', 'NISQA_TEST_LIVETALK','NISQA_TEST_P501', 'SUB_TUB_2', 'Vin726']

    #df_full = pd.read_csv(data_file)
    df = (
        df_full[df_full['db'].isin(test_dbs)]
        .copy()
        .reset_index(drop=True)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nPrediction begins...")
    print(f"Using device: {device}")

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    ds = SQA_Test(df, data_dir)

    dl = DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers
    )

    model = AST_pretrained()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    # Estimate model size
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)

    def model_size_in_MB(model):
        num_params = count_parameters(model)
        return num_params * 4 / (1024 ** 2)  # Convert bytes to MB

    num_params, model_size = utils.model_size_in_MB(model)
    print(f"Model has {num_params:,} parameters and model size is approximately {model_size:.2f} MB")


    y_hat_val = torch.full((len(ds), 5), -0.25, device='cpu') # Stores the validation outputs, later filled into ds_val df

    total_files = len(ds.df)  # Total number of files
    processed_files = 0       # Counter for processed files 

    with torch.no_grad():  # Disable gradient tracking for inference
        print("\nCalculating quality scores...")
        for b, (index, batch_features) in enumerate(dl):

            batch_features = batch_features.float().to(device)

            # Forward pass ---------------------------------------
            mos_pred, noi_pred, dis_pred, col_pred, loud_pred = model(batch_features)
            
            # Stack predictions for each dimension
            y_hat_batch = torch.stack([mos_pred, noi_pred, dis_pred, col_pred, loud_pred], dim=1).squeeze().to('cpu')
            y_hat_val[index, :] = y_hat_batch

            # Iterate through current batch to print scores with file paths
            for idx, scores in zip(index, y_hat_batch):
                idx = int(idx)  # Convert PyTorch tensor to native Python integer
                file_path = ds.df.loc[idx, 'file_path']  # Retrieve file path using index

                processed_files += 1
                # Descale predictions for display
                descaled_scores = scores * 4 + 1
                print(f"({processed_files}/{total_files}) {os.path.basename(file_path)} | MOS: {descaled_scores[0]:.2f}, "
                    f"NOI: {descaled_scores[1]:.2f}, DIS: {descaled_scores[2]:.2f}, "
                    f"COL: {descaled_scores[3]:.2f}, LOUD: {descaled_scores[4]:.2f}")

    # Scale predictions once all batches are processed
    y_hat_val_descaled = y_hat_val * 4 + 1 # On CPU
    y_hat_val_descaled = y_hat_val_descaled.detach().numpy() # On CPU

    # Convert predictions into DataFrame columns on CPU
    ds.df['mos_pred'] = y_hat_val_descaled[:, 0]
    ds.df['noi_pred'] = y_hat_val_descaled[:, 1]
    ds.df['dis_pred'] = y_hat_val_descaled[:, 2]
    ds.df['col_pred'] = y_hat_val_descaled[:, 3]
    ds.df['loud_pred'] = y_hat_val_descaled[:, 4]

    filtered_val_df = ds.df.loc[
        (ds.df['mos_pred'] != 0.0) &
        (ds.df['noi_pred'] != 0.0) &
        (ds.df['dis_pred'] != 0.0) &
        (ds.df['col_pred'] != 0.0) &
        (ds.df['loud_pred'] != 0.0)
    ]

    outfile = os.path.join(output_dir, 'multi_pred_per_file_' + current_time + '.csv')

    filtered_val_df.to_csv(outfile, index=False)
    print(f"\nPrediction results saved to {outfile}")