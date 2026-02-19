import torch
import torchaudio
from torch import nn as nn
from transformers import ASTModel, ASTConfig


class AST_48_untrained(nn.Module): # Untrained AST but the smaller size with 128 n_mels
    def __init__(self, n_mels=1024):
        super(AST_48_untrained, self).__init__()
        
        # Initialize an untrained AST model from scratch
        config = ASTConfig(num_mel_bins=128)
        self.model = ASTModel(config)
        
        # Projection layer to match AST input dimension
        self.input_projection = nn.Linear(n_mels, 128) 
        
        # Output layers for each dimension
        self.mos_fc = nn.Linear(768, 1)
        self.noi_fc = nn.Linear(768, 1)
        self.dis_fc = nn.Linear(768, 1)
        self.col_fc = nn.Linear(768, 1)
        self.loud_fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        n_mels = features.shape[-1]
        if n_mels != self.input_projection.in_features:
            raise ValueError(f"Expected {self.input_projection.in_features} mel filters, but got {n_mels}.")

        # Project the feature dimension to match AST expected input
        projected_features = self.input_projection(features)
        
        # Pass through the AST model, using input_values
        hidden_state = self.model(input_values=projected_features).pooler_output

        # Separate output heads for each dimension, with Sigmoid activation
        mos_pred = self.sigmoid(self.mos_fc(hidden_state)).squeeze()
        noi_pred = self.sigmoid(self.noi_fc(hidden_state)).squeeze()
        dis_pred = self.sigmoid(self.dis_fc(hidden_state)).squeeze()
        col_pred = self.sigmoid(self.col_fc(hidden_state)).squeeze()
        loud_pred = self.sigmoid(self.loud_fc(hidden_state)).squeeze()

        return mos_pred, noi_pred, dis_pred, col_pred, loud_pred


class AST_48_untrained_1024(nn.Module):
    def __init__(self, n_mels=1024):
        super(AST_48_untrained_1024, self).__init__()
        
        # Initialize an untrained AST model from scratch
        config = ASTConfig(num_mel_bins=n_mels)
        self.model = ASTModel(config)

        # Assume hidden size for the dummy layers (e.g., 768)
        #hidden_size = 768  # Use any reasonable size that matches your pipeline's needs DEV TEST!!!
        
        # Output layers for each dimension
        self.mos_fc = nn.Linear(config.hidden_size, 1)
        self.noi_fc = nn.Linear(config.hidden_size, 1)
        self.dis_fc = nn.Linear(config.hidden_size, 1)
        self.col_fc = nn.Linear(config.hidden_size, 1)
        self.loud_fc = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        # Pass features directly into the AST model
        hidden_state = self.model(features).pooler_output

        #batch_size = features.size(0) #DEV TEST!!!
        #hidden_state = torch.randn(batch_size, 768) # DEV TEST!!!

        # Separate output heads for each dimension, with Sigmoid activation
        mos_pred = self.sigmoid(self.mos_fc(hidden_state)).squeeze()
        noi_pred = self.sigmoid(self.noi_fc(hidden_state)).squeeze()
        dis_pred = self.sigmoid(self.dis_fc(hidden_state)).squeeze()
        col_pred = self.sigmoid(self.col_fc(hidden_state)).squeeze()
        loud_pred = self.sigmoid(self.loud_fc(hidden_state)).squeeze()

        return mos_pred, noi_pred, dis_pred, col_pred, loud_pred

class AST_48(nn.Module):
    def __init__(self, n_mels=1024):
        super(AST_48, self).__init__()
        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Projection layer to match AST input dimension
        self.input_projection = nn.Linear(n_mels, 128)  # Ensure input projection aligns with expected model input
        
        # Output layers for each dimension
        self.mos_fc = nn.Linear(768, 1)
        self.noi_fc = nn.Linear(768, 1)
        self.dis_fc = nn.Linear(768, 1)
        self.col_fc = nn.Linear(768, 1)
        self.loud_fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        n_mels = features.shape[-1]
        if n_mels != self.input_projection.in_features:
            raise ValueError(f"Expected {self.input_projection.in_features} mel filters, but got {n_mels}.")

        # Project the feature dimension to match AST expected input
        projected_features = self.input_projection(features)
        
        # Pass through the AST model, using input_values
        hidden_state = self.model(input_values=projected_features).pooler_output

        # Separate output heads for each dimension, with Sigmoid activation
        mos_pred = self.sigmoid(self.mos_fc(hidden_state)).squeeze()
        noi_pred = self.sigmoid(self.noi_fc(hidden_state)).squeeze()
        dis_pred = self.sigmoid(self.dis_fc(hidden_state)).squeeze()
        col_pred = self.sigmoid(self.col_fc(hidden_state)).squeeze()
        loud_pred = self.sigmoid(self.loud_fc(hidden_state)).squeeze()

        return mos_pred, noi_pred, dis_pred, col_pred, loud_pred


class AST_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.mos_fc = nn.Linear(768, 1)
        self.noi_fc = nn.Linear(768, 1)
        self.dis_fc = nn.Linear(768, 1)
        self.col_fc = nn.Linear(768, 1)
        self.loud_fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        hidden_state = self.model(features).pooler_output
        # Pass through each independent FC layer and apply sigmoid
        mos_pred = self.sigmoid(self.mos_fc(hidden_state)).squeeze()
        noi_pred = self.sigmoid(self.noi_fc(hidden_state)).squeeze()
        dis_pred = self.sigmoid(self.dis_fc(hidden_state)).squeeze()
        col_pred = self.sigmoid(self.col_fc(hidden_state)).squeeze()
        loud_pred = self.sigmoid(self.loud_fc(hidden_state)).squeeze()
        
        # Return all predictions
        return mos_pred, noi_pred, dis_pred, col_pred, loud_pred