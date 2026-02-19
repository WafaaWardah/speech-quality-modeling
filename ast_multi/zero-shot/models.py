# ASTXL Model: A PyTorch model using the pre-trained AST from MIT for audio feature prediction. 
# This is intended for pricting one score in the range 0-1.

import sys
import torch
import torch.nn as nn
from transformers import ASTModel, Wav2Vec2Model, HubertModel, WavLMModel, WhisperModel, logging

logging.set_verbosity_error()  # Silence unnecessary warnings

class W2V2(nn.Module):

    PRETRAINED_MODEL = "facebook/wav2vec2-base" # Theis was SSL only, no fine-tuning towards ASR

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super(W2V2, self).__init__()
        try:
            self.w2v2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e
        
        self.w2v2.gradient_checkpointing_disable()
        # Freeze W2V2 parameters to prevent training
        for p in self.w2v2.parameters():
            p.requires_grad = False


        self.fc = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            out = self.w2v2(input_values=wav, attention_mask=attention_mask)
            hs = out.last_hidden_state  # [B, S, H]

            if attention_mask is not None:
                # Convert waveform mask -> feature (hs) mask
                feat_mask = self.w2v2._get_feature_vector_attention_mask(hs.shape[1], attention_mask)  # [B, S]
                feat_mask = feat_mask.to(hs.dtype).unsqueeze(-1)  # [B, S, 1]
                pooled = (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1.0)  # [B, H]
            else:
                pooled = hs.mean(dim=1)

        pred = self.fc(pooled).squeeze(-1)
        return pred


class HuBERT(nn.Module):
    PRETRAINED_MODEL = "facebook/hubert-base-ls960"  # base checkpoint

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super().__init__()
        try:
            self.hubert = HubertModel.from_pretrained(pretrained_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e

        # Disable gradient checkpointing (clean logs; also pointless for frozen backbones)
        if hasattr(self.hubert, "gradient_checkpointing_disable"):
            self.hubert.gradient_checkpointing_disable()

        # Freeze HuBERT parameters
        for p in self.hubert.parameters():
            p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.hubert.config.hidden_size, 1),  # base -> 768
            nn.Sigmoid()
        )

        # Keep frozen backbone deterministic (dropout off)
        self.hubert.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.hubert.eval()
        return self

    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            out = self.hubert(input_values=wav, attention_mask=attention_mask)
            hs = out.last_hidden_state  # [B, S, 768]

            if attention_mask is not None:
                # Convert waveform mask -> feature (hs) mask for correct pooling
                feat_mask = self.hubert._get_feature_vector_attention_mask(hs.shape[1], attention_mask)  # [B, S]
                feat_mask = feat_mask.to(hs.dtype).unsqueeze(-1)  # [B, S, 1]
                pooled = (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1.0)  # [B, 768]
                print("\n\nhs:", hs.shape)
                print("attention_mask:", attention_mask.shape)
                print("feat_mask:", feat_mask.shape)
                print("pooled:", pooled.shape)
                sys.exit()
            else:
                pooled = hs.mean(dim=1)

        pred = self.fc(pooled).squeeze(-1)  # [B]
        return pred
    

class WavLM(nn.Module):
    PRETRAINED_MODEL = "microsoft/wavlm-base"

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super().__init__()
        try:
            self.wavlm = WavLMModel.from_pretrained(pretrained_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e

        # Disable gradient checkpointing (clean logs; pointless for frozen backbones)
        if hasattr(self.wavlm, "gradient_checkpointing_disable"):
            self.wavlm.gradient_checkpointing_disable()

        # Freeze WavLM parameters
        for p in self.wavlm.parameters():
            p.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.wavlm.config.hidden_size, 1),  # base -> 768
            nn.Sigmoid()
        )

        # Keep frozen backbone deterministic
        self.wavlm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.wavlm.eval()
        return self

    def forward(self, wav: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        with torch.no_grad():
            out = self.wavlm(input_values=wav, attention_mask=attention_mask)
            hs = out.last_hidden_state  # [B, S, 768]

            if attention_mask is not None:
                # Convert waveform mask -> feature (hs) mask for correct pooling
                feat_mask = self.wavlm._get_feature_vector_attention_mask(hs.shape[1], attention_mask)  # [B, S]
                feat_mask = feat_mask.to(hs.dtype).unsqueeze(-1)  # [B, S, 1]
                pooled = (hs * feat_mask).sum(dim=1) / feat_mask.sum(dim=1).clamp(min=1.0)  # [B, 768]
            else:
                pooled = hs.mean(dim=1)

        pred = self.fc(pooled).squeeze(-1)  # [B]
        return pred

class Whisper(nn.Module):
    PRETRAINED_MODEL = "openai/whisper-base"

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super().__init__()
        try:
            self.whisper = WhisperModel.from_pretrained(pretrained_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e

        # Freeze backbone
        for p in self.whisper.parameters():
            p.requires_grad = False

        # whisper-base: d_model = 512
        self.fc = nn.Sequential(
            nn.Linear(self.whisper.config.d_model, 1),
            nn.Sigmoid()
        )

        self.whisper.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.whisper.eval()
        return self

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        input_values: [B, 80, T']  (Whisper log-mel features)
        """
        with torch.no_grad():
            enc = self.whisper.encoder(input_features=input_values, attention_mask=attention_mask)
            hs = enc.last_hidden_state  # [B, S, 512]
            pooled = hs.mean(dim=1)     # [B, 512]

        return self.fc(pooled).squeeze(-1)  # [B]


class AST(nn.Module):

    PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super(AST, self).__init__()
        try:
            self.ast = ASTModel.from_pretrained(pretrained_model)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e
        
        # Freeze AST parameters to prevent training
        for p in self.ast.parameters():
            p.requires_grad = False


        self.fc = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            hidden_state = self.ast(features).pooler_output
        pred = self.fc(hidden_state).squeeze()
        return pred
    
    