"""
ERIC Project: Unified Multimodal Models
Contains all model definitions for baseline (utterance-level) and
contextual (dialogue-level) emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import torchvision.models as models

# --- SLOW Baseline (Utterance-Level) Components ---
# --- (These are now only used for 'baseline' mode) ---

class TextEncoder(nn.Module):
    """(SLOW) RoBERTa-based text encoder for raw text."""
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(config.ROBERTA_MODEL)
        self.projection = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, 512),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_output)

class AudioEncoder(nn.Module):
    """(SLOW) Audio feature encoder for MFCCs and Mel spectrograms."""
    def __init__(self, config):
        super().__init__()
        input_dim = config.N_MFCC * 2 + config.N_MEL * 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(config.DROPOUT),
            nn.Linear(256, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(config.DROPOUT),
            nn.Linear(512, 512), nn.ReLU(), nn.BatchNorm1d(512)
        )
        
    def forward(self, audio_features):
        return self.encoder(audio_features)

class VisualEncoder(nn.Module):
    """(SLOW) Visual feature encoder using a pre-trained ResNet."""
    def __init__(self, config):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(config.DROPOUT))
        
    def forward(self, visual_input):
        features = self.features(visual_input).view(visual_input.size(0), -1)
        return self.projection(features)

class LateFusion(nn.Module):
    """(SLOW) Averages predictions from modality-specific classifiers."""
    def __init__(self, config):
        super().__init__()
        self.text_classifier = nn.Linear(512, config.NUM_EMOTIONS)
        self.audio_classifier = nn.Linear(512, config.NUM_EMOTIONS)
        self.visual_classifier = nn.Linear(512, config.NUM_EMOTIONS)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, text_features, audio_features, visual_features):
        text_logits = self.text_classifier(text_features)
        audio_logits = self.audio_classifier(audio_features)
        visual_logits = self.visual_classifier(visual_features)
        
        weights = F.softmax(self.fusion_weights, dim=0)
        logits = weights[0] * text_logits + weights[1] * audio_logits + weights[2] * visual_logits
        return logits, weights

class MultimodalEmotionRecognizer(nn.Module):
    """(SLOW) Complete utterance-level multimodal model using Late Fusion."""
    def __init__(self, config, enabled_modalities=None):
        super().__init__()
        # ... (This whole class stays the same, it's our "slow" baseline)
        self.config = config
        self.enabled_modalities = enabled_modalities if enabled_modalities is not None else ['text', 'audio', 'visual']
        self.text_encoder = TextEncoder(config) if 'text' in self.enabled_modalities else None
        self.audio_encoder = AudioEncoder(config) if 'audio' in self.enabled_modalities else None
        self.visual_encoder = VisualEncoder(config) if 'visual' in self.enabled_modalities else None
        self.fusion = LateFusion(config)
            
    def forward(self, text=None, audio=None, visual=None, return_features=False, **kwargs):
        batch_size = -1
        if text is not None: batch_size = text['input_ids'].size(0)
        elif audio is not None: batch_size = audio.size(0)
        elif visual is not None: batch_size = visual.size(0)
        if batch_size == -1: raise ValueError("At least one input modality must be provided.")
        text_features = self.text_encoder(text['input_ids'], text['attention_mask']) if self.text_encoder and text is not None else torch.zeros(batch_size, 512).to(self.config.DEVICE)
        audio_features = self.audio_encoder(audio) if self.audio_encoder and audio is not None else torch.zeros(batch_size, 512).to(self.config.DEVICE)
        visual_features = self.visual_encoder(visual) if self.visual_encoder and visual is not None else torch.zeros(batch_size, 512).to(self.config.DEVICE)
        logits, _ = self.fusion(text_features, audio_features, visual_features)
        if return_features:
            return {'logits': logits, 'text_features': text_features, 'audio_features': audio_features, 'visual_features': visual_features}
        return logits

# --- NEW: FAST Baseline (Utterance-Level) Model ---

class BaselinePreExtracted(nn.Module):
    """
    (FAST) Utterance-level baseline that uses pre-extracted features.
    This is the model for 'baseline_pre' mode.
    It performs the same job as MultimodalEmotionRecognizer but is much faster.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Project features to a common dimension (e.g., 512)
        self.text_proj = nn.Linear(768, 512) # RoBERTa(768) -> 512
        self.audio_proj = nn.Linear(768, 512) # Wav2Vec(768) -> 512
        self.visual_proj = nn.Linear(512, 512) # ResNet(512) -> 512
        
        # We re-use the same LateFusion logic, but on the new projected features
        self.fusion = LateFusion(config)

    def forward(self, text_features, audio_features, visual_features, return_features=False, **kwargs):
        # Project features
        text_feat = F.relu(self.text_proj(text_features))
        audio_feat = F.relu(self.audio_proj(audio_features))
        visual_feat = F.relu(self.visual_proj(visual_features))
        
        # Fused logits
        logits, _ = self.fusion(text_feat, audio_feat, visual_feat)
        
        if return_features:
            return {'logits': logits, 'text_features': text_feat, 'audio_features': audio_feat, 'visual_features': visual_feat}
        
        return logits

# --- Contextual (Dialogue-Level) Model ---

class ContextualTransformer(nn.Module):
    """
    Contextual Transformer for dialogue-level emotion recognition.
    Uses pre-extracted features and models temporal context with late fusion.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_dim = 768
        self._build_late_fusion()

    def _build_late_fusion(self):
        self.text_projection = nn.Linear(768, self.model_dim)
        self.audio_projection = nn.Linear(768, self.model_dim)
        self.visual_projection = nn.Linear(512, self.model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=8, dim_feedforward=2048, dropout=self.config.DROPOUT, activation='relu', batch_first=True)
        self.transformer_text = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.transformer_audio = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.transformer_visual = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.classifier = nn.Linear(self.model_dim * 3, self.config.NUM_EMOTIONS)

    def forward(self, text_features, audio_features, visual_features, attention_mask, return_features=False, **kwargs):
        src_key_padding_mask = ~attention_mask

        text_out = self.transformer_text(self.text_projection(text_features), src_key_padding_mask=src_key_padding_mask)
        audio_out = self.transformer_audio(self.audio_projection(audio_features), src_key_padding_mask=src_key_padding_mask)
        visual_out = self.transformer_visual(self.visual_projection(visual_features), src_key_padding_mask=src_key_padding_mask)
        combined = torch.cat([text_out, audio_out, visual_out], dim=-1)
        logits = self.classifier(combined)
        
        if return_features:
            return {'logits': logits, 'text_features': text_out, 'audio_features': audio_out, 'visual_features': visual_out}
        return logits

# --- Model Factory ---

def get_model(config, enabled_modalities=None):
    if config.MODEL_MODE == 'baseline':
        print("Initializing (SLOW) 'MultimodalEmotionRecognizer'...")
        model = MultimodalEmotionRecognizer(config)
    
    # --- OUR NEW FAST BASELINE ---
    elif config.MODEL_MODE == 'baseline_pre':
        print("Initializing (FAST) 'BaselinePreExtracted'...")
        model = BaselinePreExtracted(config)
    
    elif config.MODEL_MODE == 'contextual':
        print("Initializing 'ContextualTransformer'...")
        model = ContextualTransformer(config)
    else:
        raise ValueError(f"Invalid MODEL_MODE: {config.MODEL_MODE}")

    model = model.to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} (late fusion)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model