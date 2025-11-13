"""
ERIC Project: Unified Data Loading and Preprocessing
Handles loading for all modes:
- 'baseline': Utterance-level, on-the-fly (SLOW)
- 'baseline_pre': Utterance-level, pre-extracted (FAST)
- 'contextual': Dialogue-level, pre-extracted (FAST)
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import cv2
import librosa
import numpy as np
from transformers import RobertaTokenizer
from PIL import Image
import torchvision.transforms as transforms
import os
import warnings
warnings.filterwarnings('ignore')

class MeldDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.csv_path = self._get_csv_path()
        self.df = pd.read_csv(self.csv_path)
        
        self.emotion2idx = {emotion: idx for idx, emotion in enumerate(config.EMOTION_LABELS)}

        if self.config.MODEL_MODE == 'baseline':
            self.video_dir = self._get_video_dir()
            self.tokenizer = RobertaTokenizer.from_pretrained(config.ROBERTA_MODEL)
            self.image_transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"Initialized 'baseline' (SLOW) dataset for {mode} with {len(self.df)} utterances.")
        
        elif self.config.MODEL_MODE in ['contextual', 'baseline_pre']:
            self._load_preextracted_features()
            if self.config.MODEL_MODE == 'contextual':
                self.dialogues = self._create_dialogue_list()
                print(f"Initialized 'contextual' dataset for {mode} with {len(self.dialogues)} dialogues.")
            else: # 'baseline_pre'
                # For baseline_pre, we just use the flat dataframe
                print(f"Initialized 'baseline_pre' (FAST) dataset for {mode} with {len(self.df)} utterances.")

    def _get_csv_path(self):
        if self.mode == 'train': return self.config.TRAIN_CSV
        elif self.mode == 'dev': return self.config.DEV_CSV
        else: return self.config.TEST_CSV

    def _get_video_dir(self):
        if self.mode == 'train': return self.config.TRAIN_VIDEO_PATH
        elif self.mode == 'dev': return self.config.DEV_VIDEO_PATH
        else: return self.config.TEST_VIDEO_PATH

    def _load_preextracted_features(self):
        """Loads pre-extracted features for contextual and baseline_pre modes."""
        print(f"Loading pre-extracted features for {self.mode} set...")
        features_dir = self.config.FEATURES_DIR
        self.text_features = np.load(os.path.join(features_dir, f'{self.mode}_text_features_roberta.npy'), allow_pickle=True).item()
        self.audio_features = np.load(os.path.join(features_dir, f'{self.mode}_audio_features_wav2vec.npy'), allow_pickle=True).item()
        self.visual_features = np.load(os.path.join(features_dir, f'{self.mode}_visual_features_resnet.npy'), allow_pickle=True).item()

    def _create_dialogue_list(self):
        dialogues = []
        grouped = self.df.groupby('Dialogue_ID')
        for _, group in grouped:
            group = group.sort_values('Utterance_ID')
            dialogue = [{'dialogue_id': row['Dialogue_ID'], 'utterance_id': row['Utterance_ID'], 'emotion': row['Emotion']} for _, row in group.iterrows()]
            dialogues.append(dialogue)
        return dialogues

    def __len__(self):
        if self.config.MODEL_MODE == 'contextual':
            return len(self.dialogues)
        return len(self.df) # For 'baseline' and 'baseline_pre'

    def __getitem__(self, idx):
        if self.config.MODEL_MODE == 'baseline':
            return self._get_utterance_item_slow(idx)
        elif self.config.MODEL_MODE == 'baseline_pre':
            return self._get_utterance_item_fast(idx)
        else: # contextual
            return self._get_dialogue_item(idx)

    def _get_utterance_item_slow(self, idx):
        """(SLOW) Gets a single utterance with on-the-fly feature extraction."""
        row = self.df.iloc[idx]
        dialogue_id, utterance_id = row['Dialogue_ID'], row['Utterance_ID']
        
        video_name = f"dia{dialogue_id}_utt{utterance_id}.mp4"
        if self.mode == 'test':
            video_name = f"final_videos_testdia{dialogue_id}_utt{utterance_id}.mp4"
        video_path = os.path.join(self.video_dir, video_name)

        text = str(row['Utterance'])
        text_features = self._process_text(text)
        audio_features = self._extract_audio_features(video_path)
        visual_features = self._extract_visual_features(video_path)
        
        emotion = row['Emotion'].lower()
        label = self.emotion2idx.get(emotion, 0)
        
        return {'text': text_features, 'audio': audio_features, 'visual': visual_features, 'label': torch.tensor(label, dtype=torch.long)}

    def _get_utterance_item_fast(self, idx):
        """(FAST) Gets a single utterance from pre-extracted features."""
        row = self.df.iloc[idx]
        key = (row['Dialogue_ID'], row['Utterance_ID'])
        
        text_feat = torch.tensor(self.text_features.get(key, np.zeros(768)), dtype=torch.float32)
        audio_feat = torch.tensor(self.audio_features.get(key, np.zeros(768)), dtype=torch.float32)
        visual_feat = torch.tensor(self.visual_features.get(key, np.zeros(512)), dtype=torch.float32)
        
        emotion = row['Emotion'].lower()
        label = self.emotion2idx.get(emotion, 0)

        return {'text_features': text_feat, 'audio_features': audio_feat, 'visual_features': visual_feat, 'label': torch.tensor(label, dtype=torch.long)}

    def _get_dialogue_item(self, idx):
        """Gets a full dialogue sequence from pre-extracted features."""
        dialogue = self.dialogues[idx]
        text_feats, audio_feats, visual_feats, labels = [], [], [], []
        
        for utt in dialogue:
            key = (utt['dialogue_id'], utt['utterance_id'])
            text_feats.append(torch.tensor(self.text_features.get(key, np.zeros(768)), dtype=torch.float32))
            audio_feats.append(torch.tensor(self.audio_features.get(key, np.zeros(768)), dtype=torch.float32))
            visual_feats.append(torch.tensor(self.visual_features.get(key, np.zeros(512)), dtype=torch.float32))
            labels.append(self.emotion2idx.get(utt['emotion'].lower(), 0))

        return {'text_features': torch.stack(text_feats), 'audio_features': torch.stack(audio_feats), 'visual_features': torch.stack(visual_feats), 'labels': torch.tensor(labels, dtype=torch.long)}

    # --- On-the-fly methods (only for 'baseline' mode) ---
    def _process_text(self, text):
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.config.MAX_SEQ_LENGTH, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze()}

    def _extract_audio_features(self, video_path):
        try:
            audio, sr = librosa.load(video_path, sr=self.config.SAMPLE_RATE)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.config.N_MFCC)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.config.N_MEL)
            features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.mean(mel_spec, axis=1), np.std(mel_spec, axis=1)])
            return torch.tensor(features, dtype=torch.float32)
        except Exception:
            return torch.zeros(self.config.N_MFCC * 2 + self.config.N_MEL * 2, dtype=torch.float32)

    def _extract_visual_features(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened() and len(frames) < 10:
                ret, frame = cap.read()
                if not ret: break
                frame = self.image_transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR_RGB)))
                frames.append(frame)
            cap.release()
            return torch.mean(torch.stack(frames), dim=0) if frames else torch.zeros(3, *self.config.IMAGE_SIZE)
        except Exception:
            return torch.zeros(3, *self.config.IMAGE_SIZE)

def contextual_collate_fn(batch):
    """Pads dialogue sequences for contextual mode."""
    text_features = pad_sequence([item['text_features'] for item in batch], batch_first=True)
    audio_features = pad_sequence([item['audio_features'] for item in batch], batch_first=True)
    visual_features = pad_sequence([item['visual_features'] for item in batch], batch_first=True)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)
    attention_mask = (labels != -100)
    
    return {'text_features': text_features, 'audio_features': audio_features, 'visual_features': visual_features, 'labels': labels, 'attention_mask': attention_mask}

def create_data_loaders(config):
    """Creates data loaders for train, dev, and test sets based on config mode."""
    train_dataset = MeldDataset(config, mode='train')
    dev_dataset = MeldDataset(config, mode='dev')
    test_dataset = MeldDataset(config, mode='test')
    
    # Use collate_fn only for contextual mode
    collate_fn = contextual_collate_fn if config.MODEL_MODE == 'contextual' else None
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    
    return train_loader, dev_loader, test_loader