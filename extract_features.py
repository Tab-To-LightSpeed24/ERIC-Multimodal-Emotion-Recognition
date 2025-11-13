"""
ERIC Project: Feature Extraction Pipeline
This script extracts and saves multimodal features (text, audio, visual)
for the MELD dataset. These pre-extracted features are required for
running the 'contextual' model mode, as it speeds up training significantly.

Usage:
  - Run this script before training the contextual model for the first time:
    python extract_features.py
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from transformers import RobertaTokenizer, RobertaModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
import librosa
import warnings
warnings.filterwarnings('ignore')

# Import from the new unified config file
from config import Config

def print_header(text):
    """Prints a formatted header to the console."""
    print("\n" + "="*80)
    print(f" {text} ".center(80))
    print("="*80 + "\n")

def extract_text_features(csv_path, output_path, mode='train'):
    """Extracts RoBERTa embeddings for all utterances in a given CSV."""
    print_header(f"EXTRACTING TEXT FEATURES ({mode.upper()})")
    
    tokenizer = RobertaTokenizer.from_pretrained(Config.ROBERTA_MODEL)
    model = RobertaModel.from_pretrained(Config.ROBERTA_MODEL).to(Config.DEVICE)
    model.eval()
    
    df = pd.read_csv(csv_path)
    features = {}
    
    print(f"Processing {len(df)} utterances from {csv_path}...")
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Text Features"):
            key = (row['Dialogue_ID'], row['Utterance_ID'])
            text = str(row['Utterance'])
            
            try:
                encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=Config.MAX_SEQ_LENGTH, padding='max_length')
                encoding = {k: v.to(Config.DEVICE) for k, v in encoding.items()}
                
                outputs = model(**encoding)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                features[key] = cls_embedding.astype(np.float32)
            except Exception as e:
                print(f"Error processing text for {key}: {e}")
                features[key] = np.zeros(768, dtype=np.float32) # RoBERTa base has 768 dims
    
    np.save(output_path, features)
    print(f"✓ Saved {len(features)} text features to {output_path}")
    return features

def extract_audio_features(csv_path, video_dir, output_path, mode='train'):
    """Extracts Wav2Vec 2.0 embeddings from the audio track of video files."""
    print_header(f"EXTRACTING AUDIO FEATURES ({mode.upper()})")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(Config.DEVICE)
    model.eval()
    
    df = pd.read_csv(csv_path)
    features = {}
    failed_count = 0
    
    print(f"Processing {len(df)} utterances from {video_dir}...")
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Audio Features"):
            key = (row['Dialogue_ID'], row['Utterance_ID'])
            video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(video_dir, video_name)
            
            try:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                audio, sr = librosa.load(video_path, sr=16000, mono=True, duration=10)
                
                if len(audio) == 0:
                    raise ValueError("Audio signal is empty.")

                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                features[key] = embedding.astype(np.float32)
            except Exception as e:
                # print(f"Error processing audio for {key}: {e}")
                features[key] = np.zeros(768, dtype=np.float32) # Wav2Vec base has 768 dims
                failed_count += 1
    
    np.save(output_path, features)
    print(f"✓ Saved {len(features)} audio features to {output_path}")
    if failed_count > 0:
        print(f"  - Warning: {failed_count}/{len(df)} audio extractions failed and were replaced with zeros.")
    return features

def extract_visual_features(csv_path, video_dir, output_path, mode='train'):
    """Extracts ResNet-18 embeddings from video frames."""
    print_header(f"EXTRACTING VISUAL FEATURES ({mode.upper()})")
    
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(Config.DEVICE)
    resnet.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    df = pd.read_csv(csv_path)
    features = {}
    failed_count = 0
    
    print(f"Processing {len(df)} utterances from {video_dir}...")
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Visual Features"):
            key = (row['Dialogue_ID'], row['Utterance_ID'])
            video_name = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(video_dir, video_name)
            
            try:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file: {video_path}")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0:
                    raise ValueError("Video has no frames.")

                # Sample up to 5 frames evenly
                sample_indices = np.linspace(0, total_frames - 1, min(5, total_frames), dtype=int)
                
                frames = []
                for frame_idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        frame = transform(frame)
                        frames.append(frame)
                
                cap.release()
                
                if frames:
                    frames_tensor = torch.stack(frames).to(Config.DEVICE)
                    embedding = resnet(frames_tensor).mean(dim=0).squeeze().cpu().numpy()
                    features[key] = embedding.astype(np.float32)
                else:
                    raise ValueError("No frames could be read from video.")

            except Exception as e:
                # print(f"Error processing visual for {key}: {e}")
                features[key] = np.zeros(512, dtype=np.float32) # ResNet-18 output is 512
                failed_count += 1
    
    np.save(output_path, features)
    print(f"✓ Saved {len(features)} visual features to {output_path}")
    if failed_count > 0:
        print(f"  - Warning: {failed_count}/{len(df)} visual extractions failed and were replaced with zeros.")
    return features

def verify_features(features_path, expected_count, expected_dim):
    """Verifies the integrity of the saved feature files."""
    try:
        features = np.load(features_path, allow_pickle=True).item()
        num_features = len(features)
        actual_dim = list(features.values())[0].shape[0] if num_features > 0 else 0

        print(f"\n  Verifying {os.path.basename(features_path)}...")
        print(f"  - Found {num_features} features (Expected: {expected_count})")
        print(f"  - Feature dimension: {actual_dim} (Expected: {expected_dim})")

        assert num_features == expected_count, f"Feature count mismatch!"
        assert actual_dim == expected_dim, f"Feature dimension mismatch!"
        print("  ✓ Verification successful!")
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")

def main():
    """Main feature extraction pipeline."""
    print_header("ERIC PROJECT: FEATURE EXTRACTION PIPELINE")
    print(f"This script will extract features and save them to: {Config.FEATURES_DIR}")
    print(f"Using device: {Config.DEVICE}\n")
    
    os.makedirs(Config.FEATURES_DIR, exist_ok=True)
    
    datasets = {
        'train': (Config.TRAIN_CSV, Config.TRAIN_VIDEO_PATH),
        'dev': (Config.DEV_CSV, Config.DEV_VIDEO_PATH),
        'test': (Config.TEST_CSV, Config.TEST_VIDEO_PATH)
    }
    
    for mode, (csv_path, video_dir) in datasets.items():
        print_header(f"PROCESSING {mode.upper()} SET")
        
        df = pd.read_csv(csv_path)
        num_utterances = len(df)
        print(f"Found {num_utterances} utterances in {csv_path}")

        # Define paths
        text_path = os.path.join(Config.FEATURES_DIR, f'{mode}_text_features_roberta.npy')
        audio_path = os.path.join(Config.FEATURES_DIR, f'{mode}_audio_features_wav2vec.npy')
        visual_path = os.path.join(Config.FEATURES_DIR, f'{mode}_visual_features_resnet.npy')

        # --- Extract Features ---
        extract_text_features(csv_path, text_path, mode)
        extract_audio_features(csv_path, video_dir, audio_path, mode)
        extract_visual_features(csv_path, video_dir, visual_path, mode)

        # --- Verify Features ---
        print("\n--- Verifying Extracted Features ---")
        verify_features(text_path, num_utterances, 768)
        verify_features(audio_path, num_utterances, 768)
        verify_features(visual_path, num_utterances, 512)

    print_header("FEATURE EXTRACTION COMPLETE")
    print("All features have been extracted and saved.")
    print("\nYou can now train the 'contextual' model using main.py.")

if __name__ == "__main__":
    main()