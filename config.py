import os
import torch
import numpy as np
import random
from datetime import datetime

class Config:
    # --- High-Level Mode ---
    # NEW: 'baseline_pre' is our fast baseline
    MODEL_MODE = 'baseline_pre' 

    # --- Project Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "MELD-RAW", "MELD.Raw")
    
    TRAIN_VIDEO_PATH = os.path.join(DATA_ROOT, "train", "train_splits")
    TEST_VIDEO_PATH = os.path.join(DATA_ROOT, "test", "output_repeated_splits_test")
    DEV_VIDEO_PATH = os.path.join(DATA_ROOT, "dev", "dev_splits_complete")
    
    TRAIN_CSV = os.path.join(DATA_ROOT, "train", "train_sent_emo.csv")
    TEST_CSV = os.path.join(DATA_ROOT, "test_sent_emo.csv")
    DEV_CSV = os.path.join(DATA_ROOT, "dev_sent_emo.csv")
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
    
    ROBERTA_MODEL = "roberta-base"
    MAX_SEQ_LENGTH = 128
    NUM_EMOTIONS = 7
    EMOTION_LABELS = ['neutral', 'joy', 'surprise', 'anger', 'sadness', 'disgust', 'fear']
    
    FUSION_TYPE = 'late'
    HIDDEN_SIZE = 768
    DROPOUT = 0.2
    FUSION_DROPOUT = 0.3
    
    # --- Training Configuration (DEFAULTS) ---
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 20
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 2
    
    SAMPLE_RATE = 16000
    N_MFCC = 40
    N_MEL = 128
    FRAME_RATE = 3
    IMAGE_SIZE = (224, 224)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0
    
    EXPERIMENT_NAME = f"ERIC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    SEED = 42

    @classmethod
    def create_directories(cls):
        dirs = [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.LOG_DIR, 
                cls.RESULTS_DIR, cls.FEATURES_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def set_seed(cls):
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        torch.backends.cudnn.deterministic = True

    @classmethod
    def verify_paths(cls):
        critical_paths = {'Data Root': cls.DATA_ROOT, 'Train CSV': cls.TRAIN_CSV}
        all_exist = True
        for name, path in critical_paths.items():
            if not os.path.exists(path):
                print(f"✗ {name} NOT FOUND: {path}")
                all_exist = False
        return all_exist

    @classmethod
    def verify_features(cls):
        # Now, both 'contextual' and 'baseline_pre' need the features
        if cls.MODEL_MODE not in ['contextual', 'baseline_pre']:
            return True
            
        feature_files = [
            'train_text_features_roberta.npy', 'train_audio_features_wav2vec.npy', 'train_visual_features_resnet.npy',
            'dev_text_features_roberta.npy', 'dev_audio_features_wav2vec.npy', 'dev_visual_features_resnet.npy',
            'test_text_features_roberta.npy', 'test_audio_features_wav2vec.npy', 'test_visual_features_resnet.npy',
        ]
        all_exist = True
        for filename in feature_files:
            path = os.path.join(cls.FEATURES_DIR, filename)
            if not os.path.exists(path):
                print(f"✗ MISSING FEATURE: {filename}")
                all_exist = False
        return all_exist

def init_config(args):
    Config.MODEL_MODE = args.mode
    Config.NUM_EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    
    Config.TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.EXPERIMENT_NAME = f"ERIC_{Config.MODEL_MODE.upper()}_{Config.TIMESTAMP}"
    
    Config.create_directories()
    Config.set_seed()
    
    print("="*80)
    print(f"Configuration Initialized for '{Config.MODEL_MODE}' mode".center(80))
    print("="*80)
    print(f"Experiment: {Config.EXPERIMENT_NAME}")
    print(f"Device: {Config.DEVICE}")
    print(f"Mode: {Config.MODEL_MODE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    
    if not Config.verify_paths():
        raise FileNotFoundError("Critical data paths are missing.")
        
    if Config.MODEL_MODE in ['contextual', 'baseline_pre'] and not Config.verify_features():
        print("="*80)
        print("✗ ERROR: Missing pre-extracted feature files.")
        print("Please run `python extract_features.py` first.")
        print("="*80)
        raise FileNotFoundError("Missing pre-extracted feature files.")
        
    print("✓ All necessary paths and files verified.")
    print("="*80)
    return Config