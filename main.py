import argparse
import time
import torch
import warnings
from config import Config, init_config 
from data_loader import create_data_loaders
from models import get_model
from trainer import Trainer
from fusion_analysis import FusionAnalyzer

warnings.filterwarnings('ignore')

# ... (print_header function is the same) ...
def print_header(text):
    print("\n" + "="*80)
    print(f" {text} ".center(80))
    print("="*80 + "\n")

def run_pipeline(args):
    pipeline_start_time = time.time()
    try:
        config = init_config(args)
        
        print_header("STEP 1: LOADING DATA")
        train_loader, dev_loader, test_loader = create_data_loaders(config)
        print("✓ Data loaders created successfully.")

        print_header("STEP 2: INITIALIZING MODEL")
        model = get_model(config)
        print("✓ Model initialized successfully.")

        trainer = Trainer(model, train_loader, dev_loader, test_loader, config)

        if not args.skip_training:
            print_header("STEP 3: TRAINING MODEL")
            train_metrics, val_metrics = trainer.train()
            print("✓ Training finished.")
        else:
            print_header("STEP 3: SKIPPING TRAINING")

        print_header("STEP 4: EVALUATING ON TEST SET")
        test_metrics = trainer.evaluate_test_set()
        
        print("\n--- Final Test Set Performance ---")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        
        total_time = time.time() - pipeline_start_time
        print(f"\nTotal pipeline execution time: {total_time / 60:.2f} minutes")
        print("="*80)
        print("PIPELINE FINISHED SUCCESSFULLY")

    except Exception as e:
        print(f"\n{'='*80}")
        print(" PIPELINE FAILED ".center(80, '✗'))
        print(f"An error occurred during the pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Runner for ERIC Project")
    
    parser.add_argument('--mode', type=str, required=True, 
                        # --- UPDATED CHOICES ---
                        choices=['baseline', 'baseline_pre', 'contextual'],
                        help="The model mode to run ('baseline' (slow), 'baseline_pre' (fast), or 'contextual' (fast)).")

    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument('--skip_training', action='store_true', help="Skip training.")
    parser.add_argument('--analysis', action='store_true', help="Run fusion analysis.")

    args = parser.parse_args()
    
    if args.mode == 'baseline':
        print_header("WARNING: SLOW BASELINE MODE")
        print("You are running in 'baseline' mode.")
        print("This will be EXTREMELY slow (10+ hours).")
        print("Use '--mode baseline_pre' for a fast, fair comparison.")
        print("="*80)
        time.sleep(5)
    
    run_pipeline(args)

if __name__ == "__main__":
    main()