"""
ERIC Project: Unified Training and Evaluation
Handles training and evaluation for all modes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
import time
from collections import defaultdict
import pickle

def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    def get_average(self, key):
        return np.mean(self.metrics[key]) if self.metrics[key] else 0

class Trainer:
    def __init__(self, model, train_loader, dev_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.config = config
        
        self.model.to(self.config.DEVICE)
        self.criterion = self._setup_criterion()
        
        self.optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        total_steps = len(train_loader) * config.NUM_EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=config.WARMUP_STEPS, num_training_steps=total_steps)
        
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        self.best_val_f1 = 0
        self.best_epoch = 0

    def _setup_criterion(self):
        if self.config.MODEL_MODE == 'contextual':
            return nn.CrossEntropyLoss(ignore_index=-100) # Ignore padding
        else: # 'baseline' or 'baseline_pre'
            return nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')
        
        for batch in progress_bar:
            batch = move_to_device(batch, self.config.DEVICE)
            
            # Forward pass
            logits = self.model(**batch)
            
            # Calculate loss
            if self.config.MODEL_MODE == 'contextual':
                loss = self.criterion(logits.view(-1, self.config.NUM_EMOTIONS), batch['labels'].view(-1))
            else: # 'baseline' or 'baseline_pre'
                loss = self.criterion(logits, batch['label'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            # Store predictions and labels
            with torch.no_grad():
                if self.config.MODEL_MODE == 'contextual':
                    preds = torch.argmax(logits, dim=2)
                    active_mask = (batch['labels'] != -100)
                    all_preds.extend(preds[active_mask].cpu().numpy())
                    all_labels.extend(batch['labels'][active_mask].cpu().numpy())
                else: # 'baseline' or 'baseline_pre'
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['label'].cpu().numpy())

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        
        self.train_metrics.update({'epoch_loss': avg_loss, 'epoch_acc': accuracy, 'epoch_f1': f1})
        return avg_loss, accuracy, f1

    def evaluate(self, loader, phase='val'):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'[{phase.capitalize()}]'):
                batch = move_to_device(batch, self.config.DEVICE)
                
                logits = self.model(**batch)
                
                if self.config.MODEL_MODE == 'contextual':
                    loss = self.criterion(logits.view(-1, self.config.NUM_EMOTIONS), batch['labels'].view(-1))
                    preds = torch.argmax(logits, dim=2)
                    active_mask = (batch['labels'] != -100)
                    all_preds.extend(preds[active_mask].cpu().numpy())
                    all_labels.extend(batch['labels'][active_mask].cpu().numpy())
                else: # 'baseline' or 'baseline_pre'
                    loss = self.criterion(logits, batch['label'])
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['label'].cpu().numpy())
                
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        report = classification_report(all_labels, all_preds, target_names=self.config.EMOTION_LABELS, output_dict=True, zero_division=0)
        
        return {'loss': avg_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'report': report, 'preds': all_preds, 'labels': all_labels}

    def train(self):
        print(f"\nStarting training in '{self.config.MODEL_MODE}' mode for {self.config.NUM_EPOCHS} epochs...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.dev_loader, phase='val')
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} Summary:")
            print(f"  Train -> Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Val   -> Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            self.val_metrics.update(val_metrics)
            
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt')
                print(f"  ✓ New best model saved! (Val F1: {self.best_val_f1:.4f})")

        print(f"\nTraining complete. Best Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}.")
        self.save_checkpoint('final_model.pt')
        return self.train_metrics, self.val_metrics

    def save_checkpoint(self, filename):
        filename = f"{self.config.MODEL_MODE}_{self.config.FUSION_TYPE}_{filename}"
        filepath = os.path.join(self.config.MODEL_DIR, filename)
        
        checkpoint = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {k: v for k, v in self.config.__class__.__dict__.items() if not k.startswith('__') and isinstance(v, (str, int, float, list, tuple))}
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to: {filepath}")

    def evaluate_test_set(self):
        print("\nEvaluating on the test set...")
        filename = f"{self.config.MODEL_MODE}_{self.config.FUSION_TYPE}_best_model.pt"
        checkpoint_path = os.path.join(self.config.MODEL_DIR, filename)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from: {checkpoint_path}")
        else:
            print("Warning: No best model checkpoint found. Evaluating the current model state.")
            
        test_metrics = self.evaluate(self.test_loader, phase='test')
        
        # Save predictions for final analysis
        self._save_predictions_for_analysis(test_metrics)
        
        self.generate_report(test_metrics)
        return test_metrics

    def _save_predictions_for_analysis(self, metrics):
        """Saves predictions and labels to a pickle file for later analysis."""
        
        # Define the filename for the predictions
        model_name = f"{self.config.MODEL_MODE}_{self.config.FUSION_TYPE}"
        results_path = os.path.join(self.config.RESULTS_DIR, f"{model_name}_predictions.pkl")
        
        # Prepare the data to be saved
        data_to_save = {
            'preds': metrics['preds'],
            'labels': metrics['labels']
        }
        
        # Save the data
        with open(results_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Predictions for model '{model_name}' saved to {results_path}")

    def generate_report(self, metrics):
        report = {
            'experiment_name': self.config.EXPERIMENT_NAME,
            'model_mode': self.config.MODEL_MODE,
            'fusion_type': self.config.FUSION_TYPE,
            'overall_metrics': {k: v for k, v in metrics.items() if isinstance(v, (float, int))},
            'class_wise_metrics': metrics['report'],
            'confusion_matrix': confusion_matrix(metrics['labels'], metrics['preds']).tolist()
        }
        
        report_path = os.path.join(self.config.RESULTS_DIR, f"final_report_{self.config.EXPERIMENT_NAME}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\nFinal evaluation report saved to: {report_path}")