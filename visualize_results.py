import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Hardcoded model complexity (number of parameters)
MODEL_COMPLEXITY = {
    'BASELINE_PRE': 1.5e6,  # Example value
    'CONTEXTUAL': 2.8e6    # Example value
}

# Placeholder for training journey data
TRAINING_JOURNEY_DATA = {
    'run1': {'epoch': [1, 2, 3, 4, 5], 'val_f1': [0.6, 0.65, 0.68, 0.7, 0.71]},
    'run2': {'epoch': [1, 2, 3, 4, 5], 'val_f1': [0.62, 0.66, 0.69, 0.72, 0.73]},
    'run3': {'epoch': [1, 2, 3, 4, 5], 'val_f1': [0.59, 0.64, 0.67, 0.69, 0.70]},
}

def load_all_reports(reports_dir):
    """Loads all final_report_*.json files into a pandas DataFrame."""
    print("--- Loading and Analyzing Reports ---")
    report_files = glob.glob(os.path.join(reports_dir, "final_report_*.json"))
    
    all_data = []
    for report_path in report_files:
        basename = os.path.basename(report_path)
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON file: {basename}")
            continue

        # --- UNIFIED NAME & MODE EXTRACTION ---
        experiment_name = report.get('experiment_name')
        mode = 'UNKNOWN'

        if experiment_name:  # New format
            if "BASELINE_PRE" in experiment_name:
                mode = 'BASELINE_PRE'
            elif "CONTEXTUAL" in experiment_name:
                mode = 'CONTEXTUAL'
        else:  # Potentially old format
            experiment_info = report.get('experiment_info', {})
            experiment_name = experiment_info.get('name')
            if experiment_name:
                if experiment_info.get('fusion_type'):
                    mode = 'CONTEXTUAL'
                else:
                    mode = 'BASELINE_PRE'
        
        if not experiment_name:
            print(f"Warning: Skipping report with no identifiable name: {basename}")
            continue

        # --- ROBUST METRIC EXTRACTION ---
        # Handles multiple report structures to find F1, accuracy, and CM.
        test_f1 = None
        test_accuracy = None
        cm = []
        class_report = {}

        if 'test_results' in report: # Structure 1: `test_results` key
            test_results = report['test_results']
            test_f1 = test_results.get('f1_score')
            test_accuracy = test_results.get('accuracy')
            cm = test_results.get('confusion_matrix', [])
            class_report = test_results.get('classification_report', {})
        elif 'class_wise_metrics' in report: # Structure 2: `class_wise_metrics` key
            cwm = report['class_wise_metrics']
            if 'weighted avg' in cwm:
                test_f1 = cwm['weighted avg'].get('f1-score')
            test_accuracy = cwm.get('accuracy')
            cm = report.get('confusion_matrix', [])
            class_report = cwm
        
        if isinstance(cm, dict): # Handle cases where CM might be a dict
            cm = cm.get('values', [])

        all_data.append({
            'experiment_name': experiment_name,
            'mode': mode,
            'test_f1': test_f1,
            'test_accuracy': test_accuracy,
            'params': MODEL_COMPLEXITY.get(mode, 0),
            'confusion_matrix': cm,
            'class_report': class_report
        })

    df = pd.DataFrame(all_data)
    df['test_f1'] = pd.to_numeric(df['test_f1'], errors='coerce')
    df['test_accuracy'] = pd.to_numeric(df['test_accuracy'], errors='coerce')
    return df

def plot_performance_vs_complexity(df, output_dir):
    """Plots F1-score vs. Model Complexity."""
    if df.empty:
        print("Warning: DataFrame is empty. Skipping performance vs. complexity plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='params', y='test_f1', hue='mode', style='mode', s=150, alpha=0.8)
    
    plt.xscale('log')
    plt.title('Performance vs. Model Complexity')
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Test F1-Score (weighted)')
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Model Type')
    
    output_path = os.path.join(output_dir, 'performance_vs_complexity.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved performance vs. complexity plot to {output_path}")

def plot_confusion_matrices(baseline_run, contextual_run, output_dir):
    """Plots side-by-side normalized confusion matrices."""
    if baseline_run is None:
        print("Warning: Best baseline run not found. Skipping its confusion matrix.")
    if contextual_run is None:
        print("Warning: Best contextual run not found. Skipping its confusion matrix.")
    if baseline_run is None and contextual_run is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Baseline ---
    if baseline_run is not None:
        cm_base = baseline_run['confusion_matrix']
        if cm_base and isinstance(cm_base, list) and len(cm_base) > 0 and sum(map(sum, cm_base)) > 0:
            cm_base_norm = [[cell / sum(row) if sum(row) > 0 else 0 for cell in row] for row in cm_base]
            sns.heatmap(cm_base_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[0])
            axes[0].set_title('Best Baseline Model (Normalized)')
            axes[0].set_xlabel('Predicted Label')
            axes[0].set_ylabel('True Label')
        else:
            axes[0].text(0.5, 0.5, 'Baseline CM not available\nor is empty', ha='center', va='center')
            axes[0].set_title('Best Baseline Model')
    else:
        axes[0].text(0.5, 0.5, 'Baseline Run Not Found', ha='center', va='center')
        axes[0].set_title('Best Baseline Model')


    # --- Contextual ---
    if contextual_run is not None:
        cm_context = contextual_run['confusion_matrix']
        if cm_context and isinstance(cm_context, list) and len(cm_context) > 0 and sum(map(sum, cm_context)) > 0:
            cm_context_norm = [[cell / sum(row) if sum(row) > 0 else 0 for cell in row] for row in cm_context]
            sns.heatmap(cm_context_norm, annot=True, fmt='.2%', cmap='Oranges', ax=axes[1])
            axes[1].set_title('Best Contextual Model (Normalized)')
            axes[1].set_xlabel('Predicted Label')
            axes[1].set_ylabel('') # Hide y-label for clarity
        else:
            axes[1].text(0.5, 0.5, 'Contextual CM not available\nor is empty', ha='center', va='center')
            axes[1].set_title('Best Contextual Model')
    else:
        axes[1].text(0.5, 0.5, 'Contextual Run Not Found', ha='center', va='center')
        axes[1].set_title('Best Contextual Model')


    fig.suptitle('Confusion Matrix Comparison', fontsize=16)
    output_path = os.path.join(output_dir, 'confusion_matrices_comparison.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix comparison plot to {output_path}")

def plot_f1_lift(baseline_run, contextual_run, output_dir):
    """Plots the per-class F1-score lift from baseline to contextual."""
    if baseline_run is None or contextual_run is None:
        print("Warning: Missing baseline or contextual run. Skipping F1-lift plot.")
        return

    base_report = baseline_run.get('class_report', {})
    context_report = contextual_run.get('class_report', {})

    if not base_report or not context_report:
        print("Warning: Missing classification reports. Skipping F1-lift plot.")
        return

    # Extract per-class F1 scores
    classes = sorted(list(set(base_report.keys()) & set(context_report.keys()) - {'accuracy', 'macro avg', 'weighted avg'}))
    if not classes:
        print("Warning: No common classes found between reports for F1-lift plot.")
        return
        
    base_f1s = [base_report[c]['f1-score'] for c in classes]
    context_f1s = [context_report[c]['f1-score'] for c in classes]
    
    lift_df = pd.DataFrame({'class': classes, 'baseline': base_f1s, 'contextual': context_f1s})
    lift_df['lift'] = lift_df['contextual'] - lift_df['baseline']
    lift_df = lift_df.sort_values('lift', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=lift_df, x='lift', y='class', orient='h', palette='viridis')
    
    plt.title('Per-Class F1-Score Lift (Contextual vs. Baseline)')
    plt.xlabel('F1-Score Improvement')
    plt.ylabel('Class')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    output_path = os.path.join(output_dir, 'f1_score_lift.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved F1-score lift plot to {output_path}")

def plot_training_journey(output_dir):
    """Creates a 'spaghetti plot' of different experimental runs."""
    plt.figure(figsize=(10, 6))
    
    for run_name, data in TRAINING_JOURNEY_DATA.items():
        plt.plot(data['epoch'], data['val_f1'], marker='o', linestyle='-', label=run_name, alpha=0.7)

    plt.title('Training Journey: Validation F1-Score Across Runs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation F1-Score')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    output_path = os.path.join(output_dir, 'training_journey.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved training journey plot to {output_path}")

def main():
    """Main function to orchestrate the analysis and plotting."""
    reports_dir = os.path.join('outputs', 'results')
    output_dir = os.path.join('outputs', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Load and process all reports
    df = load_all_reports(reports_dir)
    
    # Drop rows where key metrics are missing
    df.dropna(subset=['test_f1'], inplace=True)

    if df.empty:
        print("Error: No valid reports found after cleaning. Exiting.")
        return

    # Identify best baseline and contextual runs
    baseline_reports = df[df['mode'] == 'BASELINE_PRE']
    contextual_reports = df[df['mode'] == 'CONTEXTUAL']

    best_baseline_run = None
    if baseline_reports.empty:
        print("Warning: Could not find any 'BASELINE_PRE' reports after processing.")
    else:
        best_baseline_run = baseline_reports.nlargest(1, 'test_f1').iloc[0]
        print(f"Best Baseline Run: {best_baseline_run['experiment_name']} (F1: {best_baseline_run['test_f1']:.4f})")

    best_contextual_run = None
    if contextual_reports.empty:
        print("Warning: Could not find any 'CONTEXTUAL' reports after processing.")
    else:
        best_contextual_run = contextual_reports.nlargest(1, 'test_f1').iloc[0]
        print(f"Best Contextual Run: {best_contextual_run['experiment_name']} (F1: {best_contextual_run['test_f1']:.4f})")

    # --- Generate Plots ---
    plot_performance_vs_complexity(df, output_dir)
    plot_confusion_matrices(best_baseline_run, best_contextual_run, output_dir)
    plot_f1_lift(best_baseline_run, best_contextual_run, output_dir)
    plot_training_journey(output_dir)

    print("\n--- Analysis Complete ---")
    print(f"All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()