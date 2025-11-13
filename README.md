# ERIC-PROJECT: A Context-Aware Multimodal Framework for Emotion Recognition

This project is a deep learning pipeline to prove that conversational **context** is the most critical factor in Emotion Recognition in Conversations (ERC).

It compares two models in a "fair fight":
1.  **Baseline:** A context-unaware model that classifies each utterance individually.
2.  **Contextual:** A Transformer-based model that analyzes the entire dialogue history.

This repository contains the full pipeline, from feature extraction to training, analysis, and an interactive Flask demo.

## 🚀 Final Results

Our core hypothesis was proven correct. The `contextual` model outperformed the `baseline` by **+75.5%**, demonstrating that context is not just helpful, but essential.

| Model | Test F1-Score | Test Accuracy | Training Time |
| :--- | :--- | :--- | :--- |
| **Baseline (No Context)** | 31.3% | 48.1% | 1.15 Mins |
| **Contextual (Winner)** | **54.9%** | **56.9%** | 3.13 Mins |

Further experiments showed that adding SOTA features like Speaker Embeddings or Focal Loss led to overfitting on this dataset, making our simple `ContextualTransformer` the most robust and high-performing solution.

![Per-Class F1 Lift](outputs/results/f1_lift_chart.png)
*(Run visualize_results.py to generate this)*

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/ERIC-PROJECT.git](https://github.com/YourUsername/ERIC-PROJECT.git)
    cd ERIC-PROJECT
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (or venv\Scripts\activate on Windows)
    ```

3.  **Install requirements:**
    *(Ensure you have PyTorch with CUDA support installed first)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the MELD Dataset:**
    * Download the raw MELD dataset.
    * Place the `MELD.Raw` folder inside the project directory.

## 🏃‍♂️ How to Run

### Step 1: Feature Extraction
This must be run once. It will process all 13,000+ MELD clips and create the `.npy` feature files in `outputs/features/`.

```bash
python extract_features.py
```

### Step 2: Run the "Fair Fight"
Run these two commands to reproduce our main findings.

**1. Run the (failed) Baseline:**
```bash
python main.py --mode baseline_pre --epochs 20 --batch_size 16 --lr 1e-5
```

**2. Run the (winning) Contextual Model:**
```bash
python main.py --mode contextual --epochs 20 --batch_size 16 --lr 5e-5
```

### Step 3: View Results
Run the analysis script to generate all comparison graphs (like the one above) in `outputs/results/`.

```bash
python visualize_results.py
```

### Step 4: Run the Interactive Demo
This starts a Flask web server to demo the best model.

```bash
python app.py
```
Open your browser to `http://127.0.0.1:5000` to try it live.