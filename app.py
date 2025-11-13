import os
import subprocess
import json
from flask import Flask, render_template, jsonify, request, Response, stream_with_context, send_from_directory
import pandas as pd
import torch
import numpy as np
import re
from models import ContextualTransformer # Changed import
import time
import sys

# Add project root to Python path
app = Flask(__name__)

# --- Simple Config for Model ---
class SimpleConfig:
    def __init__(self):
        self.NUM_EMOTIONS = 7
        self.DROPOUT = 0.0

# --- Configuration --
VIDEO_LIST_PATH = 'MELD-RAW/MELD.Raw/test_sent_emo.csv' # Use the test set
FEATURES_DIR = 'outputs/features'
MODEL_PATH = 'outputs/models/contextual_late_best_model.pt'
VISUALIZATIONS_DIR = 'outputs/visualizations'
RESULTS_DIR = 'outputs/results'
VIDEO_DIR = 'MELD-RAW/MELD.Raw/test/output_repeated_splits_test' # Correct video directory for test set


# --- Feature Loading --
# Global feature dictionaries
audio_features = {}
visual_features = {}
text_features = {}
video_id_to_index = {}

def load_features():
    """Loads all features into memory at `startup`."""
    global audio_features, visual_features, text_features
    
    try:
        # Load the feature files, which are dictionaries stored in 0-d arrays
        audio_features = np.load(os.path.join(FEATURES_DIR, 'test_audio_features_wav2vec.npy'), allow_pickle=True).item()
        visual_features = np.load(os.path.join(FEATURES_DIR, 'test_visual_features_resnet.npy'), allow_pickle=True).item()
        text_features = np.load(os.path.join(FEATURES_DIR, 'test_text_features_roberta.npy'), allow_pickle=True).item()
            
        print("All features loaded successfully.")

    except Exception as e:
        print(f"Error loading features: {e}")


# --- Model Loading --
# Define the mapping from output index to emotion label
EMOTION_LABELS = {
    0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry', 
    4: 'fear', 5: 'disgust', 6: 'surprise'
}

# Load the model at startup
try:
    model = ContextualTransformer(SimpleConfig()) # Use ContextualTransformer
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def get_video_list():
    """Reads the video list from the CSV and constructs video IDs."""
    if os.path.exists(VIDEO_LIST_PATH):
        df = pd.read_csv(VIDEO_LIST_PATH)
        # Construct video_id from Dialogue_ID and Utterance_ID without extension
        return [f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}" for index, row in df.iterrows()]
    return []

# --- Routes --

@app.route('/')
def index():
    """Renders the Demo page."""
    videos = get_video_list()
    return render_template('index.html', videos=videos)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/video/<path:filename>')
def video(filename):
    """Serves video files by trying different filename variations."""
    possible_filenames = [
        filename,
        f"._{filename}",
        f"final_videos_test{filename}"
    ]
    
    for fname in possible_filenames:
        # Check if the file exists in the directory
        if os.path.exists(os.path.join(VIDEO_DIR, fname)):
            return send_from_directory(VIDEO_DIR, fname)
            
    # If no file is found after trying all formats, return 404
    return "File not found", 404

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """Serves files from the outputs directory."""
    # Note: The 'outputs' directory is at the root of the project.
    # We construct the full path from the project root.
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    return send_from_directory(directory, filename)

@app.route('/admin')
def admin():
    """Renders the Admin Dashboard page."""
    return render_template('admin.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request."""
    data = request.get_json()
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({'error': 'No video_id provided'}), 400

    try:
        # --- START: Key Transformation Logic ---
        # The frontend sends 'diaD_uttU', but features are keyed by (D, U) tuple.
        match = re.match(r'dia(\d+)_utt(\d+)', video_id)
        if not match:
            return jsonify({'error': f'Invalid video_id format: {video_id}'}), 400
        
        dialogue_id = int(match.group(1))
        utterance_id = int(match.group(2))
        feature_key = (dialogue_id, utterance_id)
        # --- END: Key Transformation Logic ---

        # Get features using the correct tuple key
        audio_ft = audio_features.get(feature_key)
        visual_ft = visual_features.get(feature_key)
        text_ft = text_features.get(feature_key) # This is just the numpy array

        if audio_ft is None or visual_ft is None or text_ft is None:
            # Log the key that was not found for debugging
            print(f"Features not found for video_id '{video_id}' with key {feature_key}")
            return jsonify({'error': f'Features not found for {video_id}'}), 404

        # Convert features to tensors, adding batch and sequence dimensions
        # Shape: [batch_size, seq_len, feature_dim] -> [1, 1, D]
        audio_tensor = torch.from_numpy(audio_ft).float().unsqueeze(0).unsqueeze(0)
        visual_tensor = torch.from_numpy(visual_ft).float().unsqueeze(0).unsqueeze(0)
        text_tensor = torch.from_numpy(text_ft).float().unsqueeze(0).unsqueeze(0)

        # Create an attention mask for a single item (no padding)
        # Shape: [batch_size, seq_len] -> [1, 1]
        attention_mask_tensor = torch.tensor([[True]], dtype=torch.bool)


        # Get model prediction
        with torch.no_grad():
            output = model(
                text_features=text_tensor, 
                audio_features=audio_tensor, 
                visual_features=visual_tensor,
                attention_mask=attention_mask_tensor
            )
            # Output shape is [batch, seq, num_emotions], so [1, 1, 7]
            prediction = torch.argmax(output, dim=-1).squeeze().item()
            emotion = EMOTION_LABELS[prediction]

        return jsonify({'prediction': emotion})

    except Exception as e:
        print(f"Error getting prediction: {e}")
        return jsonify({'error': f'Error getting prediction: {e}'}), 500


@app.route('/run-script/<script_name>')
def run_script(script_name):
    
    def generate_output():
        # Mapping from script name to script file path
        script_map = {
            "feature-extraction": "extract_features.py",
            "training-contextual": "trainer.py", # Corrected from train_contextual.py
            "training-baseline": "main.py",
        }

        if script_name == "test-stream":
            print("--- Running Stream Test ---")
            try:
                # This test sends pre-formatted SSE messages to isolate frontend issues.
                print("--- Yielding: Line 1 ---")
                yield "data: " + json.dumps("Line 1: Some normal output.\n") + "\n\n"
                time.sleep(1)
                
                print("--- Yielding: Line 2 (CR) ---")
                yield "data: " + json.dumps("Line 2: with a carriage return.\r") + "\n\n"
                time.sleep(1)

                print("--- Yielding: Progress 25% ---")
                yield "data: " + json.dumps("Progress: 25%") + "\n\n"
                time.sleep(1)

                print("--- Yielding: Progress 50% (CR) ---")
                yield "data: " + json.dumps("Progress: 50%\r") + "\n\n"
                time.sleep(1)
                
                print("--- Yielding: Progress 75% ---")
                yield "data: " + json.dumps("Progress: 75%") + "\n\n"
                time.sleep(1)

                print("--- Yielding: Line 3 ---")
                yield "data: " + json.dumps("Line 3: Final line.\n") + "\n\n"
                time.sleep(1)

                print("--- Yielding: Finish ---")
                yield "data: " + json.dumps({'message': 'Script Finished', 'progress': 100}) + "\n\n"

            except Exception as e:
                print(f"--- Error in test-stream: {e} ---")
                error_message = f"--- Error during stream test: {e} ---"
                yield "data: " + json.dumps({'error': error_message}) + "\n\n"
            
        else:
            script_to_run = script_map.get(script_name)
            if not script_to_run:
                print(f"--- Invalid script name: {script_name} ---")
                yield "data: " + json.dumps({'error': f'Invalid script name: {script_name}'}) + "\n\n"
                return

            command = [sys.executable, script_to_run]
            print(f"--- Running command: {' '.join(command)} ---")

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding='utf-8'
                )
            except FileNotFoundError:
                error_message = f"--- Error: Script '{script_to_run}' not found. ---"
                print(error_message)
                yield "data: " + json.dumps({'error': error_message}) + "\n\n"
                return
            except Exception as e:
                error_message = f"--- Error starting subprocess: {e} ---"
                print(error_message)
                yield "data: " + json.dumps({'error': error_message}) + "\n\n"
                return

            buffer = ""
            for char in iter(lambda: process.stdout.read(1), ''):
                if char == '\n':
                    yield "data: " + json.dumps(buffer + '\n') + "\n\n"
                    buffer = ""
                elif char == '\r':
                    yield "data: " + json.dumps(buffer + '\r') + "\n\n"
                    buffer = ""
                else:
                    buffer += char
            
            if buffer:
                yield "data: " + json.dumps(buffer) + "\n\n"

            process.stdout.close()
            return_code = process.wait()
            
            print(f"--- Script {script_name} finished with exit code {return_code} ---")
            yield "data: " + json.dumps({'message': 'Script Finished', 'progress': 100}) + "\n\n"

    return Response(stream_with_context(generate_output()), mimetype='text/event-stream')

@app.route('/get-visualizations')
def get_visualizations():
    """Returns a list of visualization image paths as URLs."""
    images = []
    # Use os.path.join for creating paths to be OS-agnostic
    viz_path = os.path.join('outputs', 'visualizations')
    if os.path.exists(viz_path):
        for f in os.listdir(viz_path):
            if f.endswith('.png'):
                # Create a URL path for the browser
                images.append(f'/outputs/visualizations/{f}')
    
    results_images = []
    res_path = os.path.join('outputs', 'results')
    if os.path.exists(res_path):
        for f in os.listdir(res_path):
            if f.endswith('.png'):
                # Create a URL path for the browser
                results_images.append(f'/outputs/results/{f}')

    return jsonify({'visualizations': images, 'results': results_images})


if __name__ == '__main__':
    load_features() # Load features before running the app
    app.run(debug=True, threaded=True)