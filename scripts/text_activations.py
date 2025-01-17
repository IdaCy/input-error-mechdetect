import os
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datetime import datetime
import traceback

# Paths
PROCESSED_DATA_PATH = "data/processed/"
RESULTS_PATH = "results_text/"
LOG_FILE = "logs/text_activations.log"

# Model configuration
MODEL_NAME = "bert-base-uncased"

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
except Exception as e:
    with open(LOG_FILE, "a") as log_file:
        log_file.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " +
            f"Model loading failed: {str(e)}\n")
    raise e


# Function to log messages
def log(message):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
        print(message)
    except Exception as e:
        print(f"Logging failed: {str(e)}")


# Function to extract activations
def extract_activations(file_name, run_id):
    try:
        log(f"Processing file: {file_name}, Run: {run_id}")
        file_path = os.path.join(PROCESSED_DATA_PATH, file_name)

        # Read CSV file without header
        df = pd.read_csv(
            file_path, header=None, names=['sentence', 'tokenized'])

        # Extract activations
        activations = []
        for sentence in df['sentence']:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze(
            ).tolist()
            activations.append(last_hidden_state)

        # Save results
        result_file = os.path.join(
            RESULTS_PATH, f"activations_run{run_id}_{file_name}")
        pd.DataFrame(activations).to_csv(result_file, index=False)
        log(f"Activations saved: {result_file}")
    except Exception as e:
        log(f"Error processing file {file_name}, Run {run_id}: {str(e)}")
        log(traceback.format_exc())


# Main
try:
    os.makedirs(RESULTS_PATH, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log("Starting activation extraction...")
    for file_name in os.listdir(PROCESSED_DATA_PATH):
        if file_name.startswith(
                'text-sentences_') and file_name.endswith('.csv'):
            for run_id in range(1, 4):  # Run each prompt 3 times
                extract_activations(file_name, run_id)
    log("Activation extraction completed.")
except Exception as e:
    log(f"Unexpected error: {str(e)}")
    log(traceback.format_exc())
