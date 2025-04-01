import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar

#dataset path
DATASET_PATH = "E:/distressdetection/dataset"

# Function to extract MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCCs
        mfcc = np.mean(mfcc, axis=1)  # Take mean over time
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


data = []
for actor_folder in tqdm(os.listdir(DATASET_PATH), desc="Processing Actors"):
    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):  # Skip non-folder files
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav"):  # Only process audio files
            file_path = os.path.join(actor_path, file)
            features = extract_features(file_path)
            if features is not None:
                label = int(file.split("-")[2])  # Extract emotion label from filename
                data.append([file_path, *features, label])

columns = ["file_path"] + [f"mfcc_{i}" for i in range(13)] + ["label"]
df = pd.DataFrame(data, columns=columns)

df.to_csv("E:/distressdetection/distress_processed_data.csv", index=False)
print("Data saved to distress_processed_data.csv")
