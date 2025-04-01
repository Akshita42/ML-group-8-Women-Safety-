import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
DATA_PATH = "E:/distressdetection/distress_processed_data.csv"
df = pd.read_csv(DATA_PATH)

#Emotion Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df['label'], palette="viridis")
plt.xlabel("Emotion Label")
plt.ylabel("Count")
plt.title("Emotion Distribution in Dataset")
plt.show()

sample_file = df.iloc[0]['file_path']
y, sr = librosa.load(sample_file, sr=None)

#Waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform of Sample Audio")
plt.show()

#Spectrogram
plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram of Sample Audio")
plt.show()

#MFCC Heatmap
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
sns.heatmap(mfcc, cmap="coolwarm", xticklabels=False, yticklabels=range(1, 14))
plt.xlabel("Time Frame")
plt.ylabel("MFCC Coefficients")
plt.title("MFCC Heatmap for Sample Audio")
plt.show()

#Boxplot of MFCC Features 
mfcc_columns = [col for col in df.columns if "mfcc_" in col]
melted_df = df.melt(id_vars=["label"], value_vars=mfcc_columns, var_name="MFCC", value_name="Value")

plt.figure(figsize=(12, 6))
sns.boxplot(x="MFCC", y="Value", hue="label", data=melted_df, palette="tab10")
plt.xticks(rotation=90)
plt.title("MFCC Feature Distribution Across Emotions")
plt.show()

print("Visualization complete!")
#all visualization images have been uploaded
