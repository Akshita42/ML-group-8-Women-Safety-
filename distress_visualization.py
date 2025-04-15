import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "E:/distressdetection/distress_processed_data.csv"
df = pd.read_csv(DATA_PATH)

sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.countplot(x='label', hue='label', data=df, palette=["#66c2a5", "#fc8d62"], legend=False)

plt.xticks([0, 1], ['Non-Distress (0)', 'Distress (1)'])
plt.xlabel("Emotion Category")
plt.ylabel("Number of Samples")
plt.title("🔹 Emotion Distribution in Dataset")
plt.tight_layout()
plt.show()

sample_file = df.iloc[0]['file_path']
y, sr = librosa.load(sample_file, sr=None)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.7)
plt.title("Waveform of Sample Audio")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", cmap="magma")
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (Log Scale) of Sample Audio")
plt.tight_layout()
plt.show()

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
sns.heatmap(mfcc, cmap="coolwarm", xticklabels=False, yticklabels=range(1, 14))
plt.title("MFCC Heatmap (13 Coefficients)")
plt.xlabel("Time Frame")
plt.ylabel("MFCC Coefficient")
plt.tight_layout()
plt.show()

mfcc_columns = [col for col in df.columns if "mfcc_" in col]
melted_df = df.melt(id_vars=["label"], value_vars=mfcc_columns, 
                    var_name="MFCC", value_name="Coefficient Value")

plt.figure(figsize=(12, 6))
sns.boxplot(x="MFCC", y="Coefficient Value", hue="label", data=melted_df,
            palette=["#66c2a5", "#fc8d62"])
plt.xticks(rotation=45)
plt.title("MFCC Feature Distribution by Class")
plt.xlabel("MFCC Coefficient")
plt.ylabel("Coefficient Value")
plt.legend(title="Emotion", labels=["Non-Distress", "Distress"])
plt.tight_layout()
plt.show()

print("All visualizations rendered successfully!")
