# Importing all important libraries
import os  # To work with file paths and directories

# librosa is used load and process audio files
import librosa  
import numpy as np 

# soundfile can save or load audio files in different formats
import soundfile as sf  
import matplotlib.pyplot as plt 

"""importing functions to split data into testing,training set 
& to convert labels into numbers and to do one hot encoding"""
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.utils import to_categorical


""" librosa.load() is used to load the audio file and convert it to the target sample rate
we get audio which is a numpy array that has audio data and sr is the sample rate of the audio"""
def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio, sr

"""rms finds Root Mean Square value,scaling factor is used to find the target dBFS 
and we normalize the audio by mutiplying it with this scaling factor """
def normalize_volume(audio, target_dBFS=-20):
    rms = np.sqrt(np.mean(audio**2))
    scaling_factor = 10 ** (target_dBFS / 20) / rms
    normalized_audio = audio * scaling_factor
    return normalized_audio


""" now we find the Mel-frequency cepstral coefficients (MFCCs)-that represent the 
characteristics of sound. librosa.feature.mfcc() is used to extract MFCC featureso"""
def extract_mfcc(audio, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    # 
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.T 


"""we plot the waveform using librosa.display.waveshow() and we save the visualization as an image file"""
def visualize_audio(audio, sr, mfcc, file_name):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfcc.T, sr=sr, x_axis='time', hop_length=512)
    plt.title("MFCCs")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    
    plt.savefig(file_name)
    plt.close()


def preprocess_dataset(dataset_path, output_path, n_mfcc=13, test_size=0.2, random_state=42):
    features = []
    labels = []

    for root, dirs, files in os.walk(dataset_path): #to iterate through all files
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio, sr = load_audio(file_path)
                audio = normalize_volume(audio)
                mfcc = extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc)
                features.append(mfcc)
                labels.append(os.path.basename(root)) 

                if len(features) <= 5:
                    visualize_audio(audio, sr, mfcc, os.path.join(output_path, f"sample_{len(features)}.png"))

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    print("Preprocessing complete! Visualizations saved to:", output_path)

#we change the dataset_path,output_path based on where we are running the code 
dataset_path = "speech_commands" 

os.makedirs(output_path, exist_ok=True)

preprocess_dataset(dataset_path, output_path)