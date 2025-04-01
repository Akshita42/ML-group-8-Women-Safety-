import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset Load
df = pd.read_csv(r"C:\Users\Eshita\OneDrive\Desktop\gesture_detection ml\dataset\merged_activity_data.csv")

df.columns = ["Time", "Seconds_Elapsed", "Z", "Y", "X", "Activity", "Shake"]

# Magnitude Compute
df["Magnitude"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)

# Low-Pass Filter (Noise Removal)
def butter_lowpass_filter(data, cutoff=3, fs=50, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

df["Filtered_Mag"] = butter_lowpass_filter(df["Magnitude"].ffill())



# Rolling Window Features (2s ka window)
window_size = 20  # Adjust based on sampling rate
df["Mean_Mag"] = df["Filtered_Mag"].rolling(window_size).mean()
df["Std_Mag"] = df["Filtered_Mag"].rolling(window_size).std()
df["Max_Mag"] = df["Filtered_Mag"].rolling(window_size).max()
df["Min_Mag"] = df["Filtered_Mag"].rolling(window_size).min()

# Drop NaN values
df.dropna(inplace=True)

# Features & Labels Split
X = df[["Mean_Mag", "Std_Mag", "Max_Mag", "Min_Mag"]]
y = df["Shake"].astype(int)  # Convert True/False → 1/0

# Feature Scaling (Normalize the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80-20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Preprocessed Data Save
df.to_csv("processed_data.csv", index=False)
print("Preprocessing Done! Data saved as `processed_data.csv`.")
