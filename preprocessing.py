import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt # scipy for signal processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib # for saving trained models 
from scipy.fft import fft

# Load dataset
df = pd.read_csv(r"C:\Users\Eshita\OneDrive\Desktop\gesture_detection ml zip file\gesture_detection ml\dataset\merged_activity_data.csv")
df.columns = ["Time", "Seconds_Elapsed", "Z", "Y", "X", "Activity", "Shake"]

# Calculate magnitude
df["Magnitude"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)

# Lowpass filter to smooth data
def butter_lowpass_filter(data, cutoff=3, fs=50, order=3):
    """Apply a lowpass Butterworth filter to remove high-frequency noise."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

df["Filtered_Mag"] = butter_lowpass_filter(df["Magnitude"].ffill())

# Rolling window features with increased window size
window_size = 75  # Adjusted to ~1.5 seconds at 50 Hz for three shakes
df["Mean_Mag"] = df["Filtered_Mag"].rolling(window_size).mean()
df["Std_Mag"] = df["Filtered_Mag"].rolling(window_size).std()
df["Max_Mag"] = df["Filtered_Mag"].rolling(window_size).max()
df["Min_Mag"] = df["Filtered_Mag"].rolling(window_size).min()

# Add peak count and energy features
def count_peaks(mag_data, threshold=12):  # Lowered threshold for sensitivity
    """Count peaks above a threshold in a window."""
    diff = np.diff(mag_data > threshold, prepend=False)
    peaks = np.where(diff)[0]
    return len(peaks) if len(peaks) > 0 else 0

def calculate_energy(mag_data):
    """Calculate energy (sum of squared magnitudes) in a window."""
    return np.sum(mag_data**2)

df["Peak_Count"] = df["Filtered_Mag"].rolling(window_size).apply(lambda x: count_peaks(x), raw=True)
df["Energy"] = df["Filtered_Mag"].rolling(window_size).apply(calculate_energy, raw=True)

# Calculating jerk to see how quickly acceleration changes
df["Jerk"] = np.abs(np.gradient(df["Filtered_Mag"], df["Seconds_Elapsed"]))

# Add FFT frequency feature to convert time based into frequency
def calculate_dominant_freq(mag_data):
    n = len(mag_data)
    yf = fft(mag_data)
    freqs = np.fft.fftfreq(n) * 50  # 50 Hz sampling rate
    idx = np.argmax(np.abs(yf[1:n//2]))  # Avoid DC component
    return freqs[idx] if idx else 0

df["Dominant_Freq"] = df["Filtered_Mag"].rolling(window_size).apply(calculate_dominant_freq, raw=True)

# Drop NaN from rolling
df.dropna(inplace=True)

# Shuffle to mix True/False shake labels
df = shuffle(df, random_state=42).reset_index(drop=True)

# Feature matrix and labels
X = df[["Mean_Mag", "Std_Mag", "Max_Mag", "Min_Mag", "Peak_Count", "Energy", "Jerk", "Dominant_Freq"]]
y = df["Shake"].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for real-time use
joblib.dump(scaler, "scaler.joblib")

# Train-test split with stratify to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save preprocessed full data
df.to_csv("processed_data.csv", index=False)
print("✅ Preprocessing Done! Data saved as `processed_data.csv`.")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Label distribution (train): {np.bincount(y_train)}")
print(f"Label distribution (test): {np.bincount(y_test)}")

# Save the training and test data as .npy files with _natural suffix
np.save("X_train_natural.npy", X_train)
np.save("y_train_natural.npy", y_train)
np.save("X_test_natural.npy", X_test)
np.save("y_test_natural.npy", y_test)

print("📦 Training and test data saved as .npy files with _natural suffix!")
print("✅ Scaler saved as `scaler.joblib` for real-time use.")