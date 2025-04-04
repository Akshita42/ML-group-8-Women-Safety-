import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from CNN_LSTM import CNN_LSTM  
from sklearn.model_selection import KFold

# Load Processed Data
df = pd.read_csv(r"C:\Users\Eshita\OneDrive\Desktop\gesture_detection ml\feature_eng\processed_data.csv")
features = ["Mean_Mag", "Std_Mag", "Max_Mag", "Min_Mag"]
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df["Shake"].astype(int).values

# Convert to Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create Dataset
dataset = TensorDataset(X_tensor, y_tensor)

# KFold Setup
kf = KFold(n_splits=5, shuffle=True, random_state=100)  # 5 folds

# Model Loading
model = CNN_LSTM()
model.load_state_dict(torch.load("shake_detection_model.pth"))
model.eval()

# Accuracy Calculation Function
def calculate_accuracy(loader):
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total * 100

# K-Fold Training and Testing
def train_kfold():
    fold_no = 1
    for train_idx, val_idx in kf.split(dataset):
        print(f"\n Fold {fold_no} Training...")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Train Model on Current Fold
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            y_batch = y_batch.view(-1).float()
            loss = torch.nn.BCELoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Test the Model on Validation Set
        model.eval()
        val_acc = calculate_accuracy(val_loader)
        print(f"Fold {fold_no} Validation Accuracy: {val_acc:.2f}%")
        fold_no += 1

# Run K-Fold Cross-Validation
train_kfold()

# Finally, calculate test accuracy after K-Fold (if needed for final test set)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
test_acc = calculate_accuracy(test_loader)
print(f"Final Test Accuracy: {test_acc:.2f}%")
