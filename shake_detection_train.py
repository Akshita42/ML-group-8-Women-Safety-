import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from CNN_LSTM import CNN_LSTM  
from preprocessing.data_processing import load_and_preprocess_data  
from sklearn.model_selection import KFold
import torch.optim as optim
from confusion_matrix import confusion_matrix
torch.manual_seed(42)
np.random.seed(42)

train_loader, test_loader = load_and_preprocess_data(r"C:\Users\Eshita\OneDrive\Desktop\gesture_detection ml\feature_eng\processed_data.csv")

# Initialize Model
model = CNN_LSTM()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

def train_model(model, train_loader, test_loader, epochs=10, threshold=0.5):
    train_losses = []
    test_losses = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 Regularization added here

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()  # Ensure shape [batch_size]
            y_batch = y_batch.view(-1).float()  # Ensure shape [batch_size]
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        #  Test Loss and Metrics Calculation
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch).squeeze()
                y_batch = y_batch.view(-1).float()
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()

                # Convert logits to binary predictions
                preds = (outputs > threshold).int()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.tolist())
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        #  Calculate Accuracy, Precision, Recall, F1-Score
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f} - "
              f"Accuracy: {acc:.4f} - Precision: {prec:.4f} - Recall: {rec:.4f} - F1: {f1:.4f}")
    
    
    #  Classification Report
    print("\n Classification Report:\n", classification_report(all_labels, all_preds))
    
    # Save Model
    torch.save(model.state_dict(), "shake_detection_model.pth")
    print("\n Model Trained & Saved!")
    
    # Plot Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", marker="o", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs. Testing Loss")
    plt.grid(True)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, label="Accuracy", marker="o", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Precision, Recall, F1-Score
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(precision_list) + 1), precision_list, label="Precision", marker="o", color="blue")
    plt.plot(range(1, len(recall_list) + 1), recall_list, label="Recall", marker="o", color="red")
    plt.plot(range(1, len(f1_list) + 1), f1_list, label="F1 Score", marker="o", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Precision, Recall, and F1-Score Over Epochs")
    plt.grid(True)
    plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=100)  # 5 folds

def train_kfold(model, train_loader, epochs=10, threshold=0.5):
    fold_no = 1
    for train_idx, val_idx in kf.split(train_loader.dataset):
        print(f"\n Fold {fold_no} Training...")

        train_subset = torch.utils.data.Subset(train_loader.dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_loader.dataset, val_idx)

        train_fold_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_fold_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

        # Train Model on Current Fold
        train_model(model, train_fold_loader, val_fold_loader, epochs, threshold)

        fold_no += 1

# Train Model with K-Fold
train_kfold(model, train_loader, epochs=10, threshold=0.4)
# Train Model
train_model(model, train_loader, test_loader, epochs=10, threshold=0.4)
