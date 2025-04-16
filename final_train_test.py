import os
import torch # for deep learning
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn # nn is neural network module
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# to fix randomness in pytorch and numpy
torch.manual_seed(42)
np.random.seed(42)

X_train = np.load("X_train_natural.npy")
y_train = np.load("y_train_natural.npy")
X_test = np.load("X_test_natural.npy")
y_test = np.load("y_test_natural.npy")

# Calculating class weights for minority shake class importance
class_weights = torch.tensor([1.0, len(y_train) / np.sum(y_train)])  

# converting numpy arrays to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

from CNN_LSTM import CNN_LSTM
model = CNN_LSTM()
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1]) # to handle class imbalance here more emphasis on the shake class part
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

def train_model(model, train_loader, test_loader, epochs=10):
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        all_probs = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch).squeeze()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.4).int()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.squeeze().tolist())
                all_probs.extend(probs.tolist())

        # Threshold tuning
        thresholds = [0.3, 0.4, 0.5]
        best_threshold = 0.4
        best_recall = recall_score(all_labels, all_preds)
        for th in thresholds:
            preds_th = (torch.tensor(all_probs) > th).int().tolist() 
            recall = recall_score(all_labels, preds_th)
            if recall > best_recall:
                best_recall = recall
                best_threshold = th

        # Final evaluation with best threshold
        preds_final = (torch.tensor(all_probs) > best_threshold).int().tolist()
        acc = accuracy_score(all_labels, preds_final)
        prec = precision_score(all_labels, preds_final, zero_division=0)
        rec = recall_score(all_labels, preds_final, zero_division=0)
        f1 = f1_score(all_labels, preds_final, zero_division=0)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Acc: {acc:.4f} - Prec: {prec:.4f} - Rec: {rec:.4f} - F1: {f1:.4f} - Best Threshold: {best_threshold}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "shake_detection_model.pth")
            print("New best model saved!")

    torch.save(model.state_dict(), "shake_detection_model.pth")
    print("\nModel Trained & Saved as 'shake_detection_model.pth'")
    print("\nClassification Report:\n", classification_report(all_labels, preds_final))

    # Add confusion matrix plotting
    cm = confusion_matrix(all_labels, preds_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Shake", "Shake"], yticklabels=["No Shake", "Shake"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"\nConfusion Matrix Values: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"FAR: {far:.4f}")
        print(f"FRR: {frr:.4f}")

if __name__ == "__main__":
    train_model(model, train_loader, test_loader, epochs=10)