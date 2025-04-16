import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names):
    # Directly use y_true and y_pred without argmax
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Also calculate and print FAR and FRR
    if cm.shape == (2, 2):  # Only defined for binary classification
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0

        print(f"\n Confusion Matrix Values: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f" False Acceptance Rate (FAR): {far:.4f}")
        print(f" False Rejection Rate (FRR): {frr:.4f}")
    else:
        print("FAR/FRR only supported for binary classification.")
