import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv("E:/distressdetection/distress_processed_data.csv")  # Ensure labels are 0 and 1

X = df.iloc[:, 1:-1].values  
y = df.iloc[:, -1].values  

sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.show()

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, 'distress_scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10, 50],
    'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
    'kernel': ['rbf']
}

svm = SVC(probability=True, random_state=42)

grid = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

best_svm_model = grid.best_estimator_

joblib.dump(best_svm_model, 'distress_model.pkl')

probs = best_svm_model.predict_proba(X_test)[:, 1]

threshold = 0.60
y_pred_thresholded = (probs > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_thresholded)
print(f"Tuned SVM Accuracy (Threshold={threshold}): {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_thresholded))

conf_matrix = confusion_matrix(y_test, y_pred_thresholded)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples", xticklabels=["Normal", "Distress"], yticklabels=["Normal", "Distress"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"SVM Confusion Matrix (Threshold={threshold})")
plt.show()
TN, FP, FN, TP = conf_matrix.ravel()

FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f" False Acceptance Rate (FAR): {FAR:.4f}")
print(f" False Rejection Rate (FRR): {FRR:.4f}")

