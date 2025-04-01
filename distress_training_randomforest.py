import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Handles class imbalance
from scipy.stats import randint

# Load preprocessed data
df = pd.read_csv("E:/distressdetection/distress_processed_data.csv")

X = df.iloc[:, 1:-1].values  # MFCC features
y = df.iloc[:, -1].values  # Emotion labels

#class distribution
sns.countplot(x=y)
plt.title("Class Distribution Before Balancing")
plt.show()

# removing class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#hyperparameters
param_dist = {
    "n_estimators": randint(100, 300),  # Trees between 100 and 300
    "max_depth": [10, 20, None],  # Tree depth
    "min_samples_split": [2, 5, 10],  # Minimum samples to split
    "min_samples_leaf": [1, 2, 4],  # Minimum leaf nodes
    "max_features": ["sqrt"],  # Fixed to sqrt for speed
}

#Randomized Search with progress bar
with tqdm(total=40, desc="🔍 Randomized Search Progress") as pbar:
    class TQDMRandomSearchCV(RandomizedSearchCV):
        def _run_search(self, evaluate_candidates):
            def wrapped_eval(candidates):
                pbar.update(len(candidates))
                return evaluate_candidates(candidates)
            super()._run_search(wrapped_eval)

    rf = RandomForestClassifier(random_state=42)
    random_search = TQDMRandomSearchCV(
        rf, param_dist, n_iter=40, cv=3, scoring="accuracy", n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.4f}")
print(" Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
#confusion matrix, distribution before normalization and results have been uplaoded              
