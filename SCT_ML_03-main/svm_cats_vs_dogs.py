import os
import zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# ===============================
# 1. Extract Dataset
# ===============================
zip_path = "/mnt/data/2661816b-4180-4689-9f4a-052017a8fb34.zip"
extract_dir = "/mnt/data/cats_dogs_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Dataset extracted successfully.")

# ===============================
# 2. Load & Preprocess Images
# ===============================
IMG_SIZE = 64  # smaller size for faster processing
X, y = [], []

for label, category in enumerate(["cats", "dogs"]):
    folder = os.path.join(extract_dir, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label)
        except Exception as e:
            pass  # skip unreadable images

X = np.array(X, dtype="float32") / 255.0  # normalize
y = np.array(y)

print(f"✅ Loaded {len(X)} images.")

# Flatten images for SVM (convert to 1D vector)
X_flat = X.reshape(len(X), -1)

# ===============================
# 3. Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features for SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 4. Train SVM Classifier
# ===============================
print("🔄 Training SVM (this may take a few minutes)...")
svm = SVC(kernel='rbf', C=1, gamma='scale')  # RBF kernel for non-linear classification
svm.fit(X_train, y_train)

# ===============================
# 5. Evaluate Model
# ===============================
y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}\n")

print("🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
