import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import classification_report, accuracy_score


# Load dataset
file_path = "csic_database.csv"
df = pd.read_csv(file_path)

# Combine text features
df["text"] = (
    df["Method"].fillna("") + " " +
    df["URL"].fillna("") + " " +
    df["content"].fillna("") + " " +
    df["User-Agent"].fillna("")
)


# Global TF-IDF vectorizer (fit once on full dataset text)
vectorizer = TfidfVectorizer(max_features=10000)
vectorizer.fit(df["text"])



print("\nBinary Classification (Normal vs Attack)")

X = df["text"]
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm_model = SVC(kernel="linear", probability=True, random_state=42)
svm_model.fit(X_train_vec, y_train)

y_pred = svm_model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred) * 100
print(f"Test Accuracy: {acc:.2f}%")
print(classification_report(y_test, y_pred))




print("\nOne-Class SVM for Anomaly Detection")

normal_indices = np.where(y_train == 0)[0]  # 0 = Normal
X_norm = X_train_vec[normal_indices]

# Train final One-Class SVM with best nu
best_nu = 0.1
ocsvm = OneClassSVM(gamma='auto', nu=best_nu)
ocsvm.fit(X_norm)


preds_ocsvm = ocsvm.predict(X_test_vec)  # X_test_vec is the same vectorizer used on X_test
anoms_svm = np.where(preds_ocsvm == -1, 1, 0)  # -1 → anomaly → 1


acc = accuracy_score(y_test, anoms_svm)
print(f"\nFinal One-Class SVM Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, anoms_svm))


with open("vectorizers/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/classifier.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("models/anomaly_detector.pkl", "wb") as f:
    pickle.dump(ocsvm, f)


print("\nAll models saved: vectorizer.pkl, classifier.pkl, anomaly_detector.pkl")
