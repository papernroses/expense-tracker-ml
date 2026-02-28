import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from scipy.sparse import hstack

# ---------------- LOAD DATA ----------------
df = pd.read_csv("expense_data_1.csv")
df = df[df["Income/Expense"].str.lower() == "expense"]
df = df[["Note", "Amount", "Category"]].dropna()

# ---------------- MERGE RARE CLASSES ----------------
min_samples = 3  # safe threshold
category_counts = df["Category"].value_counts()

rare_categories = category_counts[category_counts < min_samples].index
df["Category"] = df["Category"].replace(rare_categories, "Other")

print("✅ Rare categories merged into 'Other'")

# ---------------- FEATURES ----------------
tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
X_text = tfidf.fit_transform(df["Note"])

scaler = StandardScaler(with_mean=False)
X_amount = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

X = hstack([X_text, X_amount])
y = df["Category"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=3000, class_weight="balanced")
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- METRICS (WARNING FREE) ----------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n📊 MODEL PERFORMANCE (WARNING-FREE)")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Expense Category Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "expense_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "amount_scaler.pkl")

print("\n✅ Model & tools saved successfully (NO WARNINGS)")