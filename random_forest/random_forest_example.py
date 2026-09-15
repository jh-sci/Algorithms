"""
Random Forest Example — Loan Default Prediction
Run: python random_forest_example.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Create a small loan dataset ──────────────────────────────
data = {
    "income":      [70, 55, 80, 60, 30, 25, 40, 65, 50, 90, 35, 45, 75, 28, 62, 48, 85, 33, 58, 42],
    "credit_score": [720, 690, 750, 710, 580, 620, 650, 700, 670, 760, 600, 660, 730, 590, 705, 640, 740, 610, 680, 630],
    "debt_ratio":  [25, 35, 20, 45, 55, 60, 50, 30, 40, 15, 52, 48, 22, 58, 33, 44, 18, 56, 38, 51],
    "result":      ["safe","safe","safe","default","default","default","default","safe","safe","safe","default","default","safe","default","safe","default","safe","default","safe","default"]
}

df = pd.DataFrame(data)
print("── Training data ──")
print(df.to_string(index=False))
print()

# ── 2. Split into features (X) and labels (y) ──────────────────
X = df[["income", "credit_score", "debt_ratio"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── 3. Auto-detect the best number of trees ────────────────────
print("── Finding best number of trees ──")
candidates = [5, 10, 25, 50, 100, 200, 300]
best_score = 0
best_n = candidates[0]

for n in candidates:
    trial = RandomForestClassifier(
        n_estimators=n, max_depth=3, oob_score=True, random_state=42
    )
    trial.fit(X_train, y_train)
    score = trial.oob_score_
    marker = ""
    if score > best_score:
        best_score = score
        best_n = n
        marker = " ← best so far"
    print(f"  {n:>4} trees → OOB accuracy: {score:.0%}{marker}")

print(f"\n  Winner: {best_n} trees at {best_score:.0%}\n")

# ── 4. Build the forest with the best n_estimators ─────────────
model = RandomForestClassifier(
    n_estimators=best_n,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# ── 5. See what each tree learned ───────────────────────────────
feature_names = ["income", "credit_score", "debt_ratio"]
print("── What each tree splits on (root question) ──")
for i, tree in enumerate(model.estimators_):
    root_feature = feature_names[tree.tree_.feature[0]]
    root_threshold = round(tree.tree_.threshold[0], 1)
    print(f"  Tree {i+1}: {root_feature} <= {root_threshold}?")
print()

# ── 6. Predict on test set ──────────────────────────────────────
predictions = model.predict(X_test)
print("── Test set results ──")
print(f"  Accuracy: {accuracy_score(y_test, predictions):.0%}")
print()
print(classification_report(y_test, predictions))

# ── 7. Predict a brand new applicant ────────────────────────────
new_applicant = pd.DataFrame([{
    "income": 58,
    "credit_score": 695,
    "debt_ratio": 32
}])

prediction = model.predict(new_applicant)[0]
probabilities = model.predict_proba(new_applicant)[0]
class_labels = model.classes_

print("── New applicant: income=$58k, credit=695, debt=32% ──")
print(f"  Prediction: {prediction}")
for label, prob in zip(class_labels, probabilities):
    print(f"  P({label}) = {prob:.0%}")
print()

# ── 8. Which features mattered most? ───────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_names)
print("── Feature importance ──")
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 30)
    print(f"  {feat:14s} {imp:.2f}  {bar}")
