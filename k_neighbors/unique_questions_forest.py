"""
Unique-Question Forest vs Standard Random Forest
Each tree is forced to split on a different root feature.
Run: python unique_questions_forest.py
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 1. Create the loan dataset ─────────────────────────────────
data = {
    "income":       [70, 55, 80, 60, 30, 25, 40, 65, 50, 90, 35, 45, 75, 28, 62, 48, 85, 33, 58, 42],
    "credit_score": [720, 690, 750, 710, 580, 620, 650, 700, 670, 760, 600, 660, 730, 590, 705, 640, 740, 610, 680, 630],
    "debt_ratio":   [25, 35, 20, 45, 55, 60, 50, 30, 40, 15, 52, 48, 22, 58, 33, 44, 18, 56, 38, 51],
    "result":       ["safe","safe","safe","default","default","default","default","safe","safe","safe",
                     "default","default","safe","default","safe","default","safe","default","safe","default"]
}
df = pd.DataFrame(data)
feature_names = ["income", "credit_score", "debt_ratio"]
X = df[feature_names]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ── 2. Unique-question forest ──────────────────────────────────
# Each tree is forced to use a different feature at the root
# by training it on only that one feature for the first split,
# then letting it use all features for deeper splits.

print("=" * 52)
print("  UNIQUE-QUESTION FOREST")
print("  Each tree must ask a different first question")
print("=" * 52)

unique_trees = []
unique_predictions = []

for i, root_feature in enumerate(feature_names):
    # Train a tree, but rearrange data so the root feature
    # is strongly preferred: train first on just that feature
    # to find the best split, then build the full tree
    # with that constraint

    # Step 1: find best threshold for this feature
    best_gini = 1.0
    best_thresh = None
    values = sorted(X_train[root_feature].unique())
    for j in range(len(values) - 1):
        thresh = (values[j] + values[j + 1]) / 2
        left_mask = X_train[root_feature] <= thresh
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        # weighted gini
        left_y = y_train[left_mask]
        right_y = y_train[right_mask]
        def gini(labels):
            counts = labels.value_counts(normalize=True)
            return 1 - (counts ** 2).sum()
        n = len(y_train)
        weighted = (len(left_y)/n) * gini(left_y) + (len(right_y)/n) * gini(right_y)
        if weighted < best_gini:
            best_gini = weighted
            best_thresh = thresh

    # Step 2: split data at root, train subtrees on each side
    left_mask = X_train[root_feature] <= best_thresh
    right_mask = ~left_mask

    left_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    right_tree = DecisionTreeClassifier(max_depth=2, random_state=42)

    left_tree.fit(X_train[left_mask], y_train[left_mask])
    right_tree.fit(X_train[right_mask], y_train[right_mask])

    # Step 3: predict using this forced-root tree
    def predict_one(row):
        if row[root_feature] <= best_thresh:
            return left_tree.predict(row.to_frame().T)[0]
        else:
            return right_tree.predict(row.to_frame().T)[0]

    preds = X_test.apply(predict_one, axis=1).values
    unique_predictions.append(preds)
    acc = accuracy_score(y_test, preds)

    print(f"\n  Tree {i+1}: {root_feature} <= {best_thresh}?")
    print(f"    Gini after split: {best_gini:.3f}")
    print(f"    Individual accuracy: {acc:.0%}")

    unique_trees.append({
        "feature": root_feature,
        "threshold": best_thresh,
        "left_tree": left_tree,
        "right_tree": right_tree
    })

# Majority vote across the 3 unique trees
unique_votes = np.array(unique_predictions)
final_unique = []
for j in range(len(y_test)):
    votes = list(unique_votes[:, j])
    final_unique.append(max(set(votes), key=votes.count))

unique_acc = accuracy_score(y_test, final_unique)
print(f"\n  Combined (majority vote): {unique_acc:.0%}")

# ── 3. Standard random forest for comparison ───────────────────
from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 52)
print("  STANDARD RANDOM FOREST (3 trees)")
print("  Trees choose freely — may repeat questions")
print("=" * 52)

std_model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
std_model.fit(X_train, y_train)

for i, tree in enumerate(std_model.estimators_):
    root_feat = feature_names[tree.tree_.feature[0]]
    root_thresh = round(tree.tree_.threshold[0], 1)
    preds = std_model.classes_[tree.predict(X_test).astype(int)]
    acc = accuracy_score(y_test, preds)
    print(f"\n  Tree {i+1}: {root_feat} <= {root_thresh}?")
    print(f"    Individual accuracy: {acc:.0%}")

std_preds = std_model.predict(X_test)
std_acc = accuracy_score(y_test, std_preds)
print(f"\n  Combined (majority vote): {std_acc:.0%}")

# ── 4. Side-by-side comparison ─────────────────────────────────
print("\n" + "=" * 52)
print("  COMPARISON")
print("=" * 52)
print(f"\n  Unique questions forest: {unique_acc:.0%}")
print(f"  Standard random forest:  {std_acc:.0%}")

if unique_acc > std_acc:
    print("\n  → Unique questions won!")
elif std_acc > unique_acc:
    print("\n  → Standard forest won!")
    print("    Letting trees repeat strong features")
    print("    can be better than forcing diversity.")
else:
    print("\n  → Tied! Both approaches work here.")

# ── 5. Predict the new applicant ───────────────────────────────
print(f"\n{'=' * 52}")
print("  NEW APPLICANT: income=$58k, credit=695, debt=32%")
print("=" * 52)

new = pd.DataFrame([{"income": 58, "credit_score": 695, "debt_ratio": 32}])

print("\n  Unique-question forest:")
for i, t in enumerate(unique_trees):
    feat, thresh = t["feature"], t["threshold"]
    val = new[feat].values[0]
    direction = "yes → left" if val <= thresh else "no → right"
    subtree = t["left_tree"] if val <= thresh else t["right_tree"]
    pred = subtree.predict(new)[0]
    print(f"    Tree {i+1}: {feat} <= {thresh}? ({val}) {direction} → {pred}")

print(f"\n  Standard forest:")
for i, tree in enumerate(std_model.estimators_):
    pred = tree.predict(new)[0]
    root_feat = feature_names[tree.tree_.feature[0]]
    root_thresh = round(tree.tree_.threshold[0], 1)
    print(f"    Tree {i+1}: {root_feat} <= {root_thresh}? → {pred}")
