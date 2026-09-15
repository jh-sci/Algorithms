"""
When Unique Questions Break the Forest
Each tree can ONLY use its assigned feature — no recovery.
Run: python unique_questions_fail.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 200

# Fever perfectly separates infected vs healthy
infected_fever = np.random.uniform(101, 105, n // 2)
healthy_fever  = np.random.uniform(96, 99, n // 2)

# 10 noise features — identical range for both groups
noise_cols = [f"symptom_{i}" for i in range(1, 11)]
noise_infected = {c: np.random.uniform(1, 10, n // 2) for c in noise_cols}
noise_healthy  = {c: np.random.uniform(1, 10, n // 2) for c in noise_cols}

df = pd.DataFrame({
    "fever": np.concatenate([infected_fever, healthy_fever]).round(1),
    **{c: np.concatenate([noise_infected[c], noise_healthy[c]]).round(1) for c in noise_cols},
    "diagnosis": ["infected"] * (n // 2) + ["healthy"] * (n // 2)
})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

feature_names = ["fever"] + noise_cols
X = df[feature_names]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("── The data ──")
print(f"  {len(X_train)} training, {len(X_test)} test patients")
print(f"  fever: infected=101-105, healthy=96-99   (PERFECT signal)")
print(f"  symptom_1 to symptom_10: random 1-10     (PURE NOISE)")
print(f"\n  Only fever matters. The 10 symptoms tell you nothing.\n")

# ── Unique-question forest ──────────────────────────────────────
# Each tree sees ONLY its one assigned feature — cannot fall back on fever
print("=" * 60)
print("  UNIQUE-QUESTION FOREST")
print("  Each tree can ONLY see its assigned feature")
print("=" * 60)

unique_preds_per_tree = []

for i, feat in enumerate(feature_names):
    # Train a small forest on just this ONE column
    X_single_train = X_train[[feat]]
    X_single_test  = X_test[[feat]]

    tree = RandomForestClassifier(n_estimators=1, max_depth=3, random_state=42)
    tree.fit(X_single_train, y_train)
    preds = tree.predict(X_single_test)
    acc = accuracy_score(y_test, preds)
    unique_preds_per_tree.append(preds)

    signal = "SIGNAL ★" if feat == "fever" else "NOISE"
    print(f"  Tree {i+1:>2}: uses only {feat:>12}  →  Acc = {acc:>5.0%}  [{signal}]")

# Majority vote across all 11 single-feature trees
votes_array = np.array(unique_preds_per_tree)  # shape: (11, 60)
final_unique = []
for j in range(len(y_test)):
    col_votes = list(votes_array[:, j])
    final_unique.append(max(set(col_votes), key=col_votes.count))

unique_acc = accuracy_score(y_test, final_unique)
n_wrong = sum(a != b for a, b in zip(final_unique, y_test))

print(f"\n  Majority vote across 11 trees: {unique_acc:.0%} ({n_wrong} misdiagnosed)")

# Show specific misdiagnosed patients
wrong_indices = [j for j in range(len(y_test)) if final_unique[j] != y_test.iloc[j]]
if wrong_indices:
    print(f"\n  ⚠  MISDIAGNOSED PATIENTS:")
    for j in wrong_indices[:5]:  # show up to 5
        row = X_test.iloc[j]
        fever_vote = unique_preds_per_tree[0][j]
        noise_votes = [unique_preds_per_tree[k][j] for k in range(1, 11)]
        infected_votes = sum(1 for v in noise_votes if v == "infected")
        healthy_votes  = sum(1 for v in noise_votes if v == "healthy")
        print(f"\n    Patient: fever={row['fever']}°F  →  actually {y_test.iloc[j]}")
        print(f"      Fever tree voted:  {fever_vote}  ← CORRECT")
        print(f"      10 noise trees:    {infected_votes} infected, {healthy_votes} healthy  ← GUESSING")
        print(f"      Final vote:        {final_unique[j]}  ← {'WRONG!' if final_unique[j] != y_test.iloc[j] else 'correct'}")

# ── Standard forest ──────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("  STANDARD RANDOM FOREST (11 trees, all features available)")
print("=" * 60)

std = RandomForestClassifier(n_estimators=11, max_depth=3, random_state=42)
std.fit(X_train, y_train)

fever_roots = 0
for i, t in enumerate(std.estimators_):
    root = feature_names[t.tree_.feature[0]]
    if root == "fever":
        fever_roots += 1

std_preds = std.predict(X_test)
std_acc = accuracy_score(y_test, std_preds)
std_wrong = sum(a != b for a, b in zip(std_preds, y_test))

print(f"  {fever_roots} of 11 trees found fever at root")
print(f"  Accuracy: {std_acc:.0%} ({std_wrong} misdiagnosed)")

# ── Verdict ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  VERDICT")
print(f"{'=' * 60}")
print(f"\n  Unique questions: {unique_acc:.0%}  ({n_wrong} wrong)")
print(f"  Standard forest:  {std_acc:.0%}  ({std_wrong} wrong)")

if n_wrong > std_wrong:
    print(f"\n  ★ Forcing unique features caused {n_wrong - std_wrong} extra misdiagnoses!")
    print(f"\n  The fever tree got every patient right, but it only")
    print(f"  gets 1 vote out of 11. The other 10 trees are splitting")
    print(f"  on pure noise — they're basically flipping coins.")
    print(f"  When enough coins land wrong, they outvote the one")
    print(f"  tree that actually knows the answer.")
