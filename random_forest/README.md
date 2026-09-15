# Random Forest — Complete Summary

## 1. What Is Random Forest?

A random forest is a machine learning algorithm that builds many decision trees and combines their answers. It works for both classification (predicting categories) and regression (predicting numbers).

## 2. How a Single Tree Works

Each tree is a chain of yes/no questions about your data. At every node, the algorithm tries every feature and every cutoff value, picks whichever split produces the purest groups (measured by Gini impurity), and branches. The tree keeps splitting until the leaves are pure enough or a depth limit is reached. Each leaf is labeled with the majority class of the training examples that landed there.

### Gini Impurity

Gini measures how mixed a group is. It ranges from 0 (perfectly pure — all one class) to 0.5 (maximally mixed — 50/50). The tree picks the split that drops the Gini the most. A Gini of 0.5 means the tree has no idea yet and must keep splitting.

Gini doesn't directly decide "safe" or "default" — it just scores how good each possible split is. The actual label at a leaf is simply whichever class has more examples in that group.

### What Questions Does the Tree Ask?

The questions are always simple threshold checks on your features, like "is income > $45k?" or "is credit score > 700?" The tree tries every feature at every possible cutoff, calculates the Gini for each, and picks the winner. The questions are not preset — they are generated automatically from the data during training.

## 3. How the Forest Combines Trees

The forest builds many trees, each one different because of two sources of randomness:

- **Bootstrap sampling**: Each tree trains on a different random draw of rows (with replacement). Some rows repeat, some are left out.
- **Random feature subsets**: At each split, the tree only sees a random handful of columns. This forces each tree to find different splitting strategies.

When predicting, every tree votes independently. For classification, the class with the most votes wins (majority vote). For regression, the forest averages the numbers.

### Why It Works

Individual trees make different errors because they saw different data and different features. When you combine many independent opinions, the random mistakes cancel out and the consistent signal survives. The consensus is emergent, not engineered — each tree independently finds whatever pattern it can, and when most of them land on the same answer, that's strong evidence the answer is real.

## 4. Training vs Prediction

All questions are generated during training, which happens the moment you call `model.fit()`. In that moment, the algorithm scans through the training data, tests every feature and cutoff, and builds the entire tree structure. Once training finishes, the tree is permanently locked — no new questions, no new branches, no self-modification.

Prediction is purely mechanical. A new data point enters at the root, follows the existing yes/no paths, and reads the label at the leaf it lands on. Nothing is generated or learned during prediction.

## 5. Parts of a Tree

- **Root node**: The first question asked. Chosen because it produces the biggest Gini drop across the whole dataset.
- **Branches**: The yes/no paths leading out of each question. Every node has exactly two branches.
- **Internal nodes**: Follow-up questions in the middle of the tree. Same Gini logic, applied to the subset that arrived there.
- **Leaves**: The endpoints — no more questions, just a final label. The majority class of whoever landed there becomes the answer.

## 6. Key Parameters

- **n_estimators**: Number of trees. More trees generally means better, more stable predictions but slower training. Can be auto-detected using OOB score or grid search. Usually 100–500.
- **max_depth**: Limits how deep each tree can grow. Reduces overfitting.
- **max_features**: Controls how many columns each split sees. Default is √n for classification. Main source of tree diversity.
- **min_samples_leaf**: Minimum samples required at a leaf. Raising it smooths the model.
- **n_jobs=-1**: Use all CPU cores to speed up training.

### Auto-Detecting n_estimators

The number of trees is preset — you choose it before training. The forest doesn't grow new trees over time. However, you can auto-detect the best count using OOB score (out-of-bag accuracy from rows each tree didn't train on) or grid search with cross-validation.

## 7. Why Trees Repeat Questions

Trees are built completely independently — tree 5 doesn't know that trees 2 and 4 already asked about debt ratio. If one feature is dominant, most trees will use it when offered. This is by design: the independence is what makes the majority vote reliable.

The algorithm doesn't add more trees or branches as it goes — everything is fixed at training time. It also doesn't generate new questions during prediction; it reuses the same questions on every new data point.

## 8. Why Forced Unique Questions Fail

Forcing each tree to ask a unique root question seems like it would improve diversity, but it usually makes the forest worse:

- **Weaker trees**: Trees forced onto noisy or weak features make more mistakes individually.
- **Bad votes outnumber good votes**: If only 1 tree uses the strong feature and 10 trees use noise, the 10 coin-flippers can outvote the 1 expert.
- **Hard ceiling**: With unique questions, you can only build as many trees as you have features.

In our medical diagnosis example, the unique-question forest scored 67% accuracy (20 misdiagnosed) while the standard forest scored 95% (3 misdiagnosed) — 17 extra wrong answers caused entirely by forcing diversity.

The standard approach is better because every tree asks the best question it can, and diversity emerges naturally from the randomness in sampling.

## 9. Loan Default Example

A bank predicts loan defaults using income, credit score, and debt ratio. Each tree sees a random subset of applicants and features. Tree 1 might split on income, tree 2 on credit score, tree 3 on debt ratio. A new applicant flows through all trees, each casts a vote, and the majority wins.

## 10. Quick Code Reference

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))

# Prediction probabilities
probs = model.predict_proba(X_test)
```
