## Summary of random forest

A random forest is a machine learning algorithm that builds many decision trees and combines their answers. It works for both classification (predicting categories) and regression (predicting numbers).

**How a tree works.** Each tree is a chain of yes/no questions about your data. At every node, the algorithm tries every feature and every cutoff value, picks whichever split produces the purest groups (measured by Gini impurity), and branches. Gini ranges from 0 (perfectly pure — all one class) to 0.5 (maximally mixed — 50/50). The tree keeps splitting until the leaves are pure enough or a depth limit is reached. Each leaf is labeled with the majority class of the training examples that landed there.

**How the forest works.** The forest builds many trees, each one different because of two sources of randomness: bootstrap sampling (each tree trains on a random draw of rows, with replacement) and random feature subsets (at each split, the tree only sees a handful of columns). When predicting, every tree votes independently, and the forest takes the majority vote for classification or the average for regression.

**Why it works.** Individual trees make different errors because they saw different data and different features. When you combine many independent opinions, the random mistakes cancel out and the consistent signal survives.

**Training vs prediction.** All questions are generated during training (when you call `model.fit()`). After that, the tree structure is permanently locked — no new questions, no new branches, no self-modification. Prediction just runs new data through the existing fixed paths.

**The tree's parts.** The root node is the first question asked. Branches are the yes/no paths out of each node. Internal nodes are follow-up questions in the middle. Leaves are the endpoints that hold the final answer.

**Key parameters.** `n_estimators` sets the number of trees (can be auto-detected via OOB score or grid search). `max_depth` limits how deep trees grow. `max_features` controls how many columns each split sees. These are all set before training and don't change.

**Why trees repeat questions.** Trees are built independently with no communication between them. If one feature is dominant, most trees will use it — and that's a strength, not a weakness. Forcing unique questions is dangerous because it pushes trees onto weak or noisy features, and those bad trees can outvote the good ones. In our medical example, forcing 10 trees onto noise features caused 17 extra misdiagnoses compared to the standard forest.That covers everything we discussed — from how Gini picks the questions, to how the forest votes, to why forcing unique questions backfires. The text summary above the diagram has the full explanation, and the diagram organizes the same ideas into the six main topics we walked through.
