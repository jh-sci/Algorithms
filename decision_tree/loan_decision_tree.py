"""
Loan Approval Decision Tree
===========================

Two implementations of the loan-approval tree from the diagram:

  1. RuleBasedTree  - hand-coded, mirrors the diagram exactly, no dependencies.
                      Great for understanding and for auditable "why" explanations.

  2. train_learned_tree() - uses scikit-learn to LEARN a tree from example data,
                      the way you would in a real project.

The diagram logic:

    credit_score < 650                      -> Deny
    credit_score >= 650:
        income < 40000                      -> Deny
        income >= 40000:
            debt_ratio high (>= 0.40)       -> Deny
            debt_ratio low  (<  0.40)       -> Approve
"""

from dataclasses import dataclass


# --------------------------------------------------------------------------
# 1. Hand-coded tree that follows the diagram exactly
# --------------------------------------------------------------------------

@dataclass
class Applicant:
    name: str
    credit_score: int
    income: float
    debt_ratio: float   # existing monthly debt / monthly income, e.g. 0.30


# Thresholds (the "splits" in the tree)
CREDIT_THRESHOLD = 650
INCOME_THRESHOLD = 40_000
DEBT_THRESHOLD = 0.40


class RuleBasedTree:
    """Decision tree implemented as explicit if/else rules from the diagram."""

    def predict(self, app: Applicant) -> str:
        """Return 'Approve' or 'Deny'."""
        decision, _ = self.predict_with_reason(app)
        return decision

    def predict_with_reason(self, app: Applicant):
        """Return (decision, human-readable reason) so every result is auditable."""
        if app.credit_score < CREDIT_THRESHOLD:
            return "Deny", (
                f"credit score {app.credit_score} is below {CREDIT_THRESHOLD}"
            )

        if app.income < INCOME_THRESHOLD:
            return "Deny", (
                f"credit score OK ({app.credit_score}), but income "
                f"${app.income:,.0f} is below ${INCOME_THRESHOLD:,.0f}"
            )

        if app.debt_ratio >= DEBT_THRESHOLD:
            return "Deny", (
                f"credit and income OK, but debt ratio {app.debt_ratio:.0%} "
                f"is at or above {DEBT_THRESHOLD:.0%}"
            )

        return "Approve", (
            f"credit score {app.credit_score} >= {CREDIT_THRESHOLD}, "
            f"income ${app.income:,.0f} >= ${INCOME_THRESHOLD:,.0f}, "
            f"and debt ratio {app.debt_ratio:.0%} < {DEBT_THRESHOLD:.0%}"
        )


def demo_rule_based():
    print("=" * 68)
    print("1. HAND-CODED TREE (follows the diagram exactly)")
    print("=" * 68)

    tree = RuleBasedTree()
    applicants = [
        Applicant("Applicant A", credit_score=600, income=52_000, debt_ratio=0.20),
        Applicant("Applicant B", credit_score=720, income=55_000, debt_ratio=0.25),
        Applicant("Applicant C", credit_score=700, income=60_000, debt_ratio=0.55),
        Applicant("Applicant D", credit_score=680, income=35_000, debt_ratio=0.15),
    ]

    for app in applicants:
        decision, reason = tree.predict_with_reason(app)
        mark = "APPROVE" if decision == "Approve" else "DENY   "
        print(f"\n  [{mark}] {app.name}")
        print(f"          score={app.credit_score}, income=${app.income:,.0f}, "
              f"debt={app.debt_ratio:.0%}")
        print(f"          reason: {reason}")


# --------------------------------------------------------------------------
# 2. Learned tree using scikit-learn
# --------------------------------------------------------------------------

def train_learned_tree():
    print("\n" + "=" * 68)
    print("2. LEARNED TREE (scikit-learn trains on example data)")
    print("=" * 68)

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        print("\n  scikit-learn not installed - skipping this section.")
        print("  Install it with:  pip install scikit-learn")
        return

    # Small synthetic dataset: [credit_score, income, debt_ratio] -> approve(1)/deny(0)
    # Labels follow the same underlying logic, with a little noise-free structure.
    X = [
        [600, 52000, 0.20], [640, 80000, 0.10], [610, 45000, 0.30],  # low score -> deny
        [720, 55000, 0.25], [700, 60000, 0.15], [800, 90000, 0.10],  # strong -> approve
        [660, 35000, 0.20], [680, 30000, 0.15], [655, 38000, 0.05],  # low income -> deny
        [700, 60000, 0.55], [720, 70000, 0.50], [690, 50000, 0.45],  # high debt -> deny
        [750, 65000, 0.20], [670, 42000, 0.30], [710, 48000, 0.35],  # approve
    ]
    y = [0, 0, 0,  1, 1, 1,  0, 0, 0,  0, 0, 0,  1, 1, 1]

    feature_names = ["credit_score", "income", "debt_ratio"]

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    print("\n  Tree the model learned from the data:\n")
    rules = export_text(clf, feature_names=feature_names)
    # indent for pretty printing
    for line in rules.splitlines():
        print("    " + line)

    # Try it on new applicants
    print("\n  Predictions on new applicants:")
    new_apps = [
        ("Applicant B", [720, 55000, 0.25]),
        ("Applicant C", [700, 60000, 0.55]),
        ("Applicant E", [660, 47000, 0.22]),
    ]
    for name, features in new_apps:
        pred = clf.predict([features])[0]
        proba = clf.predict_proba([features])[0].max()
        decision = "Approve" if pred == 1 else "Deny"
        print(f"    {name}: {decision}  (confidence {proba:.0%})  "
              f"[score={features[0]}, income=${features[1]:,}, debt={features[2]:.0%}]")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    demo_rule_based()
    train_learned_tree()
    print()
