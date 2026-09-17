"""Educational gradient boosting using only standard Python.
Run: python3 gradient_boosting_demo.py
This demonstrates boosted decision stumps, not the full XGBoost algorithm.
"""

from statistics import mean


def train_stump(sizes, errors):
    """Choose a size cutoff and leaf corrections minimizing squared error."""
    unique = sorted(set(sizes))
    best = None
    for low, high in zip(unique, unique[1:]):
        cutoff = (low + high) / 2
        left = [e for size, e in zip(sizes, errors) if size <= cutoff]
        right = [e for size, e in zip(sizes, errors) if size > cutoff]
        left_value, right_value = mean(left), mean(right)
        loss = sum((e - left_value) ** 2 for e in left)
        loss += sum((e - right_value) ** 2 for e in right)
        if best is None or loss < best[0]:
            best = (loss, cutoff, left_value, right_value)
    if best is None:
        return (sizes[0], mean(errors), mean(errors))
    return best[1:]


def tree_output(tree, size):
    cutoff, left, right = tree
    return left if size <= cutoff else right


def predict(base, trees, rate, size):
    return base + rate * sum(tree_output(tree, size) for tree in trees)


def main():
    # Synthetic historical prices. Hold out houses for evaluation.
    data = [(size, 50_000 + size * 200) for size in range(800, 3201, 100)]
    training = [row for i, row in enumerate(data) if i % 5 != 2]
    testing = [row for i, row in enumerate(data) if i % 5 == 2]
    sizes = [row[0] for row in training]
    actual = [row[1] for row in training]

    rate = 0.1
    rounds = 100  # Try 5, 20, or 100.
    base = mean(actual)
    estimates = [base] * len(training)
    trees = []
    new_size = 1750

    print("GRADIENT BOOSTING: no external packages required")
    print(f"Starting prediction: ${base:,.0f}")
    print(f"Learning rate: {rate}; correction rounds: {rounds}")
    print("\nAutomatically learned rules (first five trees):")
    for step in range(1, rounds + 1):
        errors = [truth - estimate for truth, estimate in zip(actual, estimates)]
        tree = train_stump(sizes, errors)
        trees.append(tree)
        estimates = [
            estimate + rate * tree_output(tree, size)
            for estimate, size in zip(estimates, sizes)
        ]
        if step <= 5:
            cutoff, left, right = tree
            print(f"Tree {step}: size <= {cutoff:,.0f} sq ft?")
            print(f"  Yes: add ${rate * left:,.0f}; No: add ${rate * right:,.0f}")
        if step in {1, 5, 20, rounds}:
            train_error = mean(abs(a - p) for a, p in zip(actual, estimates))
            test_error = mean(
                abs(price - predict(base, trees, rate, size))
                for size, price in testing
            )
            print(f"After {step} trees: training error ${train_error:,.0f}; "
                  f"held-out error ${test_error:,.0f}")

    print("\nHeld-out houses (not used in training):")
    for size, price in testing:
        estimate = predict(base, trees, rate, size)
        print(f"{size:,} sq ft | Actual ${price:,.0f} | Predicted ${estimate:,.0f}")

    print(f"\nPrediction for a new {new_size:,} sq ft house:")
    running = base
    print(f"Starting value: ${running:,.0f}")
    for i, tree in enumerate(trees[:5], 1):
        correction = rate * tree_output(tree, new_size)
        running += correction
        print(f"Tree {i}: {correction:+,.0f} -> ${running:,.0f}")
    remaining = rate * sum(tree_output(tree, new_size) for tree in trees[5:])
    print(f"Remaining trees combined: {remaining:+,.0f}")
    print(f"Final predicted price: ${predict(base, trees, rate, new_size):,.0f}")
    print("\nToy data only: real prices need location, condition, and more data.")


if __name__ == "__main__":
    main()

