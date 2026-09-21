"""
gradient_descent.py
-------------------
A tiny, from-scratch gradient descent demo — no libraries needed.

It fits the simplest possible model:  y_hat = w * x
to a single data point (x = 1, y = 2), so the correct answer is w = 2.

Run it:   python3 gradient_descent.py
"""


def train_one_point(x=1.0, y=2.0, w=0.0, learning_rate=0.1, steps=15):
    """
    Gradient descent on one data point.

    Model:      y_hat = w * x
    Loss:       L = (y_hat - y) ** 2          (squared error)
    Gradient:   dL/dw = 2 * (y_hat - y) * x   (chain rule)
    Update:     w = w - learning_rate * gradient
    """
    print(f"{'step':>4} | {'w':>8} | {'y_hat':>8} | {'error':>8} | {'grad':>8} | {'loss':>8}")
    print("-" * 60)

    for step in range(steps + 1):
        y_hat = w * x                 # 1. predict
        error = y_hat - y             # 2. how wrong we are
        loss = error ** 2             #    the thing we want to shrink
        gradient = 2 * error * x      # 3. slope of the loss w.r.t. w

        print(f"{step:>4} | {w:>8.4f} | {y_hat:>8.4f} | "
              f"{error:>8.4f} | {gradient:>8.4f} | {loss:>8.4f}")

        w = w - learning_rate * gradient   # 4. step downhill

    print("-" * 60)
    print(f"final w = {w:.4f}   (target was {y / x:.4f})")
    return w


if __name__ == "__main__":
    print("Fitting y_hat = w * x  to the point (x=1, y=2)\n")
    train_one_point()

    # Try changing these and re-running:
    #   train_one_point(learning_rate=0.01)   # too small -> crawls
    #   train_one_point(learning_rate=0.95)   # too big   -> overshoots
    #   train_one_point(x=1, y=5, w=0)        # different target
