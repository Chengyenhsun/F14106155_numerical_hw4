import numpy as np

# 被積分的函數
def f(x):
    return np.exp(x) * np.sin(4 * x)

# 區間與步長
a, b = 1, 2
h = 0.1
n = int((b - a) / h)

x = np.linspace(a, b, n + 1)  # 節點 (包含首尾)
midpoints = (x[:-1] + x[1:]) / 2  # 區間中點

# Composite Trapezoidal Rule
def composite_trapezoidal():
    return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))

# Composite Midpoint Rule
def composite_midpoint():
    midpoint_indices = np.arange(1, n, 2)  # 對應 x_1, x_3, ..., x_9
    return 2 * h * np.sum(f(x[midpoint_indices]))


# Composite Simpson's Rule (n must be even)
def composite_simpson():
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")
    return h / 3 * (f(x[0]) + 
                   4 * np.sum(f(x[1:-1:2])) + 
                   2 * np.sum(f(x[2:-2:2])) + 
                   f(x[-1]))

# 計算與輸出
trap = composite_trapezoidal()
mid = composite_midpoint()
simp = composite_simpson()

print()
print("(a)")
print(f"Composite Trapezoidal Rule: {trap:.10f}")
print()
print("(b)")
print(f"Composite Simpson's Rule:   {simp:.10f}")
print()
print("(c)")
print(f"Composite Midpoint Rule:    {mid:.10f}")
print()

