import numpy as np

# Simpson's rule
def composite_simpson(f, a, b, n=4):
    if n % 2 != 0:
        raise ValueError("n must be even.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = f(x)
    return h / 3 * (fx[0] + 4 * np.sum(fx[1:-1:2]) + 2 * np.sum(fx[2:-2:2]) + fx[-1])

# a) Transformed integral: ∫₁^∞ t^(-7/4) * sin(1/t) dt
def fa(t):
    return t**(-7/4) * np.sin(1 / t)

# b) Transformed integral: ∫₀^1 t^2 * sin(1/t) dt
def fb(t):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(t == 0, 0, t**2 * np.sin(1 / t))

# 固定 n=4，選擇合理的截斷值
n = 4
a_trunc_upper = 20     # for ∞ → 選一個夠大
b_trunc_lower = 1e-6   # for 0 → 避免除以 0

Ia = composite_simpson(fa, 1, a_trunc_upper, n)
Ib = composite_simpson(fb, b_trunc_lower, 1, n)
print()
print(f"(a) Approximation (n=4) ≈ {Ia:.15f}")
print()
print(f"(b) Approximation (n=4) ≈ {Ib:.15f}")
print()
