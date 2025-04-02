import numpy as np
from scipy.integrate import quad

# 被積分函數 f(x, y)
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

def simpson_double(f, a, b, n, m):
    hx = (b - a) / n
    x = np.linspace(a, b, n + 1)

    total = 0
    for i in range(n + 1):
        xi = x[i]
        # 對應 y 的上下限
        ya = np.sin(xi)
        yb = np.cos(xi)
        hy = (yb - ya) / m
        y = np.linspace(ya, yb, m + 1)

        fxiy = f(xi, y)

        coeff_y = np.ones(m + 1)
        coeff_y[1:-1:2] = 4
        coeff_y[2:-2:2] = 2
        coeff_y[0] = coeff_y[-1] = 1
        I_y = hy / 3 * np.sum(coeff_y * fxiy)

        # Simpson x-weight
        cx = 1
        if i == 0 or i == n:
            cx = 1
        elif i % 2 == 1:
            cx = 4
        else:
            cx = 2

        total += cx * I_y

    return hx / 3 * total


gauss_nodes = {
    3: np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
}
gauss_weights = {
    3: np.array([5/9, 8/9, 5/9])
}

def gaussian_double(f, ax, bx, ay_func, by_func, n, m):
    ξ = gauss_nodes[n]
    c = gauss_weights[n]
    η = gauss_nodes[m]
    d = gauss_weights[m]

    total = 0
    for i in range(n):
        # 對應 x
        xi = 0.5 * (bx - ax) * ξ[i] + 0.5 * (bx + ax)
        wi = c[i]
        ya = ay_func(xi)
        yb = by_func(xi)

        inner = 0
        for j in range(m):
            yj = 0.5 * (yb - ya) * η[j] + 0.5 * (yb + ya)
            wj = d[j]
            inner += wj * f(xi, yj)
        total += wi * inner * 0.5 * (yb - ya)

    return 0.5 * (bx - ax) * total


def f_exact(x):
    return np.cos(x)**3 - np.sin(x)**3

exact, _ = quad(f_exact, 0, np.pi/4)

simpson_result = simpson_double(f, 0, np.pi/4, n=4, m=4)
gauss_result = gaussian_double(f, 0, np.pi/4, np.sin, np.cos, n=3, m=3)

print()
print(f"(a) Simpson's Rule (n=4, m=4)  = {simpson_result:.15f}")
print()
print(f"(b) Gaussian Quadrature (n=3, m=3)  = {gauss_result:.15f}")
print()
print(f"(c) Exact value  = {exact:.15f}")
print(f"    Simpson's Rule Error  = {abs(simpson_result - exact):.15e}")
print(f"    Gaussian Quadrature Error  = {abs(gauss_result - exact):.15e}")
print()
