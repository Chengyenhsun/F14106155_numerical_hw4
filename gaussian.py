import numpy as np

# 原始積分區間
a, b = 1, 1.5

def f(x):
    return x**2 * np.log(x)

# 高斯節點與權重：對應 n = 3 和 n = 4
gauss_data = {
    3: {
        "nodes": np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]),
        "weights": np.array([5/9, 8/9, 5/9])
    },
    4: {
        "nodes": np.array([
            -0.8611363116, -0.3399810436,
             0.3399810436,  0.8611363116
        ]),
        "weights": np.array([
            0.3478548451, 0.6521451549,
            0.6521451549, 0.3478548451
        ])
    }
}

# 高斯積分計算函數
def gaussian_quadrature(n):
    nodes = gauss_data[n]["nodes"]
    weights = gauss_data[n]["weights"]
    
    # 變換節點到區間 [a, b]
    x_transformed = 0.5 * (b - a) * nodes + 0.5 * (b + a)

    integral = 0.5 * (b - a) * np.sum(weights * f(x_transformed))
    return integral

from scipy.integrate import quad
exact, _ = quad(f, a, b)

I3 = gaussian_quadrature(3)
I4 = gaussian_quadrature(4)

# 輸出
print()
print(f"Exact value:                {exact:.15f}")
print()
print(f"Gaussian Quadrature (n=3):  {I3:.15f} , Error: {abs(I3 - exact):.2e}")
print()
print(f"Gaussian Quadrature (n=4):  {I4:.15f} , Error: {abs(I4 - exact):.2e}")
print()
