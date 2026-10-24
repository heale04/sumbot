"""Reproduce the paper's 3x3 normal-form experiment: MI-SPRM+ vs SPRM+ (simultaneous
updates, uniform / R-weighted averaging, duality gap). Expected at 1e5 iterations:
MI-SPRM+ ~2.6e-5, SPRM+(eta=0.1) ~4.0e-5, SPRM+(eta=1) ~2.1e-4, SPRM+(eta=0.01) ~3.9e-4."""

import sys

import numpy as np

A = np.array([[3, 0, -3], [0, 3, -4], [0, 0, 1]], dtype=np.float64)


def project(theta, R):
    y = np.maximum(theta, 0.0)
    if y.sum() >= R:
        return y
    # projection onto {x >= 0, sum x = R}
    u = np.sort(theta)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, len(u) + 1)
    cond = u - (css - R) / k > 0
    rho = np.max(np.where(cond)[0]) + 1
    tau = (css[rho - 1] - R) / rho
    return np.maximum(theta - tau, 0.0)


def duality_gap(x, y):
    # player 0 minimizes x^T A y, player 1 maximizes
    return float(np.max(A.T @ x) - np.min(A @ y))


def run(mode, T, eta=1.0, R0=1.0):
    n = 3
    th_hat = [np.full(n, 1 / n), np.full(n, 1 / n)]
    th_prev = [np.full(n, 1 / n), np.full(n, 1 / n)]
    x_prev = [np.full(n, 1 / n), np.full(n, 1 / n)]
    F_prev = [np.zeros(n), np.zeros(n)]
    R = R0 if mode == "mi" else R0 / eta  # SPRM+ with eta == MI-SPRM+ with fixed R/eta
    num = [np.zeros(n), np.zeros(n)]
    den = 0.0
    for t in range(1, T + 1):
        # prediction uses losses at x^{t-1}
        l_prev = [A @ x_prev[1], -(A.T @ x_prev[0])]
        F_prev = [l_prev[i] @ x_prev[i] - l_prev[i] for i in range(2)]
        th = [project(th_hat[i] + F_prev[i], R) for i in range(2)]
        x = [th[i] / th[i].sum() for i in range(2)]
        l = [A @ x[1], -(A.T @ x[0])]
        F = [l[i] @ x[i] - l[i] for i in range(2)]
        B_prev = sum(0.5 * np.sum((th_hat[i] - th_prev[i]) ** 2) for i in range(2))
        B_cur = sum(0.5 * np.sum((th_hat[i] - th[i]) ** 2) for i in range(2))
        th_hat = [project(th_hat[i] + F[i], R) for i in range(2)]
        if mode == "mi":
            diff = sum(np.sum((F[i] - F_prev[i]) ** 2) for i in range(2))
            if diff - (B_prev + B_cur) / 2 > 0:
                R += 1.0
        w = R if mode == "mi" else 1.0
        num = [num[i] + w * x[i] for i in range(2)]
        den += w
        th_prev, x_prev = th, x
    xb = [num[i] / den for i in range(2)]
    return duality_gap(xb[0], xb[1]), R


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    gap, R = run("mi", T)
    print(f"MI-SPRM+   T={T}: gap={gap:.3e} final R={R}")
    for eta in (0.01, 0.1, 1.0):
        gap, _ = run("sprm", T, eta=eta)
        print(f"SPRM+ eta={eta}: gap={gap:.3e}")
