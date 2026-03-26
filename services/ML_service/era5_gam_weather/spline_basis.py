from __future__ import annotations

import numpy as np


def open_uniform_knots(x_min: float, x_max: float, n_basis: int, degree: int) -> np.ndarray:
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    if n_basis <= degree:
        raise ValueError("n_basis must be > degree")

    n_internal = n_basis - degree - 1
    if n_internal > 0:
        internal = np.linspace(x_min, x_max, n_internal + 2, dtype=np.float64)[1:-1]
    else:
        internal = np.empty(0, dtype=np.float64)

    return np.concatenate(
        (
            np.full(degree + 1, x_min, dtype=np.float64),
            internal,
            np.full(degree + 1, x_max, dtype=np.float64),
        )
    )


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)

    n = x.shape[0]
    n_basis = knots.shape[0] - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector")

    basis = np.zeros((n, n_basis), dtype=np.float64)

    for i in range(n_basis):
        left = knots[i]
        right = knots[i + 1]
        mask = (x >= left) & (x < right)
        basis[mask, i] = 1.0

    basis[x == knots[-1], -1] = 1.0

    for p in range(1, degree + 1):
        next_basis = np.zeros_like(basis)
        for i in range(n_basis):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1]

            if left_den > 0.0:
                next_basis[:, i] += ((x - knots[i]) / left_den) * basis[:, i]

            if i + 1 < n_basis and right_den > 0.0:
                next_basis[:, i] += ((knots[i + p + 1] - x) / right_den) * basis[:, i + 1]

        basis = next_basis

    return basis


def harmonic_basis(x: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if n_harmonics <= 0:
        return np.zeros((x.shape[0], 0), dtype=np.float64)

    out = np.empty((x.shape[0], 2 * n_harmonics), dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * x / period
        j = 2 * (k - 1)
        out[:, j] = np.sin(angle)
        out[:, j + 1] = np.cos(angle)
    return out


def tensor_product_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.einsum("ni,nj->nij", a, b, optimize=True).reshape(a.shape[0], -1)