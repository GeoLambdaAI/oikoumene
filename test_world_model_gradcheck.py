"""
Finite-difference gradient checks for the hand-written backward passes in
world_model.py (NumPy JEPA backbone).

Each analytic gradient is compared against a central finite-difference
estimate. For a primitive y = f(inputs) we fix a random upstream gradient
`dy`, define the scalar surrogate L = sum(y * dy), and verify that the
analytic input gradients returned by `*_backward` match d L / d input
computed numerically. SIGReg has a scalar loss, so it needs no surrogate
(but its random projection matrix is pinned via a fresh seeded RandomState
on every forward call, making the function deterministic for differencing).

This is the test the README/CHANGELOG refer to when they describe the v0.2
analytic backprop as "gradient-checked against finite differences".
"""
import numpy as np

from world_model import (
    linear_forward, linear_backward,
    gelu_forward, gelu_backward,
    rms_norm_forward, rms_norm_backward,
    adaln_forward, adaln_backward,
    sigreg_forward, sigreg_backward,
)

EPS = 1e-6
ATOL = 1e-7
RTOL = 1e-5


def _numeric_grad(scalar_fn, x, eps=EPS):
    """Central finite-difference gradient of scalar_fn() w.r.t. array x (in place)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = scalar_fn()
        x[idx] = orig - eps
        fm = scalar_fn()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def _assert_close(analytic, numeric, name):
    max_abs = np.max(np.abs(analytic - numeric))
    denom = np.max(np.abs(analytic)) + np.max(np.abs(numeric)) + 1e-12
    rel = max_abs / denom
    print(f"  {name:10s} max|Δ|={max_abs:.2e} rel={rel:.2e}")
    assert np.allclose(analytic, numeric, atol=ATOL, rtol=RTOL), \
        f"{name}: analytic vs numeric mismatch (max abs {max_abs:.2e}, rel {rel:.2e})"


def test_linear_gradients():
    print("\nGradcheck: linear")
    rng = np.random.RandomState(0)
    x = rng.randn(4, 5); W = rng.randn(5, 3); b = rng.randn(3); dy = rng.randn(4, 3)

    _, cache = linear_forward(x, W, b)
    dx, dW, db = linear_backward(dy, cache)

    _assert_close(dx, _numeric_grad(lambda: np.sum(linear_forward(x, W, b)[0] * dy), x), "dx")
    _assert_close(dW, _numeric_grad(lambda: np.sum(linear_forward(x, W, b)[0] * dy), W), "dW")
    _assert_close(db, _numeric_grad(lambda: np.sum(linear_forward(x, W, b)[0] * dy), b), "db")


def test_gelu_gradients():
    print("\nGradcheck: gelu")
    rng = np.random.RandomState(1)
    x = rng.randn(4, 6); dy = rng.randn(4, 6)

    _, cache = gelu_forward(x)
    dx = gelu_backward(dy, cache)
    _assert_close(dx, _numeric_grad(lambda: np.sum(gelu_forward(x)[0] * dy), x), "dx")


def test_rms_norm_gradients():
    print("\nGradcheck: rms_norm")
    rng = np.random.RandomState(2)
    x = rng.randn(4, 7); gamma = rng.randn(7); dy = rng.randn(4, 7)

    _, cache = rms_norm_forward(x, gamma)
    dx, dgamma = rms_norm_backward(dy, cache)

    _assert_close(dx, _numeric_grad(lambda: np.sum(rms_norm_forward(x, gamma)[0] * dy), x), "dx")
    _assert_close(dgamma, _numeric_grad(lambda: np.sum(rms_norm_forward(x, gamma)[0] * dy), gamma), "dgamma")


def test_adaln_gradients():
    print("\nGradcheck: adaln")
    rng = np.random.RandomState(3)
    D, A, B = 6, 4, 5
    h = rng.randn(B, D); action = rng.randn(B, A); gamma = rng.randn(D)
    W_scale = rng.randn(A, D); W_shift = rng.randn(A, D); dy = rng.randn(B, D)

    _, cache = adaln_forward(h, action, gamma, W_scale, W_shift)
    dh, daction, dgamma, dW_scale, dW_shift = adaln_backward(dy, cache, W_scale, W_shift)

    fwd = lambda: np.sum(adaln_forward(h, action, gamma, W_scale, W_shift)[0] * dy)
    _assert_close(dh, _numeric_grad(fwd, h), "dh")
    _assert_close(daction, _numeric_grad(fwd, action), "daction")
    _assert_close(dgamma, _numeric_grad(fwd, gamma), "dgamma")
    _assert_close(dW_scale, _numeric_grad(fwd, W_scale), "dW_scale")
    _assert_close(dW_shift, _numeric_grad(fwd, W_shift), "dW_shift")


def test_sigreg_gradients():
    print("\nGradcheck: sigreg")
    rng = np.random.RandomState(4)
    Z = rng.randn(32, 6)

    # Pin the random projection matrix by re-seeding on every forward call so
    # the loss is a deterministic function of Z (otherwise central differences
    # would compare across different projection draws).
    def loss():
        return sigreg_forward(Z, n_projections=8, rng=np.random.RandomState(7))[0]

    _, cache = sigreg_forward(Z, n_projections=8, rng=np.random.RandomState(7))
    dZ = sigreg_backward(cache)
    _assert_close(dZ, _numeric_grad(loss, Z), "dZ")


if __name__ == "__main__":
    test_linear_gradients()
    test_gelu_gradients()
    test_rms_norm_gradients()
    test_adaln_gradients()
    test_sigreg_gradients()
    print("\nALL GRADIENT CHECKS PASSED")
