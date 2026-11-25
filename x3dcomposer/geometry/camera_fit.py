"""Fixed-intrinsics camera fit (NumPy), used for the camera labels of object crops.

A crop keeps the physical camera of its scene view, but its principal point is off-centre, which the 9-d camera
encoding (centred principal point) cannot express. The label of a crop is therefore a *virtual* camera with a
centred principal point, fitted by minimizing the reprojection error of the object's visible surface points in
the crop (Levenberg-Marquardt on SE(3) with a Huber loss). The solver is offline and not differentiable.
"""

from __future__ import annotations

import numpy as np

from .camera import project, transform

MAX_ITERATIONS = 60
MIN_POINTS = 12
HUBER_FRACTION = 0.02          # of the raster size
MAX_RMSE_FRACTION = 0.01 * np.sqrt(2)
MAX_P95_FRACTION = 0.04


def _exp_so3(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], np.float64)
    if theta < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(theta) / theta * W + (1 - np.cos(theta)) / theta ** 2 * W @ W


def _residual_jacobian(points: np.ndarray, pixels: np.ndarray, K: np.ndarray, w2c: np.ndarray, eps: float):
    """Reprojection residual (N, 2) and its Jacobian (N, 2, 6) for a left SE(3) update of ``w2c``."""
    y = transform(points, w2c)
    if np.any(y[:, 2] <= eps):
        return None, None
    p = y @ K.T
    residual = p[:, :2] / p[:, 2:3] - pixels
    x, yy, z = y.T
    projection = np.zeros((len(y), 2, 3))
    projection[:, 0, 0], projection[:, 0, 2] = K[0, 0] / z, -K[0, 0] * x / z ** 2
    projection[:, 1, 1], projection[:, 1, 2] = K[1, 1] / z, -K[1, 1] * yy / z ** 2
    motion = np.zeros((len(y), 3, 6))
    motion[:, 0, 1], motion[:, 0, 2] = y[:, 2], -y[:, 1]            # -skew(y)
    motion[:, 1, 0], motion[:, 1, 2] = -y[:, 2], y[:, 0]
    motion[:, 2, 0], motion[:, 2, 1] = y[:, 1], -y[:, 0]
    motion[:, :, 3:] = np.eye(3)
    return residual, projection @ motion


def fit_camera(points: np.ndarray, pixels: np.ndarray, K: np.ndarray, initial_c2w: np.ndarray, size: int) -> dict:
    """Camera pose with fixed zero-skew intrinsics ``K`` that reprojects ``points`` (N, 3) onto ``pixels`` (N, 2).

    Returns ``c2w`` (None if the fit could not start), ``valid`` (enough points, all in front of the camera,
    small residuals) and ``rmse`` in pixels.
    """
    points, pixels = np.asarray(points, np.float64), np.asarray(pixels, np.float64)
    failed = {"c2w": None, "valid": False, "rmse": float("inf")}
    if len(points) < MIN_POINTS:
        return failed
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
    if singular[-1] < 1e-5 * max(singular[0], 1e-30):          # degenerate (exactly planar or collinear) support
        return failed
    extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-12)
    eps = extent * 1e-8
    w2c = np.linalg.inv(initial_c2w)
    huber = size * HUBER_FRACTION
    damping = 1e-3

    def cost(residual: np.ndarray) -> float:
        length = np.linalg.norm(residual, axis=1)
        return float(np.sum(np.where(length <= huber, 0.5 * length ** 2, huber * (length - 0.5 * huber))))

    if _residual_jacobian(points, pixels, K, w2c, eps)[0] is None:
        return failed
    for _ in range(MAX_ITERATIONS):
        residual, jacobian = _residual_jacobian(points, pixels, K, w2c, eps)
        length = np.linalg.norm(residual, axis=1)
        weight = np.sqrt(np.minimum(1.0, huber / np.maximum(length, 1e-12)))
        J = (jacobian * weight[:, None, None]).reshape(-1, 6)
        gradient = J.T @ (residual * weight[:, None]).ravel()
        H = J.T @ J
        if np.linalg.norm(gradient, ord=np.inf) < 1e-8:
            break
        try:
            delta = np.linalg.solve(H + damping * np.diag(np.maximum(np.diag(H), 1e-12)), -gradient)
        except np.linalg.LinAlgError:
            damping *= 10
            continue
        rotation = _exp_so3(delta[:3])
        candidate = w2c.copy()
        candidate[:3, :3] = rotation @ w2c[:3, :3]
        candidate[:3, 3] = rotation @ w2c[:3, 3] + delta[3:]
        new_residual, _ = _residual_jacobian(points, pixels, K, candidate, eps)
        old_cost = cost(residual)
        if new_residual is not None and cost(new_residual) < old_cost:
            improvement = old_cost - cost(new_residual)
            w2c, damping = candidate, max(damping / 3, 1e-12)
            if improvement < 1e-10 * max(1.0, old_cost):
                break
        else:
            damping = min(damping * 10, 1e16)
            if np.linalg.norm(delta[:3]) + np.linalg.norm(delta[3:]) / extent < 1e-10:
                break
    c2w = np.linalg.inv(w2c)
    uv, z = project(points, K, c2w)
    errors = np.linalg.norm(uv - pixels, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    valid = bool(np.isfinite(errors).all() and np.mean(z > eps) >= 0.98 and rmse <= size * MAX_RMSE_FRACTION
                 and np.percentile(errors, 95) <= size * MAX_P95_FRACTION)
    return {"c2w": c2w, "valid": valid, "rmse": rmse}
