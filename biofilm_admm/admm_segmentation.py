"""
ADMM (Alternating Direction Method of Multipliers) Biofilm Segmentation Algorithms.
Implements spatial consistency (Total Variation) and area/volume constraints.
"""

import numpy as np


def admm_biofilm_segmentation_2d(
    y: np.ndarray,
    I: np.ndarray,
    lambda_val: float = 0.08,
    rho: float = 2.0,
    mu: float = 0.25,
    gamma: float = 0.35,
    max_area: float = 600.0,
    min_area: float = 120.0,
    int_threshold: float = 0.45,
    tol: float = 1e-4,
    max_iter: int = 10000,
) -> tuple[np.ndarray, int]:
    """
    Solves 2D biofilm image segmentation using Multi-Objective ADMM optimization.

    Parameters:
    -----------
    y : np.ndarray
        Reference/ground truth binary mask (Nx, Ny).
    I : np.ndarray
        Intensity image in range [0, 1] (Nx, Ny).
    lambda_val : float
        Area constraint penalty weight.
    rho : float
        ADMM penalty parameter for constraint reconciliation.
    mu : float
        Anisotropic Total Variation (TV) smoothness penalty.
    gamma : float
        Intensity penalty weight.
    max_area : float
        Maximum area allowed for segmented biofilm (in pixels).
    min_area : float
        Minimum area allowed for segmented biofilm (in pixels).
    int_threshold : float
        Intensity threshold.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    best_z : np.ndarray
        The optimized segmented mask, shape (Nx, Ny).
    best_epoch : int
        The epoch at which optimal convergence/residual was reached.
    """
    Nx, Ny = y.shape
    x = np.random.rand(Nx, Ny)
    z = x.copy()
    u = np.zeros((Nx, Ny))

    best_z = z.copy()
    best_r = float("inf")
    best_epoch = 0

    for epoch in range(1, max_iter + 1):
        # ------------------ x-update (Primal) ------------------
        # Finite-difference gradient of anisotropic Total Variation (TV)
        dx = np.zeros_like(x)
        dx[:, :-1] = x[:, 1:] - x[:, :-1]

        dy = np.zeros_like(x)
        dy[:-1, :] = x[1:, :] - x[:-1, :]

        TV_grad = np.zeros((Nx, Ny))
        TV_grad[:, :-1] -= mu * np.sign(dx[:, :-1])
        TV_grad[:, 1:] += mu * np.sign(dx[:, 1:])
        TV_grad[:-1, :] -= mu * np.sign(dy[:-1, :])
        TV_grad[1:, :] += mu * np.sign(dy[1:, :])

        intensity_penalty = gamma * 2.0 * (I - int_threshold)

        # Quadratic update rule combining data-fidelity, TV regularization, and intensity restraints
        numer = 2.0 * y + rho * (z - u) - lambda_val - TV_grad - intensity_penalty
        denom = 2.0 + rho + gamma
        x = numer / denom
        x = np.clip(x, 0.0, 1.0)  # Clamp to [0, 1]

        # ------------------ z-update (Projection) ------------------
        z_tilde = x + u
        z_vector = z_tilde.flatten()

        # Sort values descending to project onto area boundaries
        idx = np.argsort(z_vector)[::-1]
        sorted_z = z_vector[idx]
        Npix = z_vector.size

        # Enforce maximum area constraint
        z_projected = np.zeros(Npix)
        cum_sum = np.cumsum(sorted_z)
        keep_max_indices = np.where(cum_sum <= max_area)[0]
        n_keep_max = keep_max_indices[-1] + 1 if len(keep_max_indices) > 0 else 0
        z_projected[idx[:n_keep_max]] = sorted_z[:n_keep_max]

        # Enforce minimum area constraint
        current_sum = np.sum(z_projected)
        if current_sum < min_area:
            need = int(np.ceil(min_area - current_sum))
            n_add = min(need, Npix - n_keep_max)
            z_projected[idx[n_keep_max : n_keep_max + n_add]] = sorted_z[n_keep_max : n_keep_max + n_add]

        z_projected = np.clip(z_projected, 0.0, 1.0)
        z = z_projected.reshape(Nx, Ny)

        # ------------------ u-update (Dual) ------------------
        u = u + x - z

        # Primal and dual residuals calculation
        r = np.linalg.norm(x.flatten() - z.flatten())
        s = np.linalg.norm(-rho * (z.flatten() - z_projected))

        if r < best_r:
            best_r = r
            best_z = z.copy()
            best_epoch = epoch

        if r < tol and s < tol:
            break

    return best_z, best_epoch


def admm_biofilm_segmentation_3d(
    y: np.ndarray,
    I: np.ndarray,
    lambda_val: float = 0.08,
    rho: float = 2.0,
    mu: float = 0.2,
    gamma: float = 0.3,
    max_volume: float = 8000.0,
    min_volume: float = 2000.0,
    int_threshold: float = 0.35,
    tol: float = 1e-4,
    max_iter: int = 12000,
) -> tuple[np.ndarray, int]:
    """
    Solves 3D biofilm volume segmentation using Multi-Objective ADMM optimization.

    Parameters:
    -----------
    y : np.ndarray
        Reference/ground truth binary volume (Nx, Ny, Nz).
    I : np.ndarray
        Intensity volume in range [0, 1] (Nx, Ny, Nz).
    lambda_val : float
        Volume constraint penalty weight.
    rho : float
        ADMM penalty parameter for constraint reconciliation.
    mu : float
        Anisotropic Total Variation (TV) smoothness penalty.
    gamma : float
        Intensity penalty weight.
    max_volume : float
        Maximum volume allowed for segmented biofilm (in voxels).
    min_volume : float
        Minimum volume allowed for segmented biofilm (in voxels).
    int_threshold : float
        Intensity threshold.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns:
    --------
    best_z : np.ndarray
        The optimized segmented volume, shape (Nx, Ny, Nz).
    best_epoch : int
        The epoch at which optimal convergence/residual was reached.
    """
    Nx, Ny, Nz = y.shape
    x = np.random.rand(Nx, Ny, Nz)
    z = x.copy()
    u = np.zeros((Nx, Ny, Nz))

    best_z = z.copy()
    best_r = float("inf")
    best_epoch = 0

    for epoch in range(1, max_iter + 1):
        # ------------------ x-update (Primal) ------------------
        # Compute TV gradients (finite-difference anisotropic 3D TV)
        dx = np.zeros_like(x)
        dx[:-1, :, :] = x[1:, :, :] - x[:-1, :, :]

        dy = np.zeros_like(x)
        dy[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]

        dz = np.zeros_like(x)
        dz[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]

        TV_grad = np.zeros((Nx, Ny, Nz))
        TV_grad[:-1, :, :] -= mu * np.sign(dx[:-1, :, :])
        TV_grad[1:, :, :] += mu * np.sign(dx[:-1, :, :])
        TV_grad[:, :-1, :] -= mu * np.sign(dy[:, :-1, :])
        TV_grad[:, 1:, :] += mu * np.sign(dy[:, :-1, :])
        TV_grad[:, :, :-1] -= mu * np.sign(dz[:, :, :-1])
        TV_grad[:, :, 1:] += mu * np.sign(dz[:, :, :-1])

        intensity_penalty = gamma * 2.0 * (I - int_threshold)

        numer = 2.0 * y + rho * (z - u) - lambda_val - TV_grad - intensity_penalty
        denom = 2.0 + rho + gamma
        x = numer / denom
        x = np.clip(x, 0.0, 1.0)

        # ------------------ z-update (Projection) ------------------
        z_tilde = x + u
        z_vector = z_tilde.flatten()

        idx = np.argsort(z_vector)[::-1]
        sorted_z = z_vector[idx]
        Nvox = z_vector.size

        # Enforce maximum volume constraint
        z_projected = np.zeros(Nvox)
        cum_sum = np.cumsum(sorted_z)
        keep_max_indices = np.where(cum_sum <= max_volume)[0]
        n_keep_max = keep_max_indices[-1] + 1 if len(keep_max_indices) > 0 else 0
        z_projected[idx[:n_keep_max]] = sorted_z[:n_keep_max]

        # Enforce minimum volume constraint
        current_sum = np.sum(z_projected)
        if current_sum < min_volume:
            need = int(np.ceil(min_volume - current_sum))
            n_add = min(need, Nvox - n_keep_max)
            z_projected[idx[n_keep_max : n_keep_max + n_add]] = sorted_z[n_keep_max : n_keep_max + n_add]

        z_projected = np.clip(z_projected, 0.0, 1.0)
        z = z_projected.reshape(Nx, Ny, Nz)

        # ------------------ u-update (Dual) ------------------
        u = u + x - z

        # Primal and dual residuals
        r = np.linalg.norm(x.flatten() - z.flatten())
        s = np.linalg.norm(-rho * (z.flatten() - z_projected))

        if r < best_r:
            best_r = r
            best_z = z.copy()
            best_epoch = epoch

        if r < tol and s < tol:
            break

    return best_z, best_epoch
