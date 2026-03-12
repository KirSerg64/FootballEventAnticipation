"""
ball_kalman.py
--------------
Unscented Kalman Filter (UKF) with Laplacian-robust statistics for football
tracking.

Why the previous Adaptive CA Kalman filter was still insufficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Adaptive CA filter used a **hard Mahalanobis gate** (chi² = 9.21, d ≈ 3.03).
When a kick causes the ball to jump far from the predicted position, the
innovation's Mahalanobis distance often exceeds the gate — so the measurement
is **completely rejected** on the kick frame.  The filter then only predicts
(carrying the old velocity forward), delaying trajectory recovery by 3–5 frames.

Improvements: UKF + Laplacian robust statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Unscented Kalman Filter (UKF) — van der Merwe scaled sigma points**
    The UKF propagates a set of carefully chosen *sigma points* through the
    process and measurement functions.  For linear models (our CA model) this
    gives results identical to the linear KF, but the sigma-point framework is
    the correct foundation for:

    * Nonlinear process models (aerodynamic drag, projectile arc) without
      needing an analytically computed Jacobian.
    * Reliable second-order accuracy under any noise distribution.
    * Natural integration of the Laplacian robust update described below.

    Sigma-point parameters follow van der Merwe (2004):
    ``alpha = 0.3`` (wider spread for heavy-tailed Laplacian distributions),
    ``beta = 2.0`` (optimal kurtosis weight for near-Gaussian posterior),
    ``kappa = 0.0``.

**Laplacian robust M-estimator measurement update**
    Instead of a hard binary gate (accept / reject), the UKF update uses a
    *soft M-estimator* derived from the Laplacian measurement-noise model:

    .. math::

        p(\\mathbf{y}) \\propto \\exp\\!\\left(-\\frac{\\|\\mathbf{y}\\|_{S^{-1}}}{b}\\right)

    where ``b = laplacian_b`` is the Laplacian scale in Mahalanobis units and
    ``‖y‖_{S⁻¹} = d`` is the Mahalanobis distance of the innovation.

    This gives an M-estimator weight:

    .. math::

        w(d) = \\min\\!\\left(1,\\; \\frac{b}{\\max(d, \\epsilon)}\\right)

    The effective measurement-noise covariance is inflated to
    ``R_eff = R / w``, which reduces the Kalman gain proportionally.
    The key properties:

    * ``d ≤ b`` (normal flight, small innovations): ``w = 1`` → full
      Gaussian-like correction.
    * Kick frame, ``d = 4``, ``b = 2``: ``w = 0.5`` → **50 % correction**,
      velocity and acceleration states immediately start tracking the new
      trajectory (vs. 0 % with the old hard gate).
    * Extreme outlier, ``d = 20``, ``b = 2``: ``w = 0.1`` → 10 % correction
      → false YOLO detection barely moves the state.
    * A **safety hard gate** at ``gate_chi2 = 900`` (``d = 30``) discards
      only truly pathological measurements (detector completely off-screen).

**Laplacian adaptive Q**
    Under Laplacian process noise the optimal Q boost when the observed
    innovation magnitude is ``d`` is proportional to ``d / b`` — *linear* in
    the Mahalanobis distance, not quadratic.  The adaptive scale is therefore:

    .. math::

        \\lambda = \\max\\!\\left(1,\\; \\frac{\\sqrt{NIS}}{\\sqrt{NIS_{95}}}\\right)

    This activates earlier (at smaller innovations) than the previous
    ``NIS / 2`` rule and is the correct choice for heavy-tailed Laplacian
    process noise.

Public API
~~~~~~~~~~
``BallKalmanFilter``  (``AdaptiveBallKalmanFilter`` is a backward-compat alias)
    ``initialize(cx, cy)``                – seed from the first observation
    ``predict() → (cx, cy)``             – advance without a measurement
    ``update(cx, cy) → (cx, cy)``        – advance + Laplacian-robust correct
    ``position → (cx, cy) | None``       – current filtered centre
    ``velocity → (vcx, vcy) | None``     – current filtered velocity
    ``acceleration → (acx, acy) | None`` – current filtered acceleration
    ``laplacian_weight``                 – M-estimator weight from last update
    ``initialized``                      – True once seeded
    ``frames_since_detection``           – frames without an accepted measurement
    ``last_measurement_gated``           – True if last update was hard-gated
    ``reset()``                          – return to uninitialised state
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Chi-squared thresholds for 2 degrees of freedom
_CHI2_95_2DOF = 5.991   # sqrt → d = 2.448; Laplacian Q trigger
_CHI2_99_2DOF = 9.210   # sqrt → d = 3.033; kept for reference

# Default hard gate (only for truly extreme outliers, d > 30)
_HARD_GATE_CHI2 = 900.0


def _build_dwna_q(sigma_acc: float, dt: float = 1.0) -> np.ndarray:
    """Build the 6×6 DWNA process-noise covariance for a 2-D CA model.

    State order: ``[cx, cy, vcx, vcy, acx, acy]``.  Each axis follows the
    1-D Discrete White-Noise Acceleration (DWNA) model with acceleration
    noise standard deviation ``sigma_acc``.

    Parameters
    ----------
    sigma_acc:
        Acceleration-noise standard deviation (px/frame²).
    dt:
        Frame time step (default 1 frame).

    Returns
    -------
    np.ndarray
        6×6 symmetric positive-semidefinite covariance matrix.
    """
    g = np.array([0.5 * dt ** 2, dt, 1.0], dtype=np.float64)
    q1 = sigma_acc ** 2 * np.outer(g, g)

    Q = np.zeros((6, 6), dtype=np.float64)
    ix = [0, 2, 4]  # x sub-indices: cx, vcx, acx
    iy = [1, 3, 5]  # y sub-indices: cy, vcy, acy
    for r in range(3):
        for c in range(3):
            Q[ix[r], ix[c]] = q1[r, c]
            Q[iy[r], iy[c]] = q1[r, c]
    return Q


class _VanDerMerweSigmaPoints:
    """Van der Merwe scaled sigma-point generator.

    Parameters
    ----------
    n:
        State dimension.
    alpha:
        Spread parameter.  Larger values give wider sigma-point spread,
        capturing heavier-tailed distributions better.  Default ``0.3``.
    beta:
        Distribution parameter.  ``2.0`` is optimal for Gaussian; also
        appropriate for the near-Gaussian posterior here.
    kappa:
        Secondary scaling parameter.  ``0.0`` (default) is standard.
    """

    def __init__(
        self,
        n: int,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.n = n
        lam = alpha ** 2 * (n + kappa) - n
        self._lam = lam
        self._c = n + lam  # scaling factor for Cholesky

        n_sigma = 2 * n + 1
        self.Wm = np.full(n_sigma, 1.0 / (2.0 * (n + lam)))
        self.Wm[0] = lam / (n + lam)

        self.Wc = self.Wm.copy()
        self.Wc[0] += (1.0 - alpha ** 2 + beta)

    def compute(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Return ``(2n+1, n)`` array of sigma points.

        Parameters
        ----------
        x:
            State mean vector (length *n*).
        P:
            State covariance matrix (n×n).

        Returns
        -------
        np.ndarray
            Array of shape ``(2n+1, n)`` where row 0 is the mean sigma
            point and rows 1..n / n+1..2n are positive/negative shifts.

        Raises
        ------
        np.linalg.LinAlgError
            If ``P`` is not positive-definite (Cholesky fails).
        """
        n = self.n
        U = np.linalg.cholesky(self._c * P)  # lower triangular: U @ U.T = c*P

        X = np.empty((2 * n + 1, n), dtype=np.float64)
        X[0] = x
        for i in range(n):
            X[i + 1]     = x + U[:, i]
            X[n + i + 1] = x - U[:, i]
        return X


class BallKalmanFilter:
    """UKF with Laplacian-robust M-estimator for football centre tracking.

    **Key differences from the previous Adaptive CA Kalman filter:**

    1. *UKF sigma points* — the sigma-point framework correctly propagates
       uncertainty through any future nonlinear process or measurement model
       without a Jacobian.

    2. *Laplacian soft gating* — replaces the hard binary gate with a smooth
       M-estimator weight ``w = min(1, laplacian_b / d)`` where ``d`` is the
       Mahalanobis distance of the innovation.  A kick causing ``d = 4``
       (previously rejected outright) now receives a **50 % Kalman correction**
       on frame 1, allowing the velocity and acceleration states to immediately
       start tracking the new trajectory.

    3. *Laplacian adaptive Q* — the process noise is boosted proportionally to
       ``sqrt(NIS)`` (linear in Mahalanobis distance) rather than ``NIS/2``
       (quadratic).  This is the statistically correct choice for Laplacian
       process noise and activates earlier for moderate innovations.

    Parameters
    ----------
    sigma_acc:
        Acceleration-noise standard deviation (px/frame²).  Default ``30.0``.
    measurement_noise:
        Diagonal of **R** (px²).  Default ``4.0`` (≈ 2 px std).
    laplacian_b:
        Laplacian scale in Mahalanobis units.  Innovations with
        ``d > laplacian_b`` are soft-downweighted.  Default ``2.0`` (kicks
        at d=4 get 50 %, extreme outliers at d=20 get 10 %).
    gate_chi2:
        Hard gate threshold (chi² 2 DOF).  Only pathological detections
        (``d > 30`` by default) are discarded entirely.  Set lower to
        increase robustness at the cost of kick-response speed.
    adaptive_q:
        Enable Laplacian adaptive Q scaling.  Default *True*.
    adaptive_q_decay:
        Per-frame decay of the Q scale factor back toward 1.0.
        Default ``0.85``.
    adaptive_q_max_scale:
        Maximum Q scale factor.  Default ``50.0``.
    ukf_alpha:
        UKF sigma-point spread parameter.  Default ``0.3`` (wider spread
        appropriate for heavy-tailed Laplacian distributions).
    ukf_beta:
        UKF distribution parameter.  Default ``2.0``.
    ukf_kappa:
        UKF secondary scaling parameter.  Default ``0.0``.
    process_noise:
        Ignored (kept for backward compatibility with call sites that passed
        the old ``process_noise`` keyword argument).
    """

    def __init__(
        self,
        sigma_acc: float = 30.0,
        measurement_noise: float = 4.0,
        laplacian_b: float = 2.0,
        gate_chi2: float = _HARD_GATE_CHI2,
        adaptive_q: bool = True,
        adaptive_q_decay: float = 0.85,
        adaptive_q_max_scale: float = 50.0,
        ukf_alpha: float = 0.3,
        ukf_beta: float = 2.0,
        ukf_kappa: float = 0.0,
        # Legacy — silently ignored
        process_noise: float | None = None,  # noqa: ARG002
    ) -> None:
        dt = 1.0

        n = 6  # state dimension

        # -- State transition (Constant-Acceleration, linear) -------------------
        # State: [cx, cy, vcx, vcy, acx, acy]
        self.F = np.array(
            [
                [1, 0, dt,  0, 0.5 * dt**2, 0          ],
                [0, 1,  0, dt, 0,           0.5 * dt**2],
                [0, 0,  1,  0, dt,          0          ],
                [0, 0,  0,  1, 0,           dt         ],
                [0, 0,  0,  0, 1,           0          ],
                [0, 0,  0,  0, 0,           1          ],
            ],
            dtype=np.float64,
        )

        # -- Measurement matrix (observe position only) -------------------------
        self.H = np.zeros((2, n), dtype=np.float64)
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy

        # -- Process-noise covariance (DWNA, Laplacian-adaptively scaled) ------
        self._Q_base = _build_dwna_q(sigma_acc, dt)

        # -- Measurement-noise covariance --------------------------------------
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # -- Laplacian + gating settings --------------------------------------
        self._laplacian_b = laplacian_b
        self._gate_chi2 = gate_chi2

        # -- Adaptive Q settings ----------------------------------------------
        self._adaptive_q = adaptive_q
        self._adaptive_q_decay = adaptive_q_decay
        self._adaptive_q_max_scale = adaptive_q_max_scale
        self._adaptive_scale: float = 1.0

        # -- UKF sigma-point generator ----------------------------------------
        self._sp = _VanDerMerweSigmaPoints(n, alpha=ukf_alpha, beta=ukf_beta, kappa=ukf_kappa)

        # -- State (lazy-initialised) -----------------------------------------
        self.x: np.ndarray | None = None
        self.P: np.ndarray = np.diag(
            [100.0, 100.0, 900.0, 900.0, 2500.0, 2500.0]
        ).astype(np.float64)
        self._initialized = False
        self._frames_since_detection: int = 0
        self._last_gated: bool = False
        self._laplacian_weight: float = 1.0

    # ── Effective Q ────────────────────────────────────────────────────────────

    @property
    def _Q(self) -> np.ndarray:
        return self._Q_base * self._adaptive_scale

    # ── UKF predict / update internals ─────────────────────────────────────────

    def _ukf_predict(self) -> tuple[np.ndarray, np.ndarray]:
        """UKF predict step: propagate sigma points through F.

        Returns
        -------
        (x_pred, P_pred):
            Predicted state and covariance.
        """
        X = self._sp.compute(self.x, self.P)              # (2n+1, 6)
        X_pred = (self.F @ X.T).T                         # apply linear F

        x_pred = X_pred.T @ self._sp.Wm                   # weighted mean

        P_pred = self._Q.copy()
        for i, Xi in enumerate(X_pred):
            d = Xi - x_pred
            P_pred += self._sp.Wc[i] * np.outer(d, d)

        return x_pred, P_pred

    def _ukf_update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """UKF update step with Laplacian M-estimator.

        The measurement noise **R** is inflated by ``1 / w`` where
        ``w = min(1, laplacian_b / d)`` and ``d`` is the Mahalanobis distance
        of the innovation.  This soft-downweights large innovations (false
        detections) while still applying substantial correction for genuine
        kicks (moderate-to-large ``d``).

        Parameters
        ----------
        x_pred, P_pred:
            Predicted state and covariance from :meth:`_ukf_predict`.
        z:
            Measurement vector ``[cx, cy]``.

        Returns
        -------
        (x_new, P_new, nis, w):
            Updated state, updated covariance, NIS, and Laplacian weight.
        """
        X = self._sp.compute(x_pred, P_pred)

        # Propagate sigma points through linear H
        Z_pred = (self.H @ X.T).T                         # (2n+1, 2)
        z_pred = Z_pred.T @ self._sp.Wm                   # weighted mean (2,)

        # Innovation covariance S and cross-covariance Pxz
        S = self.R.copy()
        Pxz = np.zeros((6, 2), dtype=np.float64)
        for i, (Xi, Zi) in enumerate(zip(X, Z_pred)):
            dz = Zi - z_pred
            dx = Xi - x_pred
            S   += self._sp.Wc[i] * np.outer(dz, dz)
            Pxz += self._sp.Wc[i] * np.outer(dx, dz)

        # Innovation
        y = z - z_pred
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
        d = float(np.sqrt(max(nis, 0.0)))

        # ── Laplacian M-estimator weight ──────────────────────────────────
        # w = 1 for d ≤ laplacian_b (Gaussian-like region)
        # w = laplacian_b/d < 1 for d > laplacian_b (soft downweight)
        w = min(1.0, self._laplacian_b / max(d, 1e-8))

        # Inflate R by 1/w → reduce Kalman gain for large innovations
        R_eff = self.R / w
        S_eff = (S - self.R) + R_eff   # S_eff = H*P*H^T + R_eff

        K = Pxz @ np.linalg.inv(S_eff)

        x_new = x_pred + K @ y

        # Joseph-form covariance update (numerically stable)
        I_KH = np.eye(6, dtype=np.float64) - K @ self.H
        P_new = I_KH @ P_pred @ I_KH.T + K @ R_eff @ K.T

        return x_new, P_new, nis, w

    # ── Public interface ──────────────────────────────────────────────────────

    def initialize(self, cx: float, cy: float) -> None:
        """Seed the filter from the first observation."""
        self.x = np.array([cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [100.0, 100.0, 900.0, 900.0, 2500.0, 2500.0]
        ).astype(np.float64)
        self._initialized = True
        self._frames_since_detection = 0
        self._adaptive_scale = 1.0
        self._last_gated = False
        self._laplacian_weight = 1.0

    def predict(self) -> tuple[float, float]:
        """Advance state one frame without a new measurement (UKF predict only).

        Returns
        -------
        (cx, cy):
            Predicted ball centre.

        Raises
        ------
        RuntimeError
            If called before :meth:`initialize` or :meth:`update`.
        """
        if not self._initialized:
            raise RuntimeError(
                "BallKalmanFilter.initialize() must be called before predict()."
            )
        # Decay adaptive scale toward 1.0
        self._adaptive_scale = max(1.0, self._adaptive_scale * self._adaptive_q_decay)

        self.x, self.P = self._ukf_predict()
        self._frames_since_detection += 1
        self._last_gated = True
        self._laplacian_weight = 1.0
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        """Advance state and apply a Laplacian-robust measurement correction.

        The measurement is **soft-downweighted** rather than hard-gated:
        a kick causing a large Mahalanobis distance ``d`` still contributes
        a partial correction proportional to ``min(1, laplacian_b / d)``.
        Only detections beyond the safety hard gate (default ``d = 30``) are
        discarded outright.

        If the filter has not been initialised yet the state is seeded from
        ``(cx, cy)`` and the raw position is returned immediately.

        Returns
        -------
        (cx, cy):
            Filtered (or predicted) ball centre.
        """
        if not self._initialized:
            self.initialize(cx, cy)
            return cx, cy

        # Decay adaptive scale
        self._adaptive_scale = max(1.0, self._adaptive_scale * self._adaptive_q_decay)

        # -- UKF predict step -------------------------------------------------
        x_pred, P_pred = self._ukf_predict()

        # -- Pre-compute raw innovation to check hard gate --------------------
        z = np.array([cx, cy], dtype=np.float64)
        y_raw = z - self.H @ x_pred
        S_raw = self.H @ P_pred @ self.H.T + self.R
        nis_raw = float(y_raw @ np.linalg.inv(S_raw) @ y_raw)

        if nis_raw > self._gate_chi2:
            # Extreme outlier — discard measurement entirely
            logger.debug(
                "Ball UKF: hard gate triggered (NIS=%.1f > %.1f, d=%.1f); "
                "predict-only this frame.",
                nis_raw, self._gate_chi2, float(np.sqrt(nis_raw)),
            )
            self.x = x_pred
            self.P = P_pred
            self._frames_since_detection += 1
            self._last_gated = True
            self._laplacian_weight = 0.0
            return float(self.x[0]), float(self.x[1])

        # -- UKF update with Laplacian M-estimator ----------------------------
        x_new, P_new, nis, w = self._ukf_update(x_pred, P_pred, z)

        # -- Laplacian adaptive Q: sqrt(NIS) scaling (linear in d) ------------
        if self._adaptive_q:
            d = float(np.sqrt(max(nis, 0.0)))
            d_thresh = float(np.sqrt(_CHI2_95_2DOF))  # ≈ 2.45
            if d > d_thresh:
                # Laplacian-motivated: boost ∝ d (not d²)
                new_scale = min(d / d_thresh, self._adaptive_q_max_scale)
                if new_scale > self._adaptive_scale:
                    self._adaptive_scale = new_scale
                    logger.debug(
                        "Ball UKF: Laplacian Q boost %.1f× (d=%.2f)",
                        self._adaptive_scale, d,
                    )

        self.x = x_new
        self.P = P_new
        self._frames_since_detection = 0
        self._last_gated = False
        self._laplacian_weight = w

        logger.debug(
            "Ball UKF: update accepted (NIS=%.2f, d=%.2f, w=%.2f)",
            nis, float(np.sqrt(max(nis, 0.0))), w,
        )
        return float(self.x[0]), float(self.x[1])

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        """*True* once :meth:`initialize` or the first :meth:`update` has run."""
        return self._initialized

    @property
    def position(self) -> tuple[float, float] | None:
        """Current filtered ``(cx, cy)``, or *None* if not initialized."""
        if not self._initialized or self.x is None:
            return None
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Current filtered velocity ``(vcx, vcy)`` in px/frame, or *None*."""
        if not self._initialized or self.x is None:
            return None
        return float(self.x[2]), float(self.x[3])

    @property
    def acceleration(self) -> tuple[float, float] | None:
        """Current filtered acceleration ``(acx, acy)`` in px/frame², or *None*."""
        if not self._initialized or self.x is None:
            return None
        return float(self.x[4]), float(self.x[5])

    @property
    def frames_since_detection(self) -> int:
        """Consecutive frames without an accepted (non-hard-gated) measurement."""
        return self._frames_since_detection

    @property
    def last_measurement_gated(self) -> bool:
        """*True* if the last :meth:`update` was rejected by the hard gate."""
        return self._last_gated

    @property
    def laplacian_weight(self) -> float:
        """M-estimator weight from the most recent :meth:`update` call.

        * ``1.0``: measurement was in the Gaussian region (small innovation).
        * ``0 < w < 1``: measurement was soft-downweighted (kick or moderate
          outlier).
        * ``0.0``: measurement was hard-gated (extreme outlier).
        """
        return self._laplacian_weight

    def reset(self) -> None:
        """Return the filter to its uninitialised state."""
        self.x = None
        self.P = np.diag(
            [100.0, 100.0, 900.0, 900.0, 2500.0, 2500.0]
        ).astype(np.float64)
        self._initialized = False
        self._frames_since_detection = 0
        self._adaptive_scale = 1.0
        self._last_gated = False
        self._laplacian_weight = 1.0


# Backward-compatibility alias
AdaptiveBallKalmanFilter = BallKalmanFilter
