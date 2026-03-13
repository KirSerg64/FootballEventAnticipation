"""
ball_kalman.py
--------------
Ball tracking: UKF with Laplacian-robust statistics (retained for reference /
backward compatibility) **plus** the primary ``BallDCFTracker`` — a
detection-first tracker with MOSSE correlation-filter gap filling.

Why every Kalman / UKF approach fails for instant kicks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Both the adaptive CA filter and the UKF share the same fundamental flaw: they
use the *predicted state* as an anchor when deciding how much to trust a new
measurement.  Once the filter converges (after many frames of stable tracking)
the state covariance **P** is small.  For a hard kick where the ball jumps
200 px in one frame:

* Innovation Mahalanobis distance:  d ≈ 200 / √R ≈ 100  (with R = 4 px²)
* Laplacian M-estimator weight:      w = min(1, b/d) = min(1, 2/100) = 0.02
* Net correction this frame:         2 % of 200 px = 4 px  (vs. 200 px needed)

This is a **mathematical limitation of any Kalman-family filter**, not a
tuning problem.  No choice of b, gate threshold, Q, or sigma-point spread
can recover instant direction changes once P has converged.

Primary solution: ``BallDCFTracker`` (detection-first + MOSSE DCF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The root insight from tracking-by-detection research (ByteTrack 2022,
OC-SORT 2023): **the detector is almost always right when it fires**.
The Kalman filter was wrong to gate or smooth YOLO ball detections.

``BallDCFTracker`` design:

1. **Detection-first (zero latency on kicks)**
   When YOLO provides a ball detection at ``(cx, cy)`` the tracker outputs
   that position *immediately* — no Kalman smoothing, no gating.  A 200 px
   kick is captured with 100 % accuracy on frame 1.

2. **MOSSE DCF for gap filling** *(Bolme et al., CVPR 2010)*
   A Minimum Output Sum of Squared Error (MOSSE) discriminative correlation
   filter maintains an appearance model of the ball in the Fourier domain.
   When YOLO misses the ball, the filter searches for it in an expanded
   window around the velocity-extrapolated position.  The Peak-to-Sidelobe
   Ratio (PSR) measures search confidence; results below the PSR threshold
   fall back to linear extrapolation.

   Complexity is O(n log n) per frame (FFT-based) — fast enough for real
   time even on CPU.

3. **Velocity extrapolation** (tertiary fallback)
   When DCF confidence is low (PSR < threshold) or the gap exceeds
   ``max_gap_for_dcf``, the last known velocity (median of recent detections)
   extrapolates the position forward.

``BallKalmanFilter`` / UKF (retained, secondary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Public API (primary)
~~~~~~~~~~~~~~~~~~~~
``BallDCFTracker``  ← **use this**
    ``initialize(cx, cy[, frame])``       – seed from first observation
    ``predict([frame]) → (cx, cy)``       – MOSSE search or extrapolation
    ``update(cx, cy[, frame]) → (cx,cy)`` – accept detection immediately
    ``position → (cx, cy) | None``        – current position
    ``velocity → (vcx, vcy) | None``      – velocity from recent detections
    ``speed → float``                      – scalar speed (px/frame)
    ``motion_state → BallMotionState``     – UNKNOWN/STATIC/IN_FLIGHT/HIGH_SPEED
    ``predicted_position → (cx,cy)|None`` – one-frame-ahead position estimate
    ``adaptive_search_radius → int``       – ROI radius scaled by speed and gap
    ``last_source → str``                  – "detected"/"mosse"/"predicted"/"none"
    ``initialized``                        – True once seeded
    ``frames_since_detection``             – consecutive frames without YOLO
    ``last_measurement_gated``             – always False (DCF never gates)
    ``laplacian_weight``                   – always 1.0 (compat with UKF API)
    ``reset()``                            – return to uninitialised state

``BallKalmanFilter``  (``AdaptiveBallKalmanFilter`` alias, retained for
backward compatibility — not used by default any more)
"""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Chi-squared thresholds for 2 degrees of freedom
_CHI2_95_2DOF = 5.991   # sqrt → d = 2.448; Laplacian Q trigger
_CHI2_99_2DOF = 9.210   # sqrt → d = 3.033; kept for reference

# Default hard gate (only for truly extreme outliers, d > 30)
_HARD_GATE_CHI2 = 900.0

# Speed thresholds for BallMotionState classification (px/frame)
_SPEED_STATIC = 3.0    # below this → STATIC (ball nearly stationary)
_SPEED_HIGH   = 25.0   # above this → HIGH_SPEED (kick or pass)


class BallMotionState(Enum):
    """Ball motion-state classification (FRoG-MOT inspired).

    Used by :class:`BallDCFTracker` to switch between prediction strategies:

    * ``UNKNOWN`` — insufficient detection history.
    * ``STATIC`` — ball nearly stationary (speed < ``_SPEED_STATIC`` px/frame).
      Prediction: hold the last known position (velocity is noise).
    * ``IN_FLIGHT`` — moderate speed (``_SPEED_STATIC`` ≤ speed < ``_SPEED_HIGH``).
      Prediction: median velocity over recent detections (smooth estimate).
    * ``HIGH_SPEED`` — fast kick or pass (speed ≥ ``_SPEED_HIGH`` px/frame).
      Prediction: most-recent frame-to-frame displacement (direction is precise,
      median would lag behind the sudden change).

    Reference: FRoG-MOT (Fast and Robust Generic MOT by IoU and Motion-State
    Associations).  The core idea is that each tracked object has a *motion
    state*, and prediction accuracy improves when the model is adapted to
    that state rather than using a single fixed dynamics model.
    """

    UNKNOWN    = 0
    STATIC     = 1
    IN_FLIGHT  = 2
    HIGH_SPEED = 3


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


# ---------------------------------------------------------------------------
# MOSSE Discriminative Correlation Filter
# ---------------------------------------------------------------------------

class _MOSSEFilter:
    """Minimum Output Sum of Squared Error (MOSSE) correlation filter.

    Learns a frequency-domain filter that produces a peaked Gaussian response
    at the target location when convolved with the image patch.  All
    computation is in the Fourier domain — O(n log n) per frame.

    Reference: Bolme et al., "Visual Object Tracking using Adaptive Correlation
    Filters", CVPR 2010.

    Parameters
    ----------
    patch_size:
        Size (px) of the square template used for the FFT.  The search window
        and ball patches are all resized to this size before correlation.
        Default ``32``.
    lr:
        Online learning rate (``η`` in the paper).  Each update mixes the
        new filter estimate with the accumulated one: ``A ← (1-η)A + η·A_new``.
        Default ``0.125``.
    response_sigma:
        Standard deviation (in patch pixels) of the desired Gaussian response
        peak.  Smaller values force a sharper peak.  Default ``2.0``.
    """

    def __init__(
        self,
        patch_size: int = 32,
        lr: float = 0.125,
        response_sigma: float = 2.0,
    ) -> None:
        self._ps = patch_size
        self._lr = lr
        self._initialized: bool = False

        # Desired Gaussian response in frequency domain (centred peak)
        cy_, cx_ = patch_size // 2, patch_size // 2
        yi, xi = np.mgrid[0:patch_size, 0:patch_size]
        g = np.exp(-((xi - cx_) ** 2 + (yi - cy_) ** 2) / (2 * response_sigma ** 2))
        self._G: np.ndarray = np.fft.fft2(g)

        # Hanning window — reduces spectral leakage at patch borders
        h1 = np.hanning(patch_size)
        self._hann: np.ndarray = np.outer(h1, h1)

        # Accumulated filter numerator and denominator (online update)
        self._A: np.ndarray | None = None   # Σ G_i · conj(F_i)
        self._B: np.ndarray | None = None   # Σ F_i · conj(F_i)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_patch(
        self, gray: np.ndarray, cx: float, cy: float, radius: int
    ) -> np.ndarray | None:
        """Extract a (2·radius × 2·radius) grayscale patch centred on (cx, cy).

        The patch is zero-padded when the ball is near the frame boundary.
        Returns *None* if the image is too small for even a 1×1 patch.
        """
        h, w = gray.shape
        side = 2 * radius
        if side <= 0 or w <= 0 or h <= 0:
            return None

        icx = int(round(cx))
        icy = int(round(cy))
        x1, y1 = icx - radius, icy - radius
        x2, y2 = x1 + side, y1 + side

        # Fast path: patch fully inside the frame
        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
            return gray[y1:y2, x1:x2].astype(np.float64)

        # Slow path: clamp and zero-pad
        patch = np.zeros((side, side), dtype=np.float64)
        sx1, sy1 = max(x1, 0), max(y1, 0)
        sx2, sy2 = min(x2, w), min(y2, h)
        if sx2 > sx1 and sy2 > sy1:
            patch[sy1 - y1: sy2 - y1, sx1 - x1: sx2 - x1] = gray[sy1:sy2, sx1:sx2]
        return patch

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """Resize → log-normalise → z-score → apply Hanning window."""
        p = cv2.resize(patch, (self._ps, self._ps), interpolation=cv2.INTER_LINEAR)
        p = p.astype(np.float64)
        p = np.log1p(p)                               # log normalization
        mu, sigma = p.mean(), p.std()
        p = (p - mu) / (sigma + 1e-8)                # z-score
        return p * self._hann                         # Hanning-windowed

    # ── Public methods ────────────────────────────────────────────────────────

    def initialize(
        self, gray: np.ndarray, cx: float, cy: float, radius: int
    ) -> bool:
        """Initialise the filter from a single patch around ``(cx, cy)``.

        Parameters
        ----------
        gray:
            Grayscale frame (uint8 or float).
        cx, cy:
            Ball centre in frame coordinates.
        radius:
            Half-side of the extracted patch (px).

        Returns
        -------
        bool
            *True* on success, *False* if the patch could not be extracted.
        """
        patch = self._extract_patch(gray, cx, cy, radius)
        if patch is None:
            return False
        F = np.fft.fft2(self._preprocess(patch))
        self._A = self._G * np.conj(F)
        self._B = F * np.conj(F) + 1e-5              # small regularisation
        self._initialized = True
        return True

    def update(
        self, gray: np.ndarray, cx: float, cy: float, radius: int
    ) -> None:
        """Update the filter with a *confirmed* ball position.

        Only call this after a verified YOLO detection — never on predicted
        positions, to avoid filter drift.
        """
        patch = self._extract_patch(gray, cx, cy, radius)
        if patch is None:
            return
        if not self._initialized:
            self.initialize(gray, cx, cy, radius)
            return
        F = np.fft.fft2(self._preprocess(patch))
        new_A = self._G * np.conj(F)
        new_B = F * np.conj(F) + 1e-5
        self._A = (1.0 - self._lr) * self._A + self._lr * new_A
        self._B = (1.0 - self._lr) * self._B + self._lr * new_B

    def find(
        self,
        gray: np.ndarray,
        cx_pred: float,
        cy_pred: float,
        search_radius: int,
    ) -> tuple[float, float, float]:
        """Search for the ball in a window around the predicted position.

        The search window (``2·search_radius × 2·search_radius``) is resized
        to ``patch_size × patch_size`` and correlated with the learned filter.
        The peak in the response map gives the displacement; the
        Peak-to-Sidelobe Ratio (PSR) quantifies confidence.

        Parameters
        ----------
        gray:
            Grayscale frame.
        cx_pred, cy_pred:
            Centre of the search window (predicted ball position).
        search_radius:
            Half-side of the search window (px).  Should be larger than the
            maximum expected per-frame displacement.

        Returns
        -------
        (found_cx, found_cy, psr):
            Estimated ball position and PSR.  ``psr < psr_threshold``
            indicates low confidence.
        """
        if not self._initialized:
            return cx_pred, cy_pred, 0.0

        raw = self._extract_patch(gray, cx_pred, cy_pred, search_radius)
        if raw is None:
            return cx_pred, cy_pred, 0.0

        F = np.fft.fft2(self._preprocess(raw))
        H = self._A / (self._B + 1e-5)               # learned filter
        response = np.real(np.fft.ifft2(H * F))
        response = np.fft.fftshift(response)

        # Peak location
        ry, rx = np.unravel_index(np.argmax(response), response.shape)
        peak_val = float(response[ry, rx])

        # PSR: sidelobe = everything outside 11×11 window around peak
        sl = response.copy()
        r1, r2 = max(ry - 5, 0), min(ry + 6, response.shape[0])
        c1, c2 = max(rx - 5, 0), min(rx + 6, response.shape[1])
        sl[r1:r2, c1:c2] = np.nan
        sl_vals = sl[~np.isnan(sl)]
        if sl_vals.size == 0:
            return cx_pred, cy_pred, 0.0
        sl_mean = float(sl_vals.mean())
        sl_std = float(sl_vals.std()) + 1e-8
        psr = (peak_val - sl_mean) / sl_std

        # Convert peak offset in response space to frame pixel offset.
        # Each cell of the (patch_size × patch_size) response corresponds to
        # (2·search_radius / patch_size) pixels in the frame.
        ps = self._ps
        scale = (2.0 * search_radius) / ps
        dc = (rx - ps // 2) * scale   # x offset (col)
        dr = (ry - ps // 2) * scale   # y offset (row)

        h, w = gray.shape
        found_cx = float(np.clip(cx_pred + dc, 0, w - 1))
        found_cy = float(np.clip(cy_pred + dr, 0, h - 1))

        return found_cx, found_cy, float(psr)


# ---------------------------------------------------------------------------
# BallDCFTracker — detection-first tracker with MOSSE gap filling
# ---------------------------------------------------------------------------

class BallDCFTracker:
    """Detection-first ball tracker with MOSSE correlation-filter gap filling.

    Design principle
    ~~~~~~~~~~~~~~~~
    YOLO detections are **always accepted immediately** — no Kalman gating, no
    smoothing.  This gives zero-latency response to instant direction changes
    (kicks, bounces): when the ball jumps 200 px in one frame, YOLO detects it
    at the new position and the tracker outputs that position directly.

    When YOLO misses the ball (occlusion, motion blur, false negative), the
    MOSSE filter searches for the ball by *appearance* in an expanded window
    around the velocity-extrapolated position.  If the search confidence (PSR)
    is above the threshold, the MOSSE result is used; otherwise the tracker
    falls back to linear velocity extrapolation from recent detections.

    FRoG-MOT motion-state extensions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Inspired by FRoG-MOT (Fast and Robust Generic MOT by IoU and Motion-State
    Associations), the tracker classifies the ball into one of four motion
    states (:class:`BallMotionState`) and adapts both prediction and search
    accordingly:

    * ``HIGH_SPEED`` (kick / pass, ≥ ``_SPEED_HIGH`` px/frame) → use the
      most-recent frame-to-frame displacement for prediction (no smoothing);
      widen the search window proportionally to speed.
    * ``IN_FLIGHT`` (normal flight) → median velocity (stable estimate).
    * ``STATIC`` (stationary ball) → hold current position.
    * ``UNKNOWN`` (insufficient history) → median velocity.

    Parameters
    ----------
    patch_size:
        Template patch size (px) for the MOSSE FFT.  Default ``32``.
    search_radius:
        Half-side (px) of the MOSSE search window when YOLO misses.  The
        effective search diameter is ``2 × search_radius``.  Default ``60``.
    psr_threshold:
        Minimum Peak-to-Sidelobe Ratio for a MOSSE result to be accepted.
        Values below this fall back to velocity extrapolation.
        Default ``7.0`` (MOSSE paper recommendation).
    dcf_lr:
        MOSSE online learning rate.  Default ``0.125``.
    velocity_history:
        Number of recent YOLO detections used to estimate the ball velocity
        via median differencing.  Default ``5``.
    max_gap_for_dcf:
        Maximum consecutive frames without a YOLO detection for which the
        MOSSE search is attempted.  Beyond this the tracker only extrapolates
        (ball is likely lost).  Default ``5``.
    """

    def __init__(
        self,
        patch_size: int = 32,
        search_radius: int = 60,
        psr_threshold: float = 7.0,
        dcf_lr: float = 0.125,
        velocity_history: int = 5,
        max_gap_for_dcf: int = 5,
    ) -> None:
        self._search_radius = search_radius
        self._psr_threshold = psr_threshold
        self._patch_radius = patch_size // 2   # half-side for template extraction
        self._max_gap_for_dcf = max_gap_for_dcf

        self._mosse = _MOSSEFilter(
            patch_size=patch_size,
            lr=dcf_lr,
        )

        # Ring buffer of recent (cx, cy) YOLO detections for velocity estimation
        self._det_history: deque[tuple[float, float]] = deque(maxlen=velocity_history)

        # State
        self._last_cx: float | None = None
        self._last_cy: float | None = None
        self._initialized: bool = False
        self._frames_since_detection: int = 0
        self._last_gated: bool = False
        # Source of the most-recent ball position: "detected", "mosse", "predicted"
        self._last_source: str = "none"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert a BGR or grayscale frame to a grayscale uint8 array."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _estimate_velocity(self) -> tuple[float, float] | None:
        """Return (vx, vy) in px/frame from recent detection history, or *None*."""
        hist = list(self._det_history)
        if len(hist) < 2:
            return None
        diffs_x = [hist[i][0] - hist[i - 1][0] for i in range(1, len(hist))]
        diffs_y = [hist[i][1] - hist[i - 1][1] for i in range(1, len(hist))]
        return float(np.median(diffs_x)), float(np.median(diffs_y))

    def _extrapolate(self) -> tuple[float, float]:
        """Position extrapolation one frame forward, motion-state aware.

        Motion-state strategy (FRoG-MOT):

        * ``HIGH_SPEED`` — use the **most-recent** frame-to-frame displacement
          only.  During a kick the ball travels in a well-defined direction; the
          median of older frames would under-estimate speed and trail the ball.
        * ``STATIC`` — return the current position unchanged.  Small random
          movements are noise, not real motion.
        * ``IN_FLIGHT`` / ``UNKNOWN`` — use the **median** velocity over recent
          detections.  More stable in normal free-flight.
        """
        if self._last_cx is None:
            return 0.0, 0.0

        state = self.motion_state

        if state == BallMotionState.STATIC:
            return float(self._last_cx), float(self._last_cy)

        if state == BallMotionState.HIGH_SPEED:
            hist = list(self._det_history)
            if len(hist) >= 2:
                vx = hist[-1][0] - hist[-2][0]
                vy = hist[-1][1] - hist[-2][1]
            else:
                vel = self._estimate_velocity()
                vx, vy = vel if vel is not None else (0.0, 0.0)
        else:
            # IN_FLIGHT or UNKNOWN: median over history
            vel = self._estimate_velocity()
            if vel is None:
                return float(self._last_cx), float(self._last_cy)
            vx, vy = vel

        return (
            float(self._last_cx + vx),
            float(self._last_cy + vy),
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def initialize(
        self, cx: float, cy: float, frame: np.ndarray | None = None
    ) -> None:
        """Seed the tracker from the first observation.

        Parameters
        ----------
        cx, cy:
            Ball centre from the first YOLO detection.
        frame:
            Current video frame (BGR or gray).  When provided the MOSSE
            filter is initialised immediately.
        """
        self._last_cx = cx
        self._last_cy = cy
        self._det_history.clear()
        self._det_history.append((cx, cy))
        self._initialized = True
        self._frames_since_detection = 0
        self._last_gated = False
        self._last_source = "detected"
        self._mosse._initialized = False   # reset any stale filter state
        if frame is not None:
            gray = self._to_gray(frame)
            self._mosse.initialize(gray, cx, cy, self._patch_radius)

    def predict(
        self, frame: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Estimate ball position for a frame where YOLO produced no detection.

        Strategy (in priority order):

        1. **MOSSE search**: if a frame is provided, the filter is not yet in a
           long gap (``frames_since_detection ≤ max_gap_for_dcf``), and the
           search PSR exceeds ``psr_threshold`` → return the MOSSE result.
        2. **Velocity extrapolation**: linear prediction from recent detections.

        Returns
        -------
        (cx, cy):
            Estimated ball centre.

        Raises
        ------
        RuntimeError
            If called before :meth:`initialize` or :meth:`update`.
        """
        if not self._initialized:
            raise RuntimeError(
                "BallDCFTracker.initialize() must be called before predict()."
            )
        self._frames_since_detection += 1
        self._last_gated = True

        # Velocity-extrapolated prediction used as MOSSE search centre
        pred_cx, pred_cy = self._extrapolate()

        # MOSSE search if a frame is available and the gap is short
        if (
            frame is not None
            and self._mosse._initialized
            and self._frames_since_detection <= self._max_gap_for_dcf
        ):
            gray = self._to_gray(frame)
            vel = self._estimate_velocity()
            vel_mag = float(np.hypot(vel[0], vel[1])) if vel else 0.0
            # Widen search radius proportionally to estimated ball speed
            adaptive_radius = max(
                self._search_radius,
                int(vel_mag * self._frames_since_detection * 1.5),
            )
            found_cx, found_cy, psr = self._mosse.find(
                gray, pred_cx, pred_cy, adaptive_radius
            )
            if psr >= self._psr_threshold:
                logger.debug(
                    "Ball DCF: MOSSE gap-fill at (%.1f, %.1f) PSR=%.1f",
                    found_cx, found_cy, psr,
                )
                self._last_cx = found_cx
                self._last_cy = found_cy
                self._last_source = "mosse"
                return found_cx, found_cy

        # Fallback: velocity extrapolation
        self._last_cx = pred_cx
        self._last_cy = pred_cy
        self._last_source = "predicted"
        logger.debug(
            "Ball DCF: velocity extrapolation to (%.1f, %.1f)", pred_cx, pred_cy
        )
        return pred_cx, pred_cy

    def update(
        self, cx: float, cy: float, frame: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Accept a YOLO detection immediately and update the MOSSE model.

        **The ball position is returned exactly as supplied by YOLO** — no
        Kalman smoothing, no gating.  This is the key property that enables
        zero-latency capture of instant direction changes from kicks.

        If the filter has not been initialised yet it is seeded from
        ``(cx, cy)`` and the raw position is returned.

        Parameters
        ----------
        cx, cy:
            Ball centre from YOLO (pixel coordinates).
        frame:
            Current video frame for MOSSE model update (optional but
            recommended).

        Returns
        -------
        (cx, cy):
            Same as the input — the YOLO detection is passed through unchanged.
        """
        if not self._initialized:
            self.initialize(cx, cy, frame)
            return cx, cy

        # Accept immediately — no gating
        self._last_cx = cx
        self._last_cy = cy
        self._det_history.append((cx, cy))
        self._frames_since_detection = 0
        self._last_gated = False
        self._last_source = "detected"

        # Update MOSSE appearance model
        if frame is not None:
            gray = self._to_gray(frame)
            if not self._mosse._initialized:
                self._mosse.initialize(gray, cx, cy, self._patch_radius)
            else:
                self._mosse.update(gray, cx, cy, self._patch_radius)

        return cx, cy

    def reset(self) -> None:
        """Return the tracker to its uninitialised state."""
        self._last_cx = None
        self._last_cy = None
        self._det_history.clear()
        self._initialized = False
        self._frames_since_detection = 0
        self._last_gated = False
        self._last_source = "none"
        self._mosse._initialized = False
        self._mosse._A = None
        self._mosse._B = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        """*True* once :meth:`initialize` or the first :meth:`update` has run."""
        return self._initialized

    @property
    def frames_since_detection(self) -> int:
        """Consecutive frames without a YOLO detection (0 after every detection)."""
        return self._frames_since_detection

    @property
    def last_measurement_gated(self) -> bool:
        """Always *False* for YOLO updates; *True* during predict-only frames."""
        return self._last_gated

    @property
    def laplacian_weight(self) -> float:
        """Always 1.0 — retained for API compatibility with ``BallKalmanFilter``."""
        return 1.0

    @property
    def last_source(self) -> str:
        """Source of the most-recent ball position estimate.

        Possible values:

        * ``"detected"`` — position came from a YOLO detection (stage-1 global
          or stage-2 ROI); the tracker accepted it immediately.
        * ``"mosse"`` — MOSSE correlation filter found the ball in a gap frame.
        * ``"predicted"`` — velocity extrapolation (MOSSE not available or PSR
          too low).
        * ``"none"`` — tracker has not yet been initialised.

        This property is used by :class:`SegmentationTracker` to populate
        ``SegmentationResult.ball_source`` for visualisation and diagnostics.
        """
        return self._last_source

    @property
    def position(self) -> tuple[float, float] | None:
        """Current ball ``(cx, cy)``, or *None* if not yet initialized."""
        if not self._initialized or self._last_cx is None:
            return None
        return float(self._last_cx), float(self._last_cy)

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Velocity estimate ``(vx, vy)`` in px/frame from recent detections."""
        return self._estimate_velocity()

    @property
    def speed(self) -> float:
        """Ball speed (px/frame) from the **most-recent** frame-to-frame displacement.

        Using the latest pair (rather than the median) gives an immediate
        reading when the ball is kicked — the FRoG-MOT principle of
        classifying the *current* motion state from the *current* observation.
        """
        hist = list(self._det_history)
        if len(hist) < 2:
            return 0.0
        vx = hist[-1][0] - hist[-2][0]
        vy = hist[-1][1] - hist[-2][1]
        return float(np.hypot(vx, vy))

    @property
    def motion_state(self) -> BallMotionState:
        """Ball motion state (FRoG-MOT motion-state classification).

        Classifies from the **most-recent** frame-to-frame speed so that
        a sudden kick is detected on the very next detection, without waiting
        for a median to catch up.
        """
        hist = list(self._det_history)
        if len(hist) < 2:
            return BallMotionState.UNKNOWN
        vx = hist[-1][0] - hist[-2][0]
        vy = hist[-1][1] - hist[-2][1]
        spd = float(np.hypot(vx, vy))
        if spd < _SPEED_STATIC:
            return BallMotionState.STATIC
        if spd >= _SPEED_HIGH:
            return BallMotionState.HIGH_SPEED
        return BallMotionState.IN_FLIGHT

    @property
    def predicted_position(self) -> tuple[float, float] | None:
        """Ball position predicted one frame forward (without advancing state).

        Used by external callers (e.g. :meth:`SegmentationTracker._process_ball`)
        to centre the ROI for secondary low-confidence YOLO detection.
        Returns *None* if the tracker has not been initialised.
        """
        if not self._initialized or self._last_cx is None:
            return None
        return self._extrapolate()

    @property
    def adaptive_search_radius(self) -> int:
        """Search half-radius (px) for ROI-based re-detection.

        Scales proportionally to the **most-recent** frame-to-frame speed
        (so a kick immediately widens the search window) and the number of
        frames since the last confirmed YOLO detection.
        """
        hist = list(self._det_history)
        if len(hist) >= 2:
            vx = hist[-1][0] - hist[-2][0]
            vy = hist[-1][1] - hist[-2][1]
            spd = float(np.hypot(vx, vy))
        else:
            vel = self._estimate_velocity()
            spd = float(np.hypot(vel[0], vel[1])) if vel is not None else 0.0
        gap = max(1, self._frames_since_detection)
        extra = int(spd * gap * 1.5)
        return int(min(self._search_radius + extra, 350))



# ---------------------------------------------------------------------------
# BallCoTrackerTracker — CoTracker3-based ball tracker
# ---------------------------------------------------------------------------

class BallCoTrackerTracker:
    """Ball tracker that uses CoTracker3 to propagate the ball between YOLO detections.

    Design
    ~~~~~~
    1. **First YOLO detection → anchor**.  The ball centre is registered as a
       CoTracker3 query point.  All subsequent frames feed the model
       incrementally (online streaming or offline batch).

    2. **YOLO detections → re-anchor**.  Whenever YOLO detects the ball, the
       detection counter advances; every *redetect_interval* YOLO-confirmed
       frames the query point is reset to the new YOLO position so CoTracker3
       stays locked even after a kick or bounce.

    3. **YOLO miss → CoTracker3 propagation**.  When YOLO misses the ball,
       the current CoTracker3 tracked position is used.  If CoTracker3 is
       still warming up (fewer than ``2 * step`` frames buffered) the tracker
       falls back to linear velocity extrapolation.

    The tracker has the same ``update(cx, cy, frame)`` / ``predict(frame)``
    / ``reset()`` / ``position`` / ``last_source`` interface as
    :class:`BallDCFTracker` so the two are interchangeable inside
    :class:`~segmentation_tracking.segmentation_model.SegmentationTracker`.

    Parameters
    ----------
    redetect_interval:
        Number of *YOLO-confirmed* frames between forced re-anchors.
        After this many YOLO detections the query point is refreshed with
        the most-recent YOLO position even if the previous anchor is still
        tracking well.  Default ``15``.
    hub_model:
        CoTracker3 hub model identifier.  ``"cotracker3_online"`` (default,
        streaming sliding-window) or ``"cotracker3_offline"`` (batch
        inference over frames since last anchor).
    checkpoint:
        Optional local path to a ``.pth`` checkpoint.  When *None* the
        default pretrained weights are downloaded automatically via
        ``torch.hub``.
    device:
        Torch device string.  CUDA availability is checked at runtime;
        falls back to CPU when CUDA is absent.
    velocity_history:
        Number of recent YOLO positions used to estimate the ball velocity
        for the warm-up extrapolation fallback.  Default ``5``.
    """

    _HUB_SOURCE = "facebookresearch/co-tracker"

    def __init__(
        self,
        redetect_interval: int = 15,
        hub_model: str = "cotracker3_online",
        checkpoint: str | None = None,
        device: str = "cuda",
        velocity_history: int = 5,
    ) -> None:
        if hub_model not in ("cotracker3_online", "cotracker3_offline"):
            raise ValueError(
                "hub_model must be 'cotracker3_online' or 'cotracker3_offline'; "
                f"got {hub_model!r}"
            )
        self._redetect_interval = max(1, int(redetect_interval))
        self._hub_model = hub_model
        self._checkpoint = checkpoint
        self._device_str = device
        self._velocity_history = max(2, int(velocity_history))

        # Lazy-loaded CoTracker3 predictor (loaded on first use)
        self._predictor = None
        self._torch_device = None
        self._step: int = 8           # inferred from model at init

        # Frame buffer for CoTracker3 (list of (3,H,W) float32 tensors)
        self._frame_buf: list = []
        self._max_buf: int = 3 * self._step + 1  # online mode only
        # Query tensor: (1, 1, 3) — single ball point [t, x, y]
        self._queries = None
        # Whether the first CoTracker3 call has been made (online mode)
        self._ct_initialized: bool = False
        # Pending frames since last inference
        self._pending: int = 0
        # Latest CoTracker3 ball position, or None
        self._ct_position: tuple[float, float] | None = None

        # YOLO detection history for velocity extrapolation fallback
        self._det_history: deque[tuple[float, float]] = deque(
            maxlen=self._velocity_history
        )

        # Number of YOLO detections since last CoTracker re-anchor
        self._detections_since_anchor: int = 0

        # General state
        self._last_cx: float | None = None
        self._last_cy: float | None = None
        self._initialized: bool = False  # True once first YOLO detection seen
        self._frames_since_detection: int = 0
        self._last_source: str = "none"

    # ── Model loading (lazy) ──────────────────────────────────────────────────

    def _ensure_predictor(self) -> bool:
        """Lazily load the CoTracker3 model.  Returns *True* on success."""
        if self._predictor is not None:
            return True
        try:
            import torch

            device = torch.device(
                self._device_str if torch.cuda.is_available() else "cpu"
            )
            self._torch_device = device

            if self._checkpoint is not None:
                pred = torch.hub.load(
                    self._HUB_SOURCE, self._hub_model,
                    pretrained=False, trust_repo=True,
                )
                state = torch.load(self._checkpoint, map_location="cpu")
                pred.load_state_dict(state.get("model", state), strict=False)
            else:
                pred = torch.hub.load(
                    self._HUB_SOURCE, self._hub_model, trust_repo=True
                )

            pred = pred.to(device)
            pred.eval()
            self._predictor = pred

            try:
                self._step = int(pred.step)
            except AttributeError:
                self._step = 8
            self._max_buf = 3 * self._step + 1

            logger.info(
                "BallCoTrackerTracker: loaded %s on %s", self._hub_model, device
            )
            return True
        except Exception as exc:
            logger.warning(
                "BallCoTrackerTracker: CoTracker3 unavailable (%s); "
                "will use velocity extrapolation only.",
                exc,
            )
            return False

    # ── Frame / query management ──────────────────────────────────────────────

    def _add_frame(self, frame: np.ndarray) -> None:
        """Convert *frame* (BGR uint8) and append to the internal buffer."""
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        self._frame_buf.append(t)
        if self._hub_model == "cotracker3_online":
            if len(self._frame_buf) > self._max_buf:
                self._frame_buf = self._frame_buf[-self._max_buf:]
        self._pending += 1

    def _set_query(self, cx: float, cy: float) -> None:
        """Register a new query point at (cx, cy) at t=0 of the next window."""
        import torch
        self._queries = torch.tensor(
            [[[0.0, cx, cy]]], dtype=torch.float32,
            device=self._torch_device,
        )
        # Reset CoTracker3 state so the next call is a first-step
        self._ct_initialized = False
        self._pending = 0
        self._ct_position = None
        if self._hub_model == "cotracker3_offline":
            self._frame_buf = []

    def _run_cotracker(self) -> bool:
        """Run inference if enough frames are pending.  Return *True* on success."""
        import torch

        if self._queries is None or self._predictor is None:
            return False

        if self._hub_model == "cotracker3_offline":
            return self._run_offline()

        # Online mode
        need = self._step * 2 if not self._ct_initialized else self._step
        if self._pending < need or len(self._frame_buf) < need:
            return False

        chunk_frames = self._frame_buf[-need:]
        video = torch.stack(chunk_frames, dim=0).unsqueeze(0).to(self._torch_device)

        with torch.no_grad():
            pred_tracks, _ = self._predictor(
                video,
                is_first_step=not self._ct_initialized,
                queries=self._queries if not self._ct_initialized else None,
            )

        # pred_tracks: (1, T, 1, 2) — take last frame, point 0
        pos = pred_tracks[0, -1, 0].cpu().numpy()
        self._ct_position = (float(pos[0]), float(pos[1]))
        self._ct_initialized = True
        self._pending = 0
        return True

    def _run_offline(self) -> bool:
        """Run offline CoTracker3 on the full frame buffer."""
        import torch

        if len(self._frame_buf) < 2:
            return False

        queries_t0 = self._queries.clone()
        queries_t0[0, :, 0] = 0.0  # pin queries to first frame in buffer

        video = torch.stack(self._frame_buf, dim=0).unsqueeze(0).to(
            self._torch_device
        )
        with torch.no_grad():
            pred_tracks, _ = self._predictor(video, queries=queries_t0)

        pos = pred_tracks[0, -1, 0].cpu().numpy()
        self._ct_position = (float(pos[0]), float(pos[1]))
        self._ct_initialized = True
        self._pending = 0
        return True

    # ── Velocity extrapolation (warm-up fallback) ─────────────────────────────

    def _estimate_velocity(self) -> tuple[float, float] | None:
        """Return (vx, vy) from recent detection history, or *None*."""
        hist = list(self._det_history)
        if len(hist) < 2:
            return None
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        n = len(hist)
        dxs = [xs[i + 1] - xs[i] for i in range(n - 1)]
        dys = [ys[i + 1] - ys[i] for i in range(n - 1)]
        return (float(np.median(dxs)), float(np.median(dys)))

    def _extrapolate(self) -> tuple[float, float]:
        """Predict ball position one frame forward via velocity extrapolation."""
        assert self._last_cx is not None and self._last_cy is not None
        vel = self._estimate_velocity()
        if vel is None:
            return float(self._last_cx), float(self._last_cy)
        return float(self._last_cx) + vel[0], float(self._last_cy) + vel[1]

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self, cx: float, cy: float, frame: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Accept a YOLO ball detection and update the CoTracker3 model.

        On the first call the CoTracker3 model is initialised with the YOLO
        position as the query point.  On subsequent calls the query point is
        refreshed whenever *redetect_interval* YOLO-confirmed frames have
        elapsed since the last re-anchor.

        Parameters
        ----------
        cx, cy:
            Ball centre from YOLO (pixel coordinates).
        frame:
            Current BGR uint8 video frame for CoTracker3 update.

        Returns
        -------
        (cx, cy):
            YOLO detection passed through unchanged (zero latency).
        """
        if not self._ensure_predictor():
            # CoTracker3 unavailable — behave like a simple detection tracker
            self._last_cx = cx
            self._last_cy = cy
            self._det_history.append((cx, cy))
            self._initialized = True
            self._frames_since_detection = 0
            self._last_source = "detected"
            return cx, cy

        should_reanchor = (
            not self._initialized
            or self._detections_since_anchor >= self._redetect_interval
        )

        if should_reanchor:
            # Add current frame to buffer BEFORE calling _set_query (which
            # clears the offline buffer), so we always have at least one frame.
            if frame is not None:
                self._add_frame(frame)
            self._set_query(cx, cy)
            self._detections_since_anchor = 0
            if frame is not None and self._hub_model == "cotracker3_offline":
                # _set_query() cleared the buffer; re-add this frame as t=0
                self._add_frame(frame)
            self._run_cotracker()
        else:
            if frame is not None:
                self._add_frame(frame)
            self._run_cotracker()
            self._detections_since_anchor += 1

        # Sync CoTracker3 position to the confirmed YOLO detection
        self._ct_position = (cx, cy)

        self._last_cx = cx
        self._last_cy = cy
        self._det_history.append((cx, cy))
        self._initialized = True
        self._frames_since_detection = 0
        self._last_source = "detected"
        return cx, cy

    def predict(self, frame: np.ndarray | None = None) -> tuple[float, float]:
        """Propagate the ball position when YOLO misses.

        Runs CoTracker3 on the new frame and returns its tracked position.
        Falls back to velocity extrapolation if CoTracker3 has not yet
        produced output (warm-up phase) or if the model is unavailable.

        Parameters
        ----------
        frame:
            Current BGR uint8 video frame.

        Returns
        -------
        (cx, cy):
            Estimated ball position for this frame.
        """
        if not self._initialized or self._last_cx is None:
            raise RuntimeError(
                "BallCoTrackerTracker.predict() called before any YOLO detection."
            )

        self._frames_since_detection += 1

        if frame is not None and self._predictor is not None:
            self._add_frame(frame)
            ran = self._run_cotracker()
            if ran and self._ct_position is not None:
                cx, cy = self._ct_position
                self._last_cx = cx
                self._last_cy = cy
                self._last_source = "cotracker"
                return cx, cy

        # Warm-up fallback: velocity extrapolation
        cx, cy = self._extrapolate()
        self._last_cx = cx
        self._last_cy = cy
        self._last_source = "predicted"
        return cx, cy

    def reset(self) -> None:
        """Return the tracker to its uninitialised state."""
        self._frame_buf = []
        self._queries = None
        self._ct_initialized = False
        self._pending = 0
        self._ct_position = None
        self._det_history.clear()
        self._detections_since_anchor = 0
        self._last_cx = None
        self._last_cy = None
        self._initialized = False
        self._frames_since_detection = 0
        self._last_source = "none"

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        """*True* once the first YOLO detection has been processed."""
        return self._initialized

    @property
    def frames_since_detection(self) -> int:
        """Consecutive frames without a YOLO detection (0 after every detection)."""
        return self._frames_since_detection

    @property
    def last_measurement_gated(self) -> bool:
        """Always *False* — kept for API compatibility with :class:`BallDCFTracker`."""
        return False

    @property
    def laplacian_weight(self) -> float:
        """Always 1.0 — kept for API compatibility with :class:`BallKalmanFilter`."""
        return 1.0

    @property
    def last_source(self) -> str:
        """Source of the most-recent ball position estimate.

        Possible values:

        * ``"detected"``  — position came directly from a YOLO detection.
        * ``"cotracker"`` — CoTracker3 propagation (YOLO missed this frame).
        * ``"predicted"`` — linear velocity extrapolation (warm-up / fallback).
        * ``"none"``      — tracker not yet initialised.
        """
        return self._last_source

    @property
    def position(self) -> tuple[float, float] | None:
        """Current ball ``(cx, cy)``, or *None* if not yet initialized."""
        if not self._initialized or self._last_cx is None:
            return None
        return float(self._last_cx), float(self._last_cy)

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Velocity estimate ``(vx, vy)`` in px/frame from recent detections."""
        return self._estimate_velocity()

    @property
    def predicted_position(self) -> tuple[float, float] | None:
        """Ball position predicted one frame forward (without advancing state).

        Used by :meth:`SegmentationTracker._process_ball` to centre the ROI
        for secondary (FRoG-MOT stage-2) low-confidence YOLO detection.
        Returns *None* if not yet initialized.
        """
        if not self._initialized or self._last_cx is None:
            return None
        if self._ct_position is not None:
            return self._ct_position
        return self._extrapolate()

    @property
    def adaptive_search_radius(self) -> int:
        """Search radius for ROI-based re-detection (pixels).

        Returns a fixed modest value since CoTracker3 provides a good
        position estimate; 80 px covers typical tracking error.
        """
        return 80
