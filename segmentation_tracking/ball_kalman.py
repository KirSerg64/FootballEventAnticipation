"""
ball_kalman.py
--------------
Constant-velocity Kalman filter for ball centre tracking.

State vector : ``[cx, cy, vcx, vcy]``  (position + velocity in pixels)
Observation  : ``[cx, cy]``

The filter predicts the ball position every frame and updates it when a
YOLO detection is available.  When the ball is temporarily occluded or
moving too fast for YOLO to detect, the predicted position is reported
instead of discarding the ball entirely.

Public API
~~~~~~~~~~
``BallKalmanFilter``
    ``initialize(cx, cy)``  – seed from first detection
    ``predict() → (cx, cy)``  – advance state without an observation
    ``update(cx, cy) → (cx, cy)``  – advance + correct with observation
    ``position``  – current filtered centre, or *None*
    ``initialized``  – *True* once seeded
    ``frames_since_detection``  – consecutive frames without a detection
    ``reset()``  – return to uninitialised state
"""

from __future__ import annotations

import numpy as np


class BallKalmanFilter:
    """Constant-velocity Kalman filter for tracking the football.

    Parameters
    ----------
    process_noise:
        Diagonal value of the process-noise covariance **Q**.
        Increase for faster / more erratic balls; decrease for smoother
        predictions at the cost of slower adaptation to direction changes.
    measurement_noise:
        Diagonal value of the measurement-noise covariance **R**.
        Increase to trust the Kalman prediction more; decrease to follow
        raw detections more closely.
    """

    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 5.0,
    ) -> None:
        dt = 1.0  # inter-frame time step (1 frame)

        # State-transition matrix  x(t+1) = F · x(t)
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement matrix  z = H · x  (observe position only)
        self.H = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float64,
        )

        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # State estimate and covariance – reset lazily on first update
        self.x: np.ndarray | None = None
        self.P: np.ndarray = np.eye(4, dtype=np.float64) * 100.0
        self._initialized = False
        self._frames_since_detection: int = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def initialize(self, cx: float, cy: float) -> None:
        """Seed the filter from the first observation."""
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self._initialized = True
        self._frames_since_detection = 0

    def predict(self) -> tuple[float, float]:
        """Propagate state one step ahead without a new measurement.

        Returns
        -------
        (cx, cy):
            Predicted ball centre in pixel coordinates.

        Raises
        ------
        RuntimeError
            If called before :meth:`initialize` or :meth:`update`.
        """
        if not self._initialized:
            raise RuntimeError(
                "BallKalmanFilter.initialize() must be called before predict()."
            )
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self._frames_since_detection += 1
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        """Advance state and correct with a new measurement.

        If the filter has not been initialised yet, the state is seeded
        from ``(cx, cy)`` and the raw position is returned.

        Returns
        -------
        (cx, cy):
            Filtered ball centre in pixel coordinates.
        """
        if not self._initialized:
            self.initialize(cx, cy)
            return cx, cy

        # Predict step
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Correction step
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self.H @ self.x                        # innovation
        S = self.H @ self.P @ self.H.T + self.R        # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float64) - K @ self.H) @ self.P
        self._frames_since_detection = 0
        return float(self.x[0]), float(self.x[1])

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        """*True* once :meth:`initialize` or the first :meth:`update` has run."""
        return self._initialized

    @property
    def position(self) -> tuple[float, float] | None:
        """Current filtered ``(cx, cy)``, or *None* if not yet initialized."""
        if not self._initialized or self.x is None:
            return None
        return float(self.x[0]), float(self.x[1])

    @property
    def frames_since_detection(self) -> int:
        """Consecutive frames without a raw ball detection passed to :meth:`update`."""
        return self._frames_since_detection

    def reset(self) -> None:
        """Return the filter to its uninitialised state."""
        self.x = None
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self._initialized = False
        self._frames_since_detection = 0
