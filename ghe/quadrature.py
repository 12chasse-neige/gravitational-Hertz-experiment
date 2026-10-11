"""Phase-preserving projection of one real time series at a known frequency.

Trigonometric quadratures identify a complex Fourier coefficient. They cannot
identify individual physical terms at that same frequency without a source
model. Coefficients here are peak amplitudes, not FFT magnitudes or ASD values.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuadratureFit:
    """h(t)=cosine*cos(Omega*t)+sine*sin(Omega*t)+offset, at original times."""

    cosine: float
    sine: float
    offset: float
    fitted: np.ndarray
    residual_rms: float
    method: str
    condition_number: float
    orthogonality_error: float

    @property
    def phasor(self) -> complex:
        """Peak exp(-i Omega t) convention: H=cosine+i*sine."""
        return complex(self.cosine, self.sine)


def project_quadratures(time_s, signal, angular_frequency) -> QuadratureFit:
    """Project a real signal onto sine, cosine and DC without shifting time zero.

    On a uniform grid spanning an integer number of cycles (endpoint excluded),
    sqrt(2)*cos, sqrt(2)*sin and 1 are orthonormal under <a,b>=sum(a*b)/N.
    Direct projection then gives peak coefficients 2*mean(h*cos/sin).
    Otherwise a joint least-squares fit accounts for finite-window and DC
    leakage. Nonuniform increasing timestamps are supported by the same fit;
    each sample has equal weight. Scaling h first avoids losing tiny strains.
    """
    if np.iscomplexobj(time_s) or np.iscomplexobj(signal):
        raise ValueError("Time and signal arrays must be real")
    times = np.asarray(time_s, dtype=float)
    values = np.asarray(signal, dtype=float)
    if (
        times.ndim != 1
        or values.ndim != 1
        or times.shape != values.shape
        or times.size < 3
    ):
        raise ValueError(
            "time_s and signal must be one-dimensional arrays of equal length, with >=3 samples"
        )
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("Time and signal samples must be finite")
    steps = np.diff(times)
    if np.any(steps <= 0):
        raise ValueError("time_s must be strictly increasing")
    if not np.isfinite(angular_frequency) or angular_frequency <= 0:
        raise ValueError("Angular frequency must be finite and positive")
    uniform = np.allclose(steps, steps[0], rtol=1e-8, atol=0)
    if uniform and angular_frequency * steps[0] >= np.pi * (1 - 1e-12):
        raise ValueError(
            "Signal frequency must be strictly below the sampling Nyquist frequency"
        )
    phase = angular_frequency * times
    if not np.all(np.isfinite(phase)):
        raise ValueError("Time-frequency products must be finite")
    basis = np.column_stack(
        (np.sqrt(2) * np.cos(phase), np.sqrt(2) * np.sin(phase), np.ones(times.size))
    )
    gram = basis.T @ basis / times.size
    orthogonality_error = float(np.max(np.abs(gram - np.eye(3))))
    scale = float(np.max(np.abs(values))) or 1.0
    coefficients, _, rank, singular = np.linalg.lstsq(basis, values / scale, rcond=None)
    if rank < 3 or singular[-1] <= singular[0] * 1e-12:
        raise ValueError(
            "Rank-deficient or ill-conditioned sine-cosine-DC fit; use a longer, better-sampled record"
        )
    condition = float(singular[0] / singular[-1])
    if uniform and orthogonality_error <= 1e-12:
        coefficients = basis.T @ (values / scale) / times.size
        method = "orthogonal_projection"
    else:
        method = "least_squares"
    fitted_scaled = basis @ coefficients
    rms = float(
        np.linalg.norm(values / scale - fitted_scaled) / np.sqrt(times.size) * scale
    )
    cosine, sine = np.sqrt(2) * coefficients[:2] * scale
    return QuadratureFit(
        float(cosine),
        float(sine),
        float(coefficients[2] * scale),
        fitted_scaled * scale,
        rms,
        method,
        condition,
        orthogonality_error,
    )
