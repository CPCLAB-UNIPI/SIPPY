"""
Non-Parametric Frequency Domain System Identification

This module implements non-parametric frequency-domain identification using
the correlation method and spectral analysis. The algorithm estimates the
frequency response function G(e^iω) directly from input-output data without
assuming a specific parametric structure.

References:
    - Lecture Notes Chapter 10-11: Non-Parametric Linear System Identification
    - Ljung, L. (1999). System Identification: Theory for the User.
"""

from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any
import warnings
import numpy as np

from ..base import IdentificationAlgorithm, StateSpaceModel
from ...utils.spectral_utils import (
    compute_correlations_fft,
    compute_spectra_from_correlation,
    compute_output_spectrum,
    compute_frequency_response,
    compute_coherence,
    create_hamming_window,
    smooth_frequency_response,
    extract_magnitude_phase,
    validate_signal_pair,
    denormalize_frequency,
    create_window,
)

if TYPE_CHECKING:
    from ..iddata import IDData


class FrequencyDomainIdentification(IdentificationAlgorithm):
    """
    Non-parametric frequency domain system identification using correlation method.

    This algorithm implements the following procedure:
    1. Compute input autocorrelation R_u(τ) and input-output cross-correlation R_uy(τ)
    2. Apply DTFT to obtain power spectrum Φ_u(ω) and cross-spectrum Φ_uy(ω)
    3. Estimate frequency response: G(e^iω) = Φ_uy(ω) / Φ_u(ω)
    4. Apply spectral smoothing (Hamming window) to reduce variance
    5. Compute coherence function γ²(ω) for model validation

    Parameters:
        max_lag: Maximum lag for correlation computation (default: N-1)
        smoothing_window: Size of Hamming window for spectral smoothing (default: 11)
        coherence_threshold: Minimum acceptable coherence for quality assessment (default: 0.8)
        use_welch: Use Welch's method (segment averaging) for improved variance (default: False)
        welch_segments: Number of segments for Welch's method (default: 8)
        welch_overlap: Overlap fraction for Welch's method (default: 0.5)
        window_type: Window function for data tapering ('none', 'hann', 'hamming', 'blackman')
        remove_mean: Remove DC component from signals (default: True)
    """

    def __init__(
        self,
        max_lag: Optional[int] = None,
        smoothing_window: int = 11,
        coherence_threshold: float = 0.8,
        use_welch: bool = False,
        welch_segments: int = 8,
        welch_overlap: float = 0.5,
        window_type: str = "none",
        remove_mean: bool = True,
        **kwargs,
    ):
        """
        Initialize frequency domain identification algorithm.

        Args:
            max_lag: Maximum correlation lag (None = N-1)
            smoothing_window: Hamming window size (odd integer, 5-21 typical)
            coherence_threshold: Minimum γ² for reliable estimate (0.7-0.9 typical)
            use_welch: Enable Welch's method for variance reduction
            welch_segments: Number of segments (more = lower variance, less resolution)
            welch_overlap: Segment overlap fraction (0.5 = 50% overlap)
            window_type: Data window ('none', 'hann', 'hamming', 'blackman')
            remove_mean: Remove DC component before identification
        """
        super().__init__()

        # Validate parameters
        if smoothing_window < 3:
            raise ValueError("smoothing_window must be >= 3")
        if smoothing_window % 2 == 0:
            smoothing_window += 1  # Ensure odd

        if not 0 < coherence_threshold <= 1:
            raise ValueError("coherence_threshold must be in (0, 1]")

        if use_welch:
            if welch_segments < 2:
                raise ValueError("welch_segments must be >= 2")
            if not 0 < welch_overlap < 1:
                raise ValueError("welch_overlap must be in (0, 1)")

        if window_type not in ["none", "hann", "hamming", "blackman"]:
            raise ValueError(f"Unknown window_type: {window_type}")

        # Store parameters
        self.max_lag = max_lag
        self.smoothing_window = smoothing_window
        self.coherence_threshold = coherence_threshold
        self.use_welch = use_welch
        self.welch_segments = welch_segments
        self.welch_overlap = welch_overlap
        self.window_type = window_type
        self.remove_mean = remove_mean

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "FREQUENCY_DOMAIN"

    def validate_parameters(self, **kwargs) -> bool:
        """Validate algorithm-specific parameters."""
        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Perform non-parametric frequency domain identification.

        Args:
            y: Output signal (N samples)
            u: Input signal (N samples)
            iddata: IDData object with input/output data
            **kwargs: Additional parameters (dt: sampling interval)

        Returns:
            StateSpaceModel object with frequency response in identification_info
        """
        # Extract data from IDData if provided
        if iddata is not None:
            u = iddata.get_input_array()
            y = iddata.get_output_array()
            dt = iddata.sample_time
        else:
            dt = kwargs.get("dt", 1.0)

        # Validate and preprocess
        u, y = self._validate_and_preprocess(u, y, dt)
        N = len(u)

        # Set max_lag if not specified
        max_lag = self.max_lag if self.max_lag is not None else N - 1
        max_lag = min(max_lag, N - 1)

        # Perform identification
        if self.use_welch:
            results = self._identify_welch(u, y, dt, max_lag)
        else:
            results = self._identify_standard(u, y, dt, max_lag)

        # Convert to StateSpaceModel for factory compatibility
        # Use identity matrices for state-space (frequency domain is primary output)
        A = np.eye(1)
        B = np.zeros((1, 1))
        C = np.zeros((1, 1))
        D = np.zeros((1, 1))
        K = np.zeros((1, 1))
        Q = np.eye(1)
        R = np.eye(1)
        S = np.zeros((1, 1))
        Vn = 0.0

        model = StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            K=K,
            Q=Q,
            R=R,
            S=S,
            ts=dt,
            Vn=Vn,
            identification_info=results,
        )

        return model

    def _validate_and_preprocess(
        self, u: np.ndarray, y: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate inputs and apply preprocessing.

        Args:
            u: Input signal
            y: Output signal
            dt: Sampling interval

        Returns:
            Preprocessed (u, y) tuple
        """
        # Validate signal pair
        u, y = validate_signal_pair(u, y, min_length=100)

        # Additional dt validation
        if dt <= 0:
            raise ValueError(f"Sampling interval must be positive, got {dt}")

        # Remove DC component if requested
        if self.remove_mean:
            u = u - np.mean(u)
            y = y - np.mean(y)

        # Apply window if requested
        if self.window_type != "none":
            window = create_window(len(u), self.window_type)
            u = u * window
            y = y * window

        return u, y

    def _identify_standard(
        self, u: np.ndarray, y: np.ndarray, dt: float, max_lag: int
    ) -> Dict[str, Any]:
        """
        Standard identification algorithm (5-step procedure).

        Steps:
        1. Compute correlations R_u(τ), R_uy(τ)
        2. Compute spectra Φ_u(ω), Φ_uy(ω), Φ_y(ω)
        3. Estimate frequency response G(e^iω) = Φ_uy(ω) / Φ_u(ω)
        4. Apply spectral smoothing
        5. Compute coherence and quality metrics
        """
        N = len(u)

        # Step 1: Correlation computation (uses spectral_utils)
        tau, R_u, R_uy = compute_correlations_fft(u, y, max_lag)

        # Step 2: Spectral estimation (uses spectral_utils)
        Phi_u, Phi_uy, omega = compute_spectra_from_correlation(R_u, R_uy)
        Phi_y = compute_output_spectrum(y, len(Phi_u))

        # Step 3: Frequency response estimation (uses spectral_utils)
        G_raw = compute_frequency_response(Phi_uy, Phi_u)

        # Step 4: Spectral smoothing (uses spectral_utils)
        window = create_hamming_window(self.smoothing_window, normalize=True)
        G_smooth = smooth_frequency_response(G_raw, window)

        # Step 5: Coherence and quality assessment (uses spectral_utils)
        coherence = compute_coherence(Phi_uy, Phi_u, Phi_y)
        quality_metrics = self._assess_quality(coherence)

        # Extract magnitude and phase (uses spectral_utils)
        magnitude_db, phase_deg = extract_magnitude_phase(G_smooth)

        # Physical frequency vectors (uses spectral_utils)
        omega_real, freq_hz = denormalize_frequency(omega, dt)

        return {
            "method": "FREQUENCY_DOMAIN",
            "frequency_response": {
                "omega": omega,
                "omega_real": omega_real,
                "freq_hz": freq_hz,
                "G_raw": G_raw,
                "G_smooth": G_smooth,
                "magnitude_db": magnitude_db,
                "phase_deg": phase_deg,
                "coherence": coherence,
                "Phi_u": Phi_u,
                "Phi_y": Phi_y,
                "Phi_uy": Phi_uy,
                "R_u": R_u,
                "R_uy": R_uy,
                "tau": tau,
            },
            "quality_metrics": quality_metrics,
        }

    def _identify_welch(
        self, u: np.ndarray, y: np.ndarray, dt: float, max_lag: int
    ) -> Dict[str, Any]:
        """
        Welch's method: Segment averaging for reduced variance.

        Divides data into overlapping segments, computes spectra for each,
        and averages to reduce variance at the cost of frequency resolution.
        """
        N = len(u)

        # Compute segment parameters
        segment_length = N // self.welch_segments
        if segment_length < 100:
            raise ValueError(
                f"Segments too short ({segment_length} samples). "
                f"Reduce welch_segments or increase data length."
            )

        step = int(segment_length * (1 - self.welch_overlap))
        n_segments = (N - segment_length) // step + 1

        # Initialize accumulators
        Phi_u_sum = None
        Phi_uy_sum = None
        Phi_y_sum = None
        omega = None

        # Process each segment
        for i in range(n_segments):
            start = i * step
            end = start + segment_length

            if end > N:
                break

            # Extract segment
            u_seg = u[start:end]
            y_seg = y[start:end]

            # Compute segment correlations (uses spectral_utils)
            tau_seg, R_u_seg, R_uy_seg = compute_correlations_fft(
                u_seg, y_seg, min(max_lag, len(u_seg) - 1)
            )

            # Compute segment spectra (uses spectral_utils)
            Phi_u_seg, Phi_uy_seg, omega_seg = compute_spectra_from_correlation(
                R_u_seg, R_uy_seg
            )
            Phi_y_seg = compute_output_spectrum(y_seg, len(Phi_u_seg))

            # Accumulate
            if Phi_u_sum is None:
                Phi_u_sum = Phi_u_seg
                Phi_uy_sum = Phi_uy_seg
                Phi_y_sum = Phi_y_seg
                omega = omega_seg
                n_valid_segments = 1
            else:
                # Match lengths (take minimum)
                min_len = min(len(Phi_u_sum), len(Phi_u_seg))
                Phi_u_sum = Phi_u_sum[:min_len] + Phi_u_seg[:min_len]
                Phi_uy_sum = Phi_uy_sum[:min_len] + Phi_uy_seg[:min_len]
                Phi_y_sum = Phi_y_sum[:min_len] + Phi_y_seg[:min_len]
                omega = omega_seg[:min_len]
                n_valid_segments += 1

        # Average spectra
        Phi_u = Phi_u_sum / n_valid_segments
        Phi_uy = Phi_uy_sum / n_valid_segments
        Phi_y = Phi_y_sum / n_valid_segments

        # Continue with standard procedure (uses spectral_utils)
        G_raw = compute_frequency_response(Phi_uy, Phi_u)
        window = create_hamming_window(self.smoothing_window, normalize=True)
        G_smooth = smooth_frequency_response(G_raw, window)
        coherence = compute_coherence(Phi_uy, Phi_u, Phi_y)
        quality_metrics = self._assess_quality(coherence)
        magnitude_db, phase_deg = extract_magnitude_phase(G_smooth)

        # For return, use representative correlation from full data (uses spectral_utils)
        tau, R_u, R_uy = compute_correlations_fft(u, y, max_lag)

        omega_real, freq_hz = denormalize_frequency(omega, dt)

        return {
            "method": "FREQUENCY_DOMAIN",
            "frequency_response": {
                "omega": omega,
                "omega_real": omega_real,
                "freq_hz": freq_hz,
                "G_raw": G_raw,
                "G_smooth": G_smooth,
                "magnitude_db": magnitude_db,
                "phase_deg": phase_deg,
                "coherence": coherence,
                "Phi_u": Phi_u,
                "Phi_y": Phi_y,
                "Phi_uy": Phi_uy,
                "R_u": R_u,
                "R_uy": R_uy,
                "tau": tau,
            },
            "quality_metrics": quality_metrics,
        }



    def _assess_quality(self, coherence: np.ndarray) -> Dict[str, Any]:
        """Assess model quality from coherence function."""
        mean_coh = float(np.mean(coherence))
        min_coh = float(np.min(coherence))
        max_coh = float(np.max(coherence))
        median_coh = float(np.median(coherence))

        fraction_reliable = float(np.mean(coherence >= self.coherence_threshold))

        # Overall assessment
        if mean_coh >= 0.9:
            quality_label = "EXCELLENT"
        elif mean_coh >= 0.8:
            quality_label = "GOOD"
        elif mean_coh >= 0.7:
            quality_label = "ACCEPTABLE"
        else:
            quality_label = "POOR"

        return {
            "mean_coherence": mean_coh,
            "min_coherence": min_coh,
            "max_coherence": max_coh,
            "median_coherence": median_coh,
            "fraction_reliable": fraction_reliable,
            "threshold": self.coherence_threshold,
            "quality_label": quality_label,
            "is_reliable": mean_coh >= self.coherence_threshold,
        }


