## Investigation Results

### Overlap Analysis

**`get_model_uncertainty` (simulation_utils.py):**
- Computes input power spectrum via `scipy.signal.welch(u)`
- Computes output power spectrum via `scipy.signal.welch(y)`  
- Computes cross-spectral density via `scipy.signal.csd(u, y)`
- Estimates frequency response: `G = Pxy / Pxx`
- Applies Hamming window smoothing (bidirectional convolution)
- Computes SNR = Pyy / Pyy_err
- Computes 95% and 68% confidence intervals

**`FrequencyDomainIdentification` (frequency_domain.py):**
- Computes correlations manually via FFT (autocorr, cross-corr)
- Computes power spectra from correlations: `Phi_u`, `Phi_y`
- Computes cross-spectrum from correlations: `Phi_uy`
- Estimates frequency response: `G = Phi_uy / Phi_u`
- Applies Hamming window smoothing (convolution)
- Computes coherence: `γ² = |Phi_uy|² / (Phi_u * Phi_y)`
- Computes quality metrics (mean, min, max, fraction_reliable)

### Common Operations
1. ✅ Power spectrum computation (Welch's method)
2. ✅ Cross-spectrum computation
3. ✅ Frequency response estimation from spectra
4. ✅ Hamming window creation & normalization
5. ✅ Spectral smoothing via Hamming window convolution
6. ✅ Quality/confidence assessment

### Unification Plan

**Phase 1: Create `spectral_utils.py`**
- Extract shared spectral analysis functions
- No breaking changes to existing code

**Phase 2: Common Functions**
1. `compute_power_spectrum(signal, dt=1.0, method='welch', nperseg=1024)` - Wrapper around scipy.signal.welch
2. `compute_cross_spectrum(u, y, dt=1.0, method='welch', nperseg=1024)` - Wrapper around scipy.signal.csd
3. `compute_correlations(u, y, max_lag=None)` - FFT-based correlation (from frequency_domain)
4. `compute_frequency_response(cross_spectrum, input_spectrum)` - Frequency response estimation
5. `create_hamming_window(window_size, normalize=True)` - Shared window creation
6. `smooth_frequency_response(G, window)` - Shared smoothing (magnitude/phase separate)
7. `compute_coherence(cross_spectrum, input_spectrum, output_spectrum)` - Coherence computation

**Phase 3: Refactor Implementations**
- `get_model_uncertainty`: Use `compute_power_spectrum`, `compute_cross_spectrum`, `compute_frequency_response`, `smooth_frequency_response`
- `FrequencyDomainIdentification`: Use shared functions, keep FFT-based correlation approach for accuracy
- Both use shared window/smoothing utilities

**Phase 4: Optional Numba Optimization**
- Add compiled versions in `compiled_utils.py` for:
  - `compute_correlations` (FFT-based)
  - `smooth_frequency_response` (convolution)
  - `compute_coherence` (vectorized operations)

### Expected Benefits
- ✅ DRY principle: Single source of truth for spectral analysis
- ✅ Consistency: Same algorithms across both functions
- ✅ Maintainability: Bug fixes/improvements benefit both
- ✅ Testability: Shared utilities can be independently tested
- ✅ Extensibility: Other algorithms can reuse utilities
- ✅ Performance: Centralized Numba optimization opportunity
- ✅ No breaking changes: Existing APIs unchanged

### Implementation Strategy
1. Create `spectral_utils.py` with new functions
2. Refactor `get_model_uncertainty` to use utilities (backward compatible)
3. Refactor `FrequencyDomainIdentification` to use utilities
4. Add comprehensive tests for spectral utilities
5. Optional: Add Numba compilation for bottlenecks