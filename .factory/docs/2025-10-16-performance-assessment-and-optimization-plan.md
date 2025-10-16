## Findings (performance-optimization-specialist view)

- Current strategies in place:
  - Extensive Numba JIT in `src/sippy/utils/compiled_utils.py` with `parallel=True`, `fastmath=True`, `nogil=True`, `cache=True`; SIMD vs parallel variants (e.g., Vn_mat), compiled builders for ARX/FIR/BJ/ARMAX/ARARMAX regressors, compiled simulators for process form, rescaling, and matrix utilities; Joblib parallelism in PARSIM K/S/P simulation sequences; multiprocessing in subspace order selection; widespread preallocation and `np.ascontiguousarray` on hot paths.
  - Algorithms already switch to compiled paths when available (ARX, BJ, subspace/PARSIM use compiled helpers); process-form simulator is compiled; predictor-form sim is still Python.

- Hot paths and gaps:
  - FIR.identify builds Phi with triple nested Python loops (does not use `create_regression_matrix_fir_compiled`) → clear hotspot.
  - Subspace/PARSIM repeatedly call `np.linalg.pinv`/`sqrtm`/`svd`; `pinv_compiled_svd` exists but is not wired in key spots (algorithm_1, PARSIM core M constructions).
  - Predictor-form sim (`ss_lsim_predictor_form`) used heavily by PARSIM K/S/P lacks compiled implementation; joblib helps but time loop remains Python.
  - Repeated large-array copies in order selection via multiprocessing.Pool (pickle overhead) could be reduced.

## Proposed optimization plan (incremental, low-risk)

1) FIR fast path
- Replace manual Phi assembly with `create_regression_matrix_fir_compiled` in FIR.identify for both training and Yid reconstruction (expected 5–15x on nb·nu·N_eff > 1e6).

2) Subspace/PARSIM pseudoinverse
- Swap `np.linalg.pinv` for `pinv_compiled_svd` in: subspace_core.algorithm_1, force_a_stability, PARSIM core M/Gamma_L updates; guard with try/except to fall back on NumPy if needed (1.3–2.0x on ill‑conditioned medium matrices).

3) Predictor-form simulator JIT
- Add `ss_lsim_predictor_form_compiled(A_K,B_K,C,D,K,y,u,x0)` (Numba, SIMD-friendly inner loops) and route PARSIM K/S/P to it (2–4x vs Python, more on long L and n≥8).

4) Memory/layout tightening
- Ensure all regressor/simulation inputs are C-contiguous float64 before compiled calls; avoid `.flatten()` temporary copies by using views; preallocate output buffers and use in-place writes (5–20% wins, reduces GC pressure).

5) Parallel order selection
- Replace `multiprocessing.Pool` with joblib `Parallel(prefer="threads")` where BLAS/Numpy release GIL to avoid pickling large arrays; or share read-only arrays via memoryviews (up to 1.5–2x wall-time improvement for many orders).

6) Random generation in compiled noise
- Optionally vectorize RNG by pre-generating uniform arrays and feeding to Numba loops to reduce per-iteration RNG overhead (10–30% on long signals); keep default behavior unless reproducibility is required.

## Acceptance criteria
- Zero changes in numerical results vs current harold branch (within existing test tolerances); all tests pass.
- Measured speedups on representative sizes (report before/after for FIR MIMO, PARSIM-K, subspace select_order).

If you approve, I’ll implement items 1–3 first (highest impact), then 4–5 as follow-ups, and provide benchmarks.