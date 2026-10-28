# B5a/B5b SPOT and dSPOT implementation record

Versioned implementations: `SPOT-KDD2017-paper-equations-v1.0.0` and
`dSPOT-KDD2017-paper-equations-v1.0.0`, in
`experiments/calibration/spot_dspot.py`.

## Source and implementation scope

The primary algorithm source is Siffer et al., “Anomaly Detection in Streams
with Extreme Value Theory,” KDD 2017, Algorithms 1–3 and Sections 3.4.2 and
4.1–4.3. The inspected `asiffer/libspot` source is pinned at commit
`d57e921a08930b960d07adcbec3a839bd228fe94`; `src/spot.c` is blob
`fdebc8c216fffeea93afcc51efab83a2d6013351` (libspot 3.0.0). That source
contains a SPOT implementation and no dSPOT implementation. The C source was
read but was not compiled for numerical trace parity. The Python code therefore
claims paper-equation implementation and independent reference tests, not
author-code parity. The third-party `cbhua/peak-over-threshold` and
`liwen-tj/SPOT` repositories are not treated as authoritative sources.

## B5a SPOT

1. Consume the first `n_init=1000` causally observed unlabeled scores without
   emitting a decision. They remain in the full-stream evaluation denominator
   as explicitly unavailable predictions.
2. Set the initial high level `u` to the registered NumPy linear empirical
   quantile at `level=0.98`. The interpolation is a reproducibility convention;
   KDD uses 98% as a practical value but does not prescribe this interpolation.
3. Use strict excesses `Y_i = X_i-u` for `X_i>u`. Fail closed if fewer than two
   positive excesses are available.
4. Estimate zero-location GPD shape `xi` and scale `sigma` by Grimshaw's scalar
   likelihood method. For `x=xi/sigma`, solve `g(x)=u(x)v(x)-1=0`, where
   `u(x)=mean(1/(1+xY_i))` and `v(x)=1+mean(log(1+xY_i))`; derive
   `xi=mean(log(1+xY_i))` and `sigma=xi/x`, then select the finite candidate
   with greatest GPD log likelihood. The exponential limit `xi=0` is included.
5. For target tail probability `q`, accepted count `n`, and strict peak count
   `N_u`, the current alarm threshold is
   `z_q = u + (sigma/xi)*(((q*n)/N_u)^(-xi)-1)`. At `xi=0`, use
   `u + sigma*log(N_u/(q*n))`. Registered defaults require `q=0.001 < 1-level`.
6. Predict using the current threshold before observing the score. A strict
   alarm `X_t>z_q` is excluded from both `n` and the peak fit. Every non-alarm
   increments `n`; a strict peak `X_t>u` appends `X_t-u`, refits the GPD, and
   updates `z_q`.

The numerical root search uses the paper-style bounded multi-start
L-BFGS-B least-squared residual objective and a deterministic sign-bracket
supplement to recover roots that the zero-root local minimum can hide. Every
accepted root is rechecked against the scalar equation and likelihood. This
supplement and the empirical interpolation are disclosed adaptations.

## B5b dSPOT

The registered local-mean window is `d=50`. The first `d` raw scores initialize
the window and do not receive predictions. Thereafter, score `X_t` is
detrended by the mean of the previous `d` accepted raw normal observations:
`R_t = X_t - mean(W_{t-1})`. SPOT's initialization and risk-threshold equations
are applied to the causal residual stream. During the `n_init` residual
warm-up, the raw window advances so that the residual history is available;
these predictions are marked unavailable. The reconstructed threshold is
`mean(W_t)+z_q(t)`. After initialization, an alarm is excluded from both POT
state and the moving-mean window. A non-alarm updates POT counts/peaks and
advances the raw-score window.

Since the pinned libspot source has no dSPOT implementation, dSPOT is separately
implemented from the KDD paper algorithm. Its local-window initialization,
alarm exclusion, accepted-normal update and reconstructed threshold have
explicit reference tests. No theorem is claimed for score sequences whose
distribution changes as the shared native memory changes.

## Registered information and warm-up policy

Both methods may consume every causally observed unlabeled score. They ignore
queried labels, protected C64 after initialization, future observations, and
all masks. Predictions precede the current score update. Warm-up events are
retained in the full-stream denominator and reported separately as unavailable;
they are not silently dropped or converted to abstentions. `Prediction.available`
and `prediction_availability` distinguish missing startup predictions from an
available method decision to abstain.

The defaults are fixed Stage 0 implementation parameters, not tuned on
Real-IAD outcomes: `n_init=1000`, `level=0.98`, `risk_q=0.001`, ten Grimshaw
starts, root tolerance `1e-6`, and dSPOT window `50`.

## Reference tests

`tests/calibration/test_spot_dspot.py` independently checks a seeded Grimshaw
fit against `scipy.stats.genpareto.fit`, the paper tail equation and zero-shape
limit, causal warm-up and missingness, strict alarm exclusion, accepted-peak
updates, dSPOT mean-window behavior, insufficient-tail failure, and exact
checkpoint restoration. Engine tests verify that unavailable startup
predictions remain distinct from abstentions through interrupted/resumed
execution and offline metrics. These references support implementation review;
they are not empirical baseline-performance results.

Status: `IMPLEMENTED_PAPER_EQUATIONS_REFERENCE_TESTED`, with the numerical
differences above explicit. B6 remains independently blocked, so the complete
all-baseline and G6/G7 gates remain blocked.
