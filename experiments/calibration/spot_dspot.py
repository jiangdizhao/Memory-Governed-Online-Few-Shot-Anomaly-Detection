"""Paper-equation SPOT and dSPOT streaming threshold baselines.

Primary algorithm reference: Siffer et al., KDD 2017, Algorithms 1--3 and
Sections 3.4.2, 4.1--4.3. The Grimshaw scalar likelihood search follows the
paper's bounded-interval, multi-start L-BFGS-B procedure. These methods use the
entire causally observed unlabelled score stream and do not consume feedback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq, minimize

from .protocol import Observation, Prediction


SPOT_VERSION = "SPOT-KDD2017-paper-equations-v1.0.0"
DSPOT_VERSION = "dSPOT-KDD2017-paper-equations-v1.0.0"


@dataclass(frozen=True)
class POTConfig:
    n_init: int = 1000
    level: float = 0.98
    risk_q: float = 0.001
    grimshaw_starts: int = 10
    root_tolerance: float = 1e-6
    version: str = SPOT_VERSION

    def __post_init__(self) -> None:
        if self.n_init < 3:
            raise ValueError("SPOT initialization requires n_init >= 3")
        if not 0.0 < self.level < 1.0 or not 0.0 < self.risk_q < 1.0 - self.level:
            raise ValueError("Require 0 < q < 1-level < 1")
        if self.grimshaw_starts < 2 or self.root_tolerance <= 0:
            raise ValueError("Invalid Grimshaw numerical-search settings")


@dataclass(frozen=True)
class GPDParameters:
    shape: float
    scale: float
    log_likelihood: float
    roots: tuple[float, ...]
    method: str = "Grimshaw scalar likelihood; exponential x=0 candidate"


def _grimshaw_w(x: float, excesses: np.ndarray) -> float:
    factors = 1.0 + float(x) * excesses
    if np.any(factors <= 0.0) or not np.isfinite(factors).all():
        return math.inf
    u = float(np.mean(1.0 / factors))
    v = 1.0 + float(np.mean(np.log1p(float(x) * excesses)))
    return u * v - 1.0


def _gpd_log_likelihood(excesses: np.ndarray, shape: float,
                        scale: float) -> float:
    if not math.isfinite(shape) or not math.isfinite(scale) or scale <= 0:
        return -math.inf
    if abs(shape) < 1e-10:
        return float(-len(excesses) * math.log(scale) -
                     np.sum(excesses) / scale)
    support = 1.0 + shape * excesses / scale
    if np.any(support <= 0.0) or not np.isfinite(support).all():
        return -math.inf
    return float(-len(excesses) * math.log(scale) -
                 (1.0 + 1.0 / shape) * np.log(support).sum())


def grimshaw_fit(excesses: Iterable[float], *, starts: int = 10,
                 root_tolerance: float = 1e-6) -> GPDParameters:
    """Fit a zero-location GPD with the paper's Grimshaw likelihood method.

    The paper's scalar root is ``u(x)v(x)=1`` for
    ``x=shape/scale``. Roots are searched in the two paper intervals by
    minimizing the sum of squared root residuals from ``starts`` distributed
    initial values, and candidates are ranked by the original GPD log
    likelihood. The always-admissible exponential limit is included at x=0.
    """
    y = np.asarray(tuple(float(x) for x in excesses), dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("Grimshaw requires at least two finite positive excesses")
    minimum, maximum, mean = float(y.min()), float(y.max()), float(y.mean())
    if minimum <= 0 or maximum <= 0 or mean <= 0:
        raise ValueError("Grimshaw excess domain must be positive")

    eps = min(1e-8, 0.5 / maximum)
    intervals = [(-1.0 / maximum + eps, -eps),
                 (eps, 2.0 * (mean - minimum) / (minimum * minimum))]
    roots: list[float] = []

    def add_root(root: float) -> None:
        residual = _grimshaw_w(float(root), y)
        nonzero_floor = max(eps * 4.0, 1e-10 / mean)
        if (abs(float(root)) <= nonzero_floor or not math.isfinite(residual) or
                abs(residual) > root_tolerance):
            return
        if all(abs(float(root) - old) > 1e-7 * max(1.0, abs(old))
               for old in roots):
            roots.append(float(root))

    for lower, upper in intervals:
        if not (math.isfinite(lower) and math.isfinite(upper)) or upper <= lower:
            continue
        if upper <= 0.0:
            # Log-spaced distances from the universal root at zero retain
            # resolution near zero while covering the full negative interval.
            initial = -np.geomspace(max(eps, np.finfo(np.float64).eps),
                                    max(abs(lower), eps), starts)[::-1]
        elif lower >= 0.0:
            initial = np.geomspace(max(eps, np.finfo(np.float64).eps),
                                   max(upper, eps), starts)
        else:
            initial = np.linspace(lower, upper, starts, dtype=np.float64)

        def objective(values: np.ndarray) -> float:
            residuals = [_grimshaw_w(float(x), y) for x in values]
            if not np.isfinite(residuals).all():
                return 1e100
            return float(np.dot(residuals, residuals))

        result = minimize(
            objective, initial, method="L-BFGS-B",
            bounds=[(lower, upper)] * starts,
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000,
                     "maxls": 50})
        for root in np.asarray(result.x, dtype=np.float64):
            add_root(float(root))

        # L-BFGS-B is the paper's specified multi-start search. A deterministic
        # log-spaced sign-bracket pass supplements it where a near-zero local
        # minimum masks a distinct Grimshaw root; roots are still accepted only
        # after the paper's scalar equation and likelihood checks.
        if upper <= 0.0:
            grid = -np.geomspace(max(eps, np.finfo(np.float64).eps),
                                 max(abs(lower), eps), 513)[::-1]
        elif lower >= 0.0:
            grid = np.geomspace(max(eps, np.finfo(np.float64).eps),
                                max(upper, eps), 513)
        else:
            grid = np.linspace(lower, upper, 513)
        prior_x, prior_w = float(grid[0]), _grimshaw_w(float(grid[0]), y)
        for x in grid[1:]:
            current_x = float(x)
            current_w = _grimshaw_w(current_x, y)
            if (math.isfinite(prior_w) and math.isfinite(current_w) and
                    prior_w * current_w < 0.0):
                try:
                    add_root(brentq(lambda value: _grimshaw_w(value, y),
                                    prior_x, current_x, xtol=1e-14,
                                    rtol=1e-13, maxiter=1000))
                except ValueError:
                    pass
            prior_x, prior_w = current_x, current_w

    # x=0 has the exponential-limit MLE shape=0, scale=mean(excess).
    candidates: list[tuple[float, float, float]] = [
        (_gpd_log_likelihood(y, 0.0, mean), 0.0, mean)]
    for root in roots:
        mean_log = float(np.mean(np.log1p(root * y)))
        if not math.isfinite(mean_log):
            continue
        if abs(root) < 1e-12:
            shape, scale = 0.0, mean
        else:
            shape, scale = mean_log, mean_log / root
        ll = _gpd_log_likelihood(y, shape, scale)
        if math.isfinite(ll):
            candidates.append((ll, shape, scale))
    ll, shape, scale = max(candidates, key=lambda item: item[0])
    if not math.isfinite(ll) or not math.isfinite(shape) or scale <= 0:
        raise ValueError("Grimshaw likelihood search found no finite GPD fit")
    return GPDParameters(float(shape), float(scale), float(ll), tuple(roots))


def pot_threshold(base: float, shape: float, scale: float, *,
                  accepted_count: int, excess_count: int, risk_q: float) -> float:
    """KDD Eq. (1), using accepted count k and observed peak count N_t."""
    if accepted_count <= 0 or excess_count <= 0 or scale <= 0 or risk_q <= 0:
        raise ValueError("POT threshold needs positive counts, scale, and risk")
    ratio = risk_q * accepted_count / excess_count
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("POT risk must be below the observed tail fraction N_t/k")
    if abs(shape) < 1e-8:
        threshold = base + scale * math.log(1.0 / ratio)
    else:
        threshold = base + (scale / shape) * (ratio ** (-shape) - 1.0)
    if not math.isfinite(threshold) or threshold <= base:
        raise ValueError("POT threshold is non-finite or below its excess base")
    return float(threshold)


class _SPOTCore:
    def __init__(self, config: POTConfig):
        self.config = config
        self.initial: list[float] = []
        self.peaks: list[float] = []
        self.n = 0
        self.threshold_base: float | None = None
        self.threshold: float | None = None
        self.shape: float | None = None
        self.scale: float | None = None
        self.log_likelihood: float | None = None
        self.accepted_updates = 0
        self.alarms = 0
        self.threshold_refits = 0

    @property
    def ready(self) -> bool:
        return self.threshold is not None

    def observe(self, value: float) -> str:
        x = float(value)
        if not math.isfinite(x):
            raise ValueError("SPOT received a non-finite score")
        if not self.ready:
            self.initial.append(x)
            if len(self.initial) == self.config.n_init:
                self._initialize(self.initial)
            return "INITIALIZATION"

        assert self.threshold is not None and self.threshold_base is not None
        if x > self.threshold:  # KDD Alg. 2: alarm is excluded from all updates.
            self.alarms += 1
            return "ANOMALY_EXCLUDED"
        self.n += 1
        if x > self.threshold_base:  # strict KDD paper peak convention
            self.peaks.append(x - self.threshold_base)
            self._refit_threshold()
            self.accepted_updates += 1
            return "ACCEPTED_PEAK"
        return "NORMAL"

    def _initialize(self, values: Iterable[float]) -> None:
        data = np.asarray(tuple(values), dtype=np.float64)
        if len(data) != self.config.n_init or not np.isfinite(data).all():
            raise ValueError("Invalid SPOT initial score batch")
        # The paper specifies a high empirical quantile and gives 98% as the
        # practical convention. NumPy's linear empirical interpolation is the
        # registered implementation convention for reproducibility.
        base = float(np.quantile(data, self.config.level, method="linear"))
        peaks = (data[data > base] - base).tolist()
        if len(peaks) < 2:
            raise ValueError("BLOCKED_INSUFFICIENT_TAIL: fewer than two strict peaks")
        fit = grimshaw_fit(peaks, starts=self.config.grimshaw_starts,
                           root_tolerance=self.config.root_tolerance)
        self.threshold_base = base
        self.initial = [float(x) for x in data]
        self.peaks = [float(v) for v in peaks]
        self.n = len(data)
        self.shape, self.scale = fit.shape, fit.scale
        self.log_likelihood = fit.log_likelihood
        self.threshold = pot_threshold(
            base, fit.shape, fit.scale, accepted_count=self.n,
            excess_count=len(self.peaks), risk_q=self.config.risk_q)
        self.threshold_refits += 1

    def _refit_threshold(self) -> None:
        assert self.threshold_base is not None
        fit = grimshaw_fit(self.peaks, starts=self.config.grimshaw_starts,
                           root_tolerance=self.config.root_tolerance)
        self.shape, self.scale = fit.shape, fit.scale
        self.log_likelihood = fit.log_likelihood
        self.threshold = pot_threshold(
            self.threshold_base, fit.shape, fit.scale,
            accepted_count=self.n, excess_count=len(self.peaks),
            risk_q=self.config.risk_q)
        self.threshold_refits += 1

    def state_dict(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "initial": list(self.initial),
                "peaks": list(self.peaks), "n": self.n,
                "threshold_base": self.threshold_base,
                "threshold": self.threshold, "shape": self.shape,
                "scale": self.scale, "log_likelihood": self.log_likelihood,
                "accepted_updates": self.accepted_updates,
                "alarms": self.alarms, "threshold_refits": self.threshold_refits}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("config") != asdict(self.config):
            raise ValueError("SPOT checkpoint config/version mismatch")
        initial = [float(x) for x in state.get("initial", ())]
        peaks = [float(x) for x in state.get("peaks", ())]
        n = int(state.get("n", -1))
        if len(initial) > self.config.n_init or n < 0 or any(
                not math.isfinite(x) for x in (*initial, *peaks)) or any(x <= 0 for x in peaks):
            raise ValueError("Invalid SPOT checkpoint state")
        self.initial, self.peaks, self.n = initial, peaks, n
        for name in ("threshold_base", "threshold", "shape", "scale", "log_likelihood"):
            value = state.get(name)
            setattr(self, name, None if value is None else float(value))
        self.accepted_updates = int(state.get("accepted_updates", 0))
        self.alarms = int(state.get("alarms", 0))
        self.threshold_refits = int(state.get("threshold_refits", 0))
        if self.ready != (len(initial) == self.config.n_init):
            raise ValueError("SPOT checkpoint initialization status is inconsistent")
        if self.ready and (self.shape is None or self.scale is None or
                           self.threshold_base is None or self.n < self.config.n_init):
            raise ValueError("Ready SPOT checkpoint lacks fitted tail state")


class SPOTController:
    """B5a: original stationary SPOT, with explicit whole-stream warm-up."""
    name = "B5a"

    def __init__(self, config: POTConfig | None = None):
        self.config = config or POTConfig()
        if self.config.version != SPOT_VERSION:
            raise ValueError("B5a requires its registered source-faithful version")
        self.core = _SPOTCore(self.config)
        self.observations = 0

    @property
    def theta(self) -> float | None:
        return self.core.threshold

    def predict(self, observation: Observation) -> Prediction:
        if not self.core.ready:
            return Prediction(None, None, "WARMUP_UNAVAILABLE", {"method": "SPOT"},
                              available=False)
        assert self.core.threshold is not None
        return Prediction(int(observation.score > self.core.threshold),
                          self.core.threshold, "SPOT_STRICT_UPPER_TAIL")

    def observe_score(self, observation: Observation) -> None:
        self.core.observe(observation.score)
        self.observations += 1

    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Any) -> None:
        # SPOT is score-only and ignores all queried-label feedback.
        return None

    def on_memory_change(self, evidence: Any) -> None:
        return None

    def finalize_step(self) -> float | None:
        return (float(self.core.threshold) if self.core.threshold is not None
                else None)

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": SPOT_VERSION,
                "observations": self.observations, "core": self.core.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("name") != self.name or state.get("version") != SPOT_VERSION:
            raise ValueError("B5a checkpoint identity mismatch")
        self.core.load_state_dict(state["core"])
        self.observations = int(state.get("observations", -1))
        if self.observations < len(self.core.initial):
            raise ValueError("SPOT checkpoint observation count is inconsistent")


class DSPOTController:
    """B5b: paper dSPOT with the accepted-normal moving-mean window."""
    name = "B5b"

    def __init__(self, config: POTConfig | None = None, *, window: int = 50):
        self.config = config or POTConfig(version=DSPOT_VERSION)
        if self.config.version != DSPOT_VERSION or window < 1:
            raise ValueError("B5b requires registered dSPOT version and d>=1")
        self.window = int(window)
        self.core = _SPOTCore(self.config)
        self.local_window: list[float] = []
        self.initial_residuals: list[float] = []
        self.observations = 0
        self.alarms = 0

    @property
    def theta(self) -> float | None:
        if not self.core.ready or not self.local_window or self.core.threshold is None:
            return None
        return float(sum(self.local_window) / len(self.local_window) + self.core.threshold)

    def predict(self, observation: Observation) -> Prediction:
        threshold = self.theta
        if threshold is None:
            return Prediction(None, None, "WARMUP_UNAVAILABLE", {"method": "dSPOT"},
                              available=False)
        return Prediction(int(observation.score > threshold), threshold,
                          "DSPOT_STRICT_RESIDUAL_UPPER_TAIL")

    def observe_score(self, observation: Observation) -> None:
        value = float(observation.score)
        if not math.isfinite(value):
            raise ValueError("dSPOT received a non-finite score")
        self.observations += 1
        if len(self.local_window) < self.window:
            # Algorithm 3 starts from d values before computing residuals.
            self.local_window.append(value)
            return
        baseline = float(sum(self.local_window) / self.window)
        residual = value - baseline
        if not self.core.ready:
            # KDD Alg. 3 warm-up computes n residuals and slides raw values
            # without filtering because its first threshold is not yet known.
            self.initial_residuals.append(residual)
            self.local_window = [*self.local_window[1:], value]
            if len(self.initial_residuals) == self.config.n_init:
                self.core._initialize(self.initial_residuals)
            return
        threshold = self.core.threshold
        assert threshold is not None
        if residual > threshold:
            # Alarm: neither POT counters nor the accepted-normal window move.
            self.core.alarms += 1
            self.alarms += 1
            return
        self.core.n += 1
        if residual > float(self.core.threshold_base):
            self.core.peaks.append(residual - float(self.core.threshold_base))
            self.core._refit_threshold()
            self.core.accepted_updates += 1
        self.local_window = [*self.local_window[1:], value]

    def observe_feedback(self, observation: Observation, prediction: Prediction,
                         feedback: Any) -> None:
        return None

    def on_memory_change(self, evidence: Any) -> None:
        return None

    def finalize_step(self) -> float | None:
        value = self.theta
        return float(value) if value is not None else None

    def state_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": DSPOT_VERSION,
                "config": asdict(self.config), "window": self.window,
                "core": self.core.state_dict(),
                "local_window": list(self.local_window),
                "initial_residuals": list(self.initial_residuals),
                "observations": self.observations, "alarms": self.alarms}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state.get("name") != self.name or state.get("version") != DSPOT_VERSION or
                state.get("config") != asdict(self.config) or
                state.get("window") != self.window):
            raise ValueError("B5b checkpoint identity/config mismatch")
        self.core.load_state_dict(state["core"])
        self.local_window = [float(x) for x in state.get("local_window", ())]
        self.initial_residuals = [float(x) for x in state.get("initial_residuals", ())]
        self.observations = int(state.get("observations", -1))
        self.alarms = int(state.get("alarms", 0))
        if (len(self.local_window) > self.window or
                len(self.initial_residuals) > self.config.n_init or
                self.observations < len(self.local_window) + len(self.initial_residuals)):
            raise ValueError("Invalid dSPOT checkpoint state")
