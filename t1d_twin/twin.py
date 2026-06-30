"""
Common ``Twin`` interface for all twinning methods.

Every twinning method (MCMC, SBI, NN state-space) produces a ``Twin`` subclass
that implements this ABC. The interface has three purposes:

1. **Prediction** — ``predict_ig`` simulates the IG trajectory under a new
   therapy, returning a point estimate and (for posterior-based methods) an
   uncertainty band.

2. **CGM generation** — ``generate_cgm`` samples IG at the CGM observation
   times and applies the simglucose sensor noise model so twin output is
   directly comparable to ground-truth CGM.

3. **Convenience** — ``replay_run`` is a one-call wrapper that pulls inputs
   from a ``RunResult`` and returns a predicted CGM aligned to that run's
   samples; ``summary`` returns a human-readable parameter summary.

Design decisions (from IMPLEMENTATION_PLAN.md §0)
--------------------------------------------------
* Twin CGM = twin IG passed through simglucose's ``CGMSensor`` model (method
  ``add_cgm_noise`` in ``sensor.py``), so twin output is directly comparable to
  ground-truth CGM. Gaussian fallback available for likelihood fitting.
* ``predict_ig`` always returns ``(ig_median, ig_iqr)``; subclasses with a
  posterior also expose the full ensemble via an optional ``n_samples`` kwarg.
* ``replay_run`` uses ``run.replaybg_inputs(dt)`` to get the dt-grid inputs
  and ``run.sample_time`` for the CGM sample rate — no unit conversion needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .replaybg_model import sample_indices
from .sensor import add_cgm_noise, gaussian_noise
from .simglucose_adapter import RunResult


class Twin(ABC):
    """
    Abstract base class for a personalized glucose-dynamics digital twin.

    Subclasses must implement ``predict_ig`` and ``summary``.  All other
    methods are provided here using those two primitives plus the ``Ib``,
    ``sample_time``, ``dt``, and ``sensor_name`` attributes that every
    subclass must set.

    Required attributes (set by ``__init__`` of each subclass)
    -----------------------------------------------------------
    Ib : float
        Basal insulin rate [mU/kg/min] used for steady-state initial
        conditions.
    sample_time : float
        CGM sampling interval [min] (typically 3.0 for Dexcom).
    dt : float
        ODE integration step [min] (typically 1.0).
    sensor_name : str
        simglucose sensor name used for CGM noise (default ``"Dexcom"``).
    """

    # Subclasses set these in __init__; listed here for documentation.
    Ib: float
    sample_time: float
    dt: float
    sensor_name: str

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def predict_ig(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate IG under a new therapy.

        Parameters
        ----------
        insulin : (T,) insulin rate [mU/kg/min] on the dt-grid.
        cho : (T,) CHO rate [mg/kg/min] on the dt-grid.
        n_samples : optional; for posterior-based twins draw this many
            samples from the posterior (ignored by point-estimate twins).

        Returns
        -------
        ig_median : (T,) posterior-median (or single-point) IG [mg/dL].
        ig_iqr : (2, T) [25th, 75th] percentile band [mg/dL].
            Point-estimate twins return ``np.tile(ig_median, (2, 1))``.
        """

    @abstractmethod
    def summary(self) -> dict:
        """
        Human-readable parameter summary.

        Returns
        -------
        dict mapping parameter name -> {``"median"``: float, ``"ci95"``: (lo, hi)}.
        For point-estimate twins ``ci95`` equals ``(value, value)``.
        """

    # ------------------------------------------------------------------
    # Concrete methods — provided by the base class
    # ------------------------------------------------------------------

    def generate_cgm(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Simulate CGM from the twin under a new therapy.

        If ``add_noise`` is ``False`` the sensor stage is skipped entirely and
        the noise-free IG sampled at the CGM observation times is returned. This
        is the signal used for the decision-quality (reward / RMSE / MARD)
        comparison, which is run on IG vs IG with no sensor noise on either
        side. ``add_noise`` only applies to the point-prediction path
        (``n_samples is None``); the ensemble path always adds noise.

        Runs ``predict_ig``, extracts IG at the CGM sample times, then adds
        sensor noise via simglucose's ``CGMSensor`` (or Gaussian fallback).

        Parameters
        ----------
        insulin : (T,) insulin rate [mU/kg/min] on the dt-grid.
        cho : (T,) CHO rate [mg/kg/min] on the dt-grid.
        seed : RNG / sensor seed for reproducible noise.
        use_sensor_model : if ``True`` (default) use ``add_cgm_noise``
            (simglucose sensor); otherwise use ``gaussian_noise``.
        sigma_fallback : std for the Gaussian fallback [mg/dL].
        n_samples : passed to ``predict_ig``; if given returns
            ``(n_samples, n_cgm)`` CGM ensemble (one noisy trace per sample,
            each with an independently seeded sensor).

        Returns
        -------
        cgm : (n_cgm,) predicted CGM [mg/dL], or ``(n_samples, n_cgm)``
            if ``n_samples`` is given.
        """
        T = len(insulin)
        n_cgm = T // int(round(self.sample_time / self.dt))
        obs_idx = sample_indices(n_cgm, self.sample_time, self.dt)

        if n_samples is None:
            ig_median, _ = self.predict_ig(insulin, cho, n_samples=None)
            ig_obs = ig_median[obs_idx]
            if not add_noise:
                return ig_obs                      # noise-free IG at CGM times
            if use_sensor_model:
                return add_cgm_noise(ig_obs, seed=seed,
                                     sensor_name=self.sensor_name,
                                     sample_time=self.sample_time)
            else:
                return gaussian_noise(ig_obs, sigma=sigma_fallback, seed=seed)
        else:
            # Return an ensemble: one noisy trace per posterior sample.
            # Posterior twins override _ig_ensemble to return actual draws;
            # point-estimate twins fall back to a degenerate (tiled) ensemble.
            ig_ens = self._ig_ensemble(insulin, cho, n_samples)  # (n_samples, T)
            ig_obs_ens = ig_ens[:, obs_idx]                     # (n_samples, n_cgm)
            cgm_ens = np.empty_like(ig_obs_ens)
            for i in range(n_samples):
                if use_sensor_model:
                    cgm_ens[i] = add_cgm_noise(
                        ig_obs_ens[i], seed=seed + i,
                        sensor_name=self.sensor_name,
                        sample_time=self.sample_time,
                    )
                else:
                    cgm_ens[i] = gaussian_noise(
                        ig_obs_ens[i], sigma=sigma_fallback, seed=seed + i,
                    )
            return cgm_ens

    def _ig_ensemble(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Return a (n_samples, T) IG ensemble.

        Default: call ``predict_ig(n_samples=n_samples)`` and treat the median
        as a degenerate ensemble. Posterior twins should override this to return
        the actual posterior draws.
        """
        ig_median, _ = self.predict_ig(insulin, cho, n_samples=n_samples)
        return np.tile(ig_median, (n_samples, 1))

    def generate_cgm_band(
        self,
        insulin: np.ndarray,
        cho: np.ndarray,
        n_samples: int = 200,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardized CGM-scale prediction: median + 25-75%% band.

        This is the canonical object for a CGM-vs-CGM comparison across *all*
        twinning methods. It draws an IG ensemble, samples each draw at the CGM
        observation times, pushes each through the UVA/Padova sensor model
        (``add_cgm_noise``) with an independent noise realization, and returns
        percentile statistics on the resulting CGM ensemble.

        * Posterior twins (MCMC, SBI): the band reflects parameter/initial-state
          uncertainty *plus* sensor noise.
        * Point-estimate twins (NN, ConstantTheta): ``_ig_ensemble`` is
          degenerate, so the band reflects sensor noise alone -- still a valid
          CGM band, directly comparable to observed CGM.

        Parameters
        ----------
        insulin, cho : (T,) input rates on the dt-grid.
        n_samples : ensemble size used to estimate the band.
        seed : base seed; draw ``i`` uses sensor seed ``seed + i`` so noise
            realizations are independent across the ensemble and (by choosing a
            different base seed) independent of the ground-truth CGM.
        use_sensor_model : UVA/Padova sensor (default) vs Gaussian fallback.
        sigma_fallback : std for the Gaussian fallback [mg/dL].

        Returns
        -------
        t_cgm_min : (n_cgm,) CGM observation times [minutes].
        cgm_median : (n_cgm,) per-time median of the CGM ensemble [mg/dL].
        cgm_iqr : (2, n_cgm) [25th, 75th] percentile band [mg/dL].
        """
        cgm_ens = self.generate_cgm(
            insulin, cho,
            seed=seed,
            use_sensor_model=use_sensor_model,
            sigma_fallback=sigma_fallback,
            n_samples=n_samples,
        )                                            # (n_samples, n_cgm)
        cgm_median = np.median(cgm_ens, axis=0)
        cgm_iqr = np.percentile(cgm_ens, [25.0, 75.0], axis=0)

        n_cgm = cgm_ens.shape[1]
        t_cgm_min = (np.arange(n_cgm) + 1) * self.sample_time
        return t_cgm_min, cgm_median, cgm_iqr

    def replay_run(
        self,
        run: RunResult,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """
        Predict CGM (or noise-free IG) for the therapy recorded in ``run``.

        With ``add_noise=False`` returns the twin's posterior-median IG sampled
        at the run's CGM times -- the signal compared against the plant's
        noise-free IG (``RunResult.bg()``) for the decision-quality metrics.
        """
        _, insulin, cho = run.replaybg_inputs(dt=self.dt)
        return self.generate_cgm(
            insulin, cho,
            seed=seed,
            use_sensor_model=use_sensor_model,
            sigma_fallback=sigma_fallback,
            add_noise=add_noise,
        )

    def replay_run_band(
        self,
        run: RunResult,
        n_samples: int = 200,
        seed: int = 1,
        use_sensor_model: bool = True,
        sigma_fallback: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ``generate_cgm_band`` for the therapy recorded in ``run``.

        Returns ``(t_cgm_min, cgm_median, cgm_iqr)`` aligned to ``run.cgm()``,
        ready to overlay against the observed CGM for a CGM-vs-CGM comparison.
        """
        _, insulin, cho = run.replaybg_inputs(dt=self.dt)
        return self.generate_cgm_band(
            insulin, cho,
            n_samples=n_samples,
            seed=seed,
            use_sensor_model=use_sensor_model,
            sigma_fallback=sigma_fallback,
        )

    def __repr__(self) -> str:
        s = self.summary()
        cls = type(self).__name__
        lines = [f"{cls} summary:"]
        for name, v in s.items():
            lo, hi = v["ci95"]
            lines.append(
                f"  {name:6s}: median={v['median']:.4g}  "
                f"95%CI=[{lo:.4g}, {hi:.4g}]"
            )
        return "\n".join(lines)