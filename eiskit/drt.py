"""
Distribution of relaxation times (DRT).

The impedance is written as

    Z(w) = R_inf + jwL + 1/(jwC) + ∫ gamma(ln tau) / (1 + jw tau) d ln tau

and gamma(ln tau) >= 0 is obtained by Tikhonov-regularised, non-negative least squares on
the real and imaginary parts together (cf. Wan, Saccoccio, Chen & Ciucci, Electrochim. Acta
184 (2015) 483). The regularisation strength lambda is chosen by generalised
cross-validation (GCV) unless given. The inductance L (signed, for cable artefacts) and the
series capacitance C (for blocking, non-closing spectra) are optional.

Each peak of gamma is one relaxation process: its position gives the time constant
tau (f = 1/(2 pi tau)) and its area gives that process's polarisation resistance. Peaks
outside the measured frequency window (``in_window=False``) are extrapolations and are not
counted as major peaks.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.signal import find_peaks


@dataclass
class DRTResult:
    tau: np.ndarray
    gamma: np.ndarray            # Ohm (per unit ln tau)
    R_inf: float
    L: float
    C: float                     # np.inf when not used
    lam: float
    f: np.ndarray
    Z: np.ndarray
    Z_drt: np.ndarray
    peaks: pd.DataFrame
    name: str = ""

    @property
    def R_pol(self):
        return float(np.sum(self.gamma) * np.mean(np.diff(np.log(self.tau))))

    @property
    def rms_pct(self):
        r = (self.Z - self.Z_drt) / np.abs(self.Z)
        return float(100 * np.sqrt(np.mean(np.concatenate([r.real, r.imag]) ** 2)))

    @property
    def n_major_peaks(self):
        return int(self.peaks["major"].sum()) if len(self.peaks) else 0

    @property
    def f_tau(self):
        return 1 / (2 * np.pi * self.tau)

    def to_frame(self):
        return pd.DataFrame({"tau_s": self.tau, "f_Hz": self.f_tau, "gamma_Ohm": self.gamma})

    def __repr__(self):
        return (f"DRTResult({len(self.peaks)} peak(s), R_inf {self.R_inf:.4g} Ohm, R_pol {self.R_pol:.4g} Ohm, "
                f"lambda {self.lam:.2g}, reconstruction RMS {self.rms_pct:.2f} %)")


def _matrices(f, tau, use_L, use_C):
    w = 2 * np.pi * f
    dlt = np.mean(np.diff(np.log(tau)))
    wt = np.outer(w, tau)
    K = dlt / (1 + 1j * wt)                                # (N, M)
    extra = [np.ones_like(w, complex)]                     # R_inf
    names = ["R_inf"]
    if use_L:
        extra.append(1j * w); names.append("L")
    if use_C:
        extra.append(1 / (1j * w)); names.append("1/C")
    return np.column_stack(extra + [K]), names


def drt(spectrum, lam="auto", n_tau=None, tau_range=None, inductance=True, capacitance="auto",
        weighting="modulus", prominence=0.05, min_fraction=0.05, extend=0.5):
    """
    Compute the DRT of the non-excluded points of ``spectrum``.

    lam         : regularisation parameter; "auto" (default) = the smoothest DRT whose reconstruction
                  error is within 25 % of the best attainable (discrepancy principle);
                  "gcv" = generalised cross-validation (tends to under-regularise noisy data)
    n_tau       : number of time constants (default: 10 per decade of the measured range + margins)
    tau_range   : (tau_min, tau_max) in s (default: ``extend`` decades beyond the measured range)
    inductance  : include a (signed) series inductance
    capacitance : include a series capacitance (True/False/"auto": used when the lowest-frequency
                  phase exceeds 45 deg, i.e. the spectrum does not close on the real axis)
    prominence  : minimum relative peak prominence for peak detection
    min_fraction: peaks carrying less than this fraction of R_pol are marked minor (major=False)
    """
    sp = spectrum.used
    f, Z = sp.f, sp.Z
    if capacitance == "auto":
        capacitance = bool(-np.degrees(np.angle(Z[np.argmin(f)])) > 45)
    ext = 10 ** extend
    tmin, tmax = tau_range or (1 / (2 * np.pi * f.max()) / ext, 1 / (2 * np.pi * f.min()) * ext)
    n_tau = n_tau or int(np.ceil(10 * np.log10(tmax / tmin))) + 1
    tau = np.logspace(np.log10(tmin), np.log10(tmax), n_tau)
    A, names = _matrices(f, tau, inductance, capacitance)
    wt = np.abs(Z) if weighting == "modulus" else np.ones(len(Z))
    Ar = np.vstack([A.real / wt[:, None], A.imag / wt[:, None]])
    b = np.concatenate([Z.real / wt, Z.imag / wt])
    ne, M = len(names), n_tau
    # second-derivative regularisation acting on gamma only
    D = np.zeros((M - 2, ne + M))
    for i in range(M - 2):
        D[i, ne + i:ne + i + 3] = [1, -2, 1]
    # scale columns so that lambda is dimensionless and comparable between spectra
    s = np.linalg.norm(Ar, axis=0); s[s == 0] = 1
    As = Ar / s
    Ds = D / s

    def solve(l):
        Aaug = np.vstack([As, np.sqrt(l) * Ds])
        baug = np.concatenate([b, np.zeros(M - 2)])
        lo = np.zeros(ne + M); hi = np.full(ne + M, np.inf)
        if inductance:
            lo[names.index("L")] = -np.inf
        return lsq_linear(Aaug, baug, bounds=(lo, hi), method="bvls", lsmr_tol="auto").x

    def rms_of(x):
        r = As @ x - b
        return np.sqrt(np.mean(r ** 2))

    if lam == "auto":
        grid = np.logspace(-7, 0, 29)
        sols = [(l, solve(l)) for l in grid]
        errs = np.array([rms_of(x) for _, x in sols])
        ok = np.where(errs <= errs.min() * 1.25 + 1e-12)[0]
        lam, x = sols[ok.max()]
    elif lam == "gcv":
        grid = np.logspace(-8, 0, 33)
        AtA, DtD = As.T @ As, Ds.T @ Ds
        n = len(b)
        best = None
        for l in grid:
            try:
                H = As @ np.linalg.solve(AtA + l * DtD + 1e-14 * np.eye(len(AtA)), As.T)
            except np.linalg.LinAlgError:
                continue
            x = solve(l)
            r = As @ x - b
            g = n * (r @ r) / max(n - np.trace(H), 1e-9) ** 2
            if best is None or g < best[0]:
                best = (g, l, x)
        _, lam, x = best
    else:
        x = solve(float(lam))
    x = x / s
    R_inf = float(x[0])
    L = float(x[names.index("L")]) if inductance else 0.0
    C = float(1 / x[names.index("1/C")]) if capacitance and x[names.index("1/C")] > 0 else np.inf
    gamma = x[ne:]
    Z_drt = A @ x
    peaks = _peaks(tau, gamma, prominence, min_fraction, (f.min(), f.max()))
    return DRTResult(tau, gamma, R_inf, L, C, float(lam), f, Z, Z_drt, peaks, sp.name)


def _peaks(tau, gamma, prominence, min_fraction=0.05, window=None):
    if gamma.max() <= 0:
        return pd.DataFrame(columns=["tau_s", "f_Hz", "gamma_Ohm", "R_Ohm", "fraction", "in_window", "major"])
    idx, _ = find_peaks(gamma, prominence=prominence * gamma.max())
    if gamma[0] > gamma[1] and gamma[0] > prominence * gamma.max():
        idx = np.r_[0, idx]
    if gamma[-1] > gamma[-2] and gamma[-1] > prominence * gamma.max():
        idx = np.r_[idx, len(gamma) - 1]
    idx = np.unique(idx)
    dlt = np.mean(np.diff(np.log(tau)))
    # split the distribution at the minima between neighbouring peaks
    bounds = [0] + [i + int(np.argmin(gamma[i:j + 1])) for i, j in zip(idx[:-1], idx[1:])] + [len(gamma)]
    total = gamma.sum() * dlt
    rows = []
    for k, i in enumerate(idx):
        R = gamma[bounds[k]:bounds[k + 1]].sum() * dlt
        fpk = 1 / (2 * np.pi * tau[i])
        inside = window is None or (window[0] <= fpk <= window[1])
        rows.append({"tau_s": tau[i], "f_Hz": fpk, "gamma_Ohm": gamma[i],
                     "R_Ohm": R, "fraction": R / total if total > 0 else np.nan, "in_window": inside,
                     "major": bool(inside and total > 0 and R / total >= min_fraction)})
    return pd.DataFrame(rows)
