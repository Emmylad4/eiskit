"""
Complex non-linear least-squares (CNLS) fitting of equivalent circuits.

* :func:`fit`     – fit one circuit to one spectrum
* :func:`search`  – fit a library of circuits (default: >60 literature circuits) and rank them
* :func:`fit_all` – fit several samples (materials) and collect publication tables

Robust fitting (default)
------------------------
Real spectra contain odd points (noise spikes, instrument artefacts, drift). Instead of
requiring them to be removed by hand, the fit uses a robust (Cauchy) loss whose scale is
estimated from the data: points that follow the general trend are fitted normally, points
far from it are progressively down-weighted. They are reported as ``outliers`` (drawn
hollow in plots) and the statistics are computed on the remaining points.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .circuit import Circuit, circuit_library, initial_guess
from .validation import lin_kk

LOG_LB, LOG_UB = -20.0, 12.0        # bounds of log10(parameter) for positive parameters
N_LB, N_UB = 0.3, 1.0               # CPE exponent
L_BOUND = 1e-2                      # |signed inductance| bound (H)


# ---------------------------------------------------------------------------------------
@dataclass
class FitResult:
    circuit: Circuit
    p: np.ndarray
    sd: np.ndarray
    rel: np.ndarray
    corr: np.ndarray
    chi2: float
    red_chi2: float
    rms_pct: float
    r2: float
    aic: float
    aicc: float
    bic: float
    f: np.ndarray
    Z: np.ndarray
    Zfit: np.ndarray
    res_re: np.ndarray
    res_im: np.ndarray
    flags: list
    valid: bool
    derived: list
    weighting: str
    name: str = ""
    outliers: np.ndarray = None
    robust: bool = False
    n_eval: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def params(self):
        return dict(zip(self.circuit.names, self.p))

    @property
    def k(self):
        return len(self.p)

    @property
    def n_outliers(self):
        return int(self.outliers.sum()) if self.outliers is not None else 0

    def table(self):
        """Parameters with standard errors and 95 % confidence intervals."""
        return pd.DataFrame({"value": self.p, "std_error": self.sd, "rel_error_pct": self.rel,
                             "ci95_low": self.p - 1.96 * self.sd, "ci95_high": self.p + 1.96 * self.sd,
                             "unit": self.circuit.units}, index=self.circuit.names)

    def predict(self, f):
        return self.circuit(self.p, f)

    def summary(self):
        head = (f"{self.name}: {self.circuit.string}   [{self.circuit.explain()}]\n"
                f"RMS residual {self.rms_pct:.3f} % of |Z|, chi2 {self.chi2:.3g}, reduced chi2 {self.red_chi2:.3g}, "
                f"R2 {self.r2:.5f}, AICc {self.aicc:.1f}, {'valid' if self.valid else 'NOT VALID'}")
        if self.n_outliers:
            head += f"\n{self.n_outliers} point(s) deviate from the trend and were down-weighted (robust fit)"
        lines = [head]
        for nm, v, s, e, u in zip(self.circuit.names, self.p, self.sd, self.rel, self.circuit.units):
            lines.append(f"  {nm:<12} {v:>12.5g} ± {s:<10.3g} ({e:6.1f} %) {u}")
        lines += [f"  ! {fl}" for fl in self.flags]
        return "\n".join(lines)

    def __repr__(self):
        return (f"FitResult({self.name}: {self.circuit.string}, RMS {self.rms_pct:.3f} %, AICc {self.aicc:.1f}, "
                f"{'valid' if self.valid else 'not valid'})")


def derived_quantities(circ, p):
    """Effective capacitance (Hsu–Mansfeld), time constant and characteristic frequency."""
    val = lambda name: p[circ.names.index(name)]
    out = []

    def walk(node):
        if node[0] == "e":
            return
        if node[0] == "p" and len(node[1]) == 2:
            for cap, other in ((node[1][0], node[1][1]), (node[1][1], node[1][0])):
                if cap[0] != "e" or cap[1] not in ("C", "CPE") or cap[2].endswith("9"):
                    continue
                R = (other[2] if other[0] == "e" and other[1] == "R" else
                     other[1][0][2] if other[0] == "s" and other[1][0][0] == "e"
                     and other[1][0][1] == "R" else None)
                if R is None:
                    continue
                Rv = val(R)
                if cap[1] == "C":
                    C = val(cap[2]); tau = Rv * C
                else:
                    Q, n = val(cap[2] + "_Q"), val(cap[2] + "_n")
                    C = Q ** (1 / n) * Rv ** ((1 - n) / n); tau = (Rv * Q) ** (1 / n)
                out.append(dict(element=f"p({R},{cap[2]})", C_eff=C, tau=tau, f_c=1 / (2 * np.pi * tau)))
        for c in node[1]:
            walk(c)

    walk(circ.tree)
    for t, n in circ.elements:
        if t in ("Wo", "Ws", "G"):
            tau = val(n + "_tau")
            out.append(dict(element=n, tau=tau, f_c=1 / (2 * np.pi * tau)))
    return out


# ---------------------------------------------------------------------------------------
def _weights(Z, weighting):
    if weighting == "modulus":
        return np.abs(Z), np.abs(Z)
    if weighting == "proportional":
        fl = 1e-3 * np.abs(Z)
        return np.maximum(np.abs(Z.real), fl), np.maximum(np.abs(Z.imag), fl)
    if weighting == "unit":
        return np.ones(len(Z)), np.ones(len(Z))
    raise ValueError("weighting must be 'modulus', 'proportional' or 'unit'")


def fit(spectrum, circuit, weighting="modulus", robust=True, n_starts=8, p0=None, seed=0,
        order_arcs=True, tol=1e-10, max_nfev=2000, outlier_threshold=4.0):
    """
    Fit ``circuit`` (any notation, or a :class:`Circuit`) to the non-excluded points of
    ``spectrum``.

    Real and imaginary parts are fitted together (CNLS) with an analytic Jacobian; parameters
    stay physical (R, C, Q, tau > 0; 0.3 <= n <= 1). ``n_starts`` multi-starts are tried and
    ``p0`` adds a user starting point. With ``robust=True`` (default) points that deviate from
    the trend by more than ``outlier_threshold`` robust standard deviations are down-weighted
    and reported in ``result.outliers``.
    """
    circ = circuit if isinstance(circuit, Circuit) else Circuit(circuit)
    sp = spectrum.used
    f, Z = sp.f, sp.Z
    if 2 * len(f) <= circ.n_params:
        raise ValueError(f"{circ.string} has {circ.n_params} parameters but only {len(f)} points")
    w = 2 * np.pi * f
    islog, isn = circ.kinds == "log", circ.kinds == "n"
    lb = np.where(islog, LOG_LB, np.where(isn, N_LB, -L_BOUND))
    ub = np.where(islog, LOG_UB, np.where(isn, N_UB, L_BOUND))
    wr, wi = _weights(Z, weighting)
    ln10 = np.log(10)

    def unpack(u):
        return np.where(islog, 10.0 ** u, u)

    def resid(u):
        Zm = circ.impedance(unpack(u), w)
        r = np.concatenate([(Zm.real - Z.real) / wr, (Zm.imag - Z.imag) / wi])
        return np.where(np.isfinite(r), r, 1e10)

    def jac(u):
        p = unpack(u)
        _, J = circ.impedance(p, w, jacobian=True)
        J = J * np.where(islog, p * ln10, 1.0)[None, :]
        out = np.vstack([J.real / wr[:, None], J.imag / wi[:, None]])
        return np.where(np.isfinite(out), out, 0.0)

    starts = []
    if p0 is not None:
        pi = np.asarray(p0, float)
        starts.append(np.where(islog, np.log10(np.maximum(pi, 1e-30)), pi))
    for rev in (False, True):
        q = initial_guess(circ, f, Z, reverse=rev)
        starts.append(np.where(islog, np.log10(np.maximum(q, 1e-30)), q))
    rng = np.random.default_rng(seed)
    base = list(starts)
    while len(starts) < max(n_starts, len(base)):
        u = base[len(starts) % len(base)] + np.where(islog, rng.normal(0, 1.0, circ.n_params), 0)
        starts.append(np.where(isn, rng.uniform(0.6, 1.0, circ.n_params), u))

    def run(u0, loss="linear", scale=1.0):
        return least_squares(resid, np.clip(u0, lb + 1e-6, ub - 1e-6), jac=jac, bounds=(lb, ub),
                             method="trf", x_scale="jac", ftol=tol, xtol=tol, gtol=tol,
                             max_nfev=max_nfev, loss=loss, f_scale=scale)

    loss, scale = ("soft_l1", 0.02) if robust else ("linear", 1.0)
    best, n_eval = None, 0
    for us in starts:
        try:
            r = run(us, loss, scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        n_eval += r.nfev
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError(f"fit of {circ.string} failed")

    outliers = np.zeros(len(f), bool)
    if robust:
        d = np.hypot(*best.fun.reshape(2, -1))
        s = max(1.4826 * np.median(d), 1e-6)
        try:                                            # Cauchy refinement: follow the trend
            r = run(best.x, "cauchy", 1.5 * s)
            n_eval += r.nfev
            best = r
        except (ValueError, np.linalg.LinAlgError):
            pass
        d = np.hypot(*best.fun.reshape(2, -1))
        s = max(1.4826 * np.median(d), 1e-6)
        outliers = d > max(outlier_threshold * s, 0.002)   # deviations below 0.2 % are never outliers
    # statistics from the true (unscaled) Jacobian: with robust losses scipy returns a
    # loss-modified Jacobian whose rows for larger residuals are nearly zero
    best.jac = jac(best.x)
    res = _statistics(circ, best, unpack, lb, ub, f, Z, w, weighting, sp, ~outliers)
    res.n_eval, res.outliers, res.robust = n_eval, outliers, robust
    return reorder_arcs(res) if order_arcs else res


def _statistics(circ, best, unpack, lb, ub, f, Z, w, weighting, sp, inl):
    u, p = best.x, unpack(best.x)
    islog = circ.kinds == "log"
    rows = np.concatenate([inl, inl])
    N, k = int(inl.sum()), len(p)
    nobs, dof = 2 * N, max(2 * N - k, 1)
    chi2 = float(np.sum(best.fun[rows] ** 2))
    red = chi2 / dof
    J = best.jac[rows]
    cov = np.linalg.pinv(J.T @ J) * red
    sd_u = np.sqrt(np.abs(np.diag(cov)))
    sd = np.where(islog, p * np.log(10) * sd_u, sd_u)
    with np.errstate(all="ignore"):
        corr = cov / np.outer(sd_u, sd_u)
        rel = 100 * sd / np.abs(p)
    Zfit = circ.impedance(p, w)
    aic = nobs * np.log(chi2 / nobs) + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(nobs - k - 1, 1)
    bic = nobs * np.log(chi2 / nobs) + k * np.log(nobs)
    Zi = Z[inl]
    r2 = 1 - np.sum(np.abs(Zi - Zfit[inl]) ** 2) / np.sum(np.abs(Zi - Zi.mean()) ** 2)

    flags, critical = [], False
    # sensitivity to a *relative* change of each parameter (scale-invariant, also for signed L in H)
    colnorm = np.linalg.norm(J, axis=0) * np.where(circ.kinds == "lin", np.maximum(np.abs(p), 1e-30), 1.0)
    dead = colnorm < 1e-6 * max(colnorm.max(), 1e-300)
    zmax = np.abs(Zi).max()
    for i, nm in enumerate(circ.names):
        if dead[i]:
            flags.append(f"{nm} has no influence on the fit (value {p[i]:.3g})"); critical = True
            rel[i] = np.inf
            continue
        if not np.isfinite(rel[i]) or rel[i] > 100:
            flags.append(f"{nm} undetermined (rel. error {rel[i]:.0f} %)"); critical = True
        if circ.kinds[i] == "lin":
            continue
        if circ.units[i] == "Ohm" and p[i] > 1e4 * zmax:
            flags.append(f"{nm} = {p[i]:.2g} Ohm is effectively infinite"); critical = True
        if circ.units[i] == "Ohm" and p[i] < 1e-4 * zmax:
            flags.append(f"{nm} = {p[i]:.2g} Ohm is negligible"); critical = True
        at_bound = u[i] - lb[i] < 1e-3 or ub[i] - u[i] < 1e-3
        if at_bound and not (circ.kinds[i] == "n" and ub[i] - u[i] < 1e-3):
            flags.append(f"{nm} at bound"); critical = True
        if circ.kinds[i] == "n":
            if p[i] < 0.5:
                flags.append(f"{nm} = {p[i]:.2f} < 0.5: strongly distributed / diffusion-like")
            elif p[i] > 0.98:
                flags.append(f"{nm} = {p[i]:.3f} ~ 1: ideal capacitor (simpler model possible)")
    for i in range(k):
        for j in range(i + 1, k):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) > 0.98:
                flags.append(f"{circ.names[i]} / {circ.names[j]} strongly correlated (r = {corr[i, j]:.3f})")
    derived = derived_quantities(circ, p)
    for dq in derived:
        if dq["tau"] < 0.1 / w.max() or dq["tau"] > 10 / w.min():
            flags.append(f"time constant of {dq['element']} (f_c = {dq['f_c']:.3g} Hz) outside the measured window")
    return FitResult(circ, p, sd, rel, corr, chi2, red, 100 * np.sqrt(chi2 / nobs), r2, aic, aicc, bic,
                     f, Z, Zfit, (Z.real - Zfit.real) / np.abs(Z), (Z.imag - Zfit.imag) / np.abs(Z),
                     flags, not critical, derived, weighting, sp.name)


def reorder_arcs(res):
    """Renumber interchangeable series R||CPE blocks so that block 1 has the smallest time
    constant (highest frequency): element labels then mean the same in every sample."""
    c, p = res.circuit, res.p
    perm = np.arange(c.n_params)
    for kind in ("CPE", "C"):
        bl = [b for b in c.series_blocks() if b[0] == kind]
        if len(bl) < 2:
            continue
        slots = [[c.names.index(b[1])] + [i for i, nm in enumerate(c.names)
                                          if nm == b[2] or nm.startswith(b[2] + "_")] for b in bl]

        def tau(b):
            R = p[c.names.index(b[1])]
            if kind == "C":
                return R * p[c.names.index(b[2])]
            return (R * p[c.names.index(b[2] + "_Q")]) ** (1 / p[c.names.index(b[2] + "_n")])
        order = sorted(range(len(bl)), key=lambda i: tau(bl[i]))
        for dst, i in zip(slots, order):
            perm[dst] = slots[i]
    if (perm != np.arange(len(perm))).any():
        res.p, res.sd, res.rel = res.p[perm], res.sd[perm], res.rel[perm]
        res.corr = res.corr[np.ix_(perm, perm)]
        res.derived = derived_quantities(c, res.p)
    return res


# ---------------------------------------------------------------------------------------
def _worker(args):
    string, desc, f, Z, weighting, robust, n_starts, tol, max_nfev = args
    from .data import Spectrum
    try:
        return fit(Spectrum(f, Z), Circuit(string, desc), weighting, robust, n_starts, tol=tol,
                   max_nfev=max_nfev, order_arcs=False)
    except Exception:
        return None


def _run(jobs, n_jobs):
    if n_jobs and n_jobs > 1 and len(jobs) > 4:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            return list(ex.map(_worker, jobs, chunksize=max(1, len(jobs) // (4 * n_jobs))))
    return [_worker(j) for j in jobs]


@dataclass
class SearchResult:
    fits: list
    ranking: pd.DataFrame
    seconds: float
    n_screened: int
    prefilter_info: str = ""

    @property
    def best(self):
        return self.fits[0]

    def __repr__(self):
        return (f"SearchResult({self.n_screened} circuits in {self.seconds:.1f} s; best: "
                f"{self.best.circuit.string}, RMS {self.best.rms_pct:.3f} %)")


def search(spectrum, circuits="literature", weighting="modulus", robust=True, prefilter=None,
           max_params=None, screen_starts=3, refine_starts=10, refine_top=15, n_jobs=None, verbose=True):
    """
    Fit a library of circuits and rank them: valid fits first (all parameters determined, none
    at a bound), then by AICc.

    circuits : "literature" (>60 named circuits, default), "extended" (~290 generated),
               "exhaustive", "all", or a list / dict of circuits in any notation
    prefilter: "drt" keeps only circuits whose number of time constants is compatible with the
               number of DRT peaks (n-1 to n+2); None screens every circuit
    n_jobs   : CPU cores to use (default: all)
    """
    t0 = time.time()
    if isinstance(circuits, str):
        lib = circuit_library(circuits, max_params)
    elif isinstance(circuits, dict):
        lib = {Circuit(k).string: v for k, v in circuits.items()}
    else:
        lib = {Circuit(c).string: "" for c in circuits}
    info = ""
    if prefilter == "drt":
        from .drt import drt as _drt
        npk = _drt(spectrum).n_major_peaks
        lo, hi = max(1, npk - 1), npk + 2           # overlapping processes merge in the DRT: allow +2
        before = len(lib)
        lib = {s: d for s, d in lib.items() if lo <= Circuit(s).n_time_constants() <= hi}
        info = f"DRT shows {npk} peak(s): kept {len(lib)} of {before} circuits with {lo}-{hi} time constants"
        if verbose:
            print(info)
    sp = spectrum.used
    n_jobs = os.cpu_count() if n_jobs is None else n_jobs
    lib = {s: d for s, d in lib.items() if 2 * len(sp.f) - Circuit(s).n_params > 1}
    jobs = [(s, d, sp.f, sp.Z, weighting, robust, screen_starts, 1e-6, 300) for s, d in lib.items()]
    if verbose:
        print(f"screening {len(jobs)} circuits on {n_jobs} core(s) ...", flush=True)
    fits = [r for r in _run(jobs, n_jobs) if r is not None]
    fits.sort(key=lambda r: (not r.valid, r.aicc))
    top = fits[:refine_top]
    jobs = [(r.circuit.string, r.circuit.description, sp.f, sp.Z, weighting, robust, refine_starts, 1e-10, 2000)
            for r in top]
    refined = {r.circuit.string: r for r in _run(jobs, n_jobs) if r is not None}
    fits = [refined[r.circuit.string] if r.circuit.string in refined and
            refined[r.circuit.string].chi2 <= r.chi2 * 1.000001 else r for r in fits]
    fits = [reorder_arcs(r) for r in fits]
    fits.sort(key=lambda r: (not r.valid, r.aicc))
    for r in fits:
        r.name = sp.name
    dt = time.time() - t0
    ranking = pd.DataFrame([{"rank": i + 1, "circuit": r.circuit.string, "k": r.k, "RMS_%": r.rms_pct,
                             "AICc": r.aicc, "valid": r.valid, "outliers": r.n_outliers,
                             "description": r.circuit.description, "flags": "; ".join(r.flags)}
                            for i, r in enumerate(fits)]).set_index("rank")
    if verbose:
        print(f"done: {len(fits)} circuits in {dt:.1f} s; best {fits[0].circuit.string} "
              f"(RMS {fits[0].rms_pct:.3f} %)")
    return SearchResult(fits, ranking, dt, len(lib), info)


# ---------------------------------------------------------------------------------------
@dataclass
class Results:
    """Fits of several samples (materials, conditions ...) with publication tables."""
    fits: dict                       # name -> FitResult
    kk: dict                         # name -> KKResult
    spectra: dict                    # name -> Spectrum
    selection: dict                  # name -> how the circuit was chosen
    searches: dict = field(default_factory=dict)
    title: str = ""

    def tables(self, labels=None):
        """Publication tables; ``labels`` renames elements, e.g. ``{"R2": "R_ct"}``."""
        from .report import tables
        return tables(self, labels)

    def comparison_table(self, labels=None):
        """Elements as rows, samples as columns (value ± SD)."""
        from .report import comparison_table
        return comparison_table(self, labels)

    def to_excel(self, path, labels=None):
        from .report import to_excel
        return to_excel(self, path, labels)

    def to_docx(self, path, figures=None, labels=None):
        """Word report; ``figures`` = {caption: image path}."""
        from .report import to_docx
        return to_docx(self, path, figures, labels)

    def __repr__(self):
        rows = [f"  {n:<22} {r.circuit.string:<32} RMS {r.rms_pct:.3f} %  "
                f"{'valid' if r.valid else 'check flags'}" +
                (f"  ({r.n_outliers} outlier(s))" if r.n_outliers else "") for n, r in self.fits.items()]
        return "Results\n" + "\n".join(rows)


def fit_all(spectra, circuit="search", weighting="modulus", robust=True, n_starts=12,
            search_kw=None, verbose=True, title=""):
    """
    Fit every sample in ``spectra`` ({name: Spectrum}).

    circuit:
        a circuit in any notation      – the same circuit for every sample
        {name: circuit}                – a circuit per sample
        "search"                       – the best literature circuit for each sample
        "search-common"                – the one circuit that is best over all samples (lowest
                                         summed AICc among circuits valid for at least half
                                         of the samples), so that parameters can be compared
    """
    names = list(spectra)
    skw = {"verbose": False, "robust": robust, **(search_kw or {})}
    fits, sel, searches = {}, {}, {}
    if circuit in ("search", "search-common"):
        for n in names:
            if verbose:
                print(f"searching circuits for {n} ...", flush=True)
            searches[n] = search(spectra[n], weighting=weighting, **skw)
    if circuit == "search-common":
        per = {n: {r.circuit.string: r for r in searches[n].fits} for n in names}
        common = set.intersection(*(set(p) for p in per.values()))
        nvalid = {c: sum(per[n][c].valid for n in names) for c in common}
        need = max(1, int(np.ceil(len(names) / 2)))
        cand = [c for c in common if nvalid[c] >= need] or list(common)
        best = min(cand, key=lambda c: sum(per[n][c].aicc for n in names))
        plan = {n: best for n in names}
        why = {n: f"best common circuit (lowest summed AICc; valid for {nvalid[best]}/{len(names)} samples)"
               for n in names}
    elif circuit == "search":
        plan = {}
        for n in names:
            v = [r for r in searches[n].fits if r.valid] or searches[n].fits
            plan[n] = v[0].circuit.string
        why = {n: f"best of {searches[n].n_screened} circuits (AICc)" for n in names}
    elif isinstance(circuit, dict):
        plan = {n: circuit[n] for n in names}
        why = {n: "user-defined" for n in names}
    else:
        plan = {n: circuit for n in names}
        why = {n: "user-defined" for n in names}
    for n in names:
        r = fit(spectra[n], plan[n], weighting, robust, n_starts)
        r.name = n
        fits[n] = r
        sel[n] = why[n]
        if verbose:
            print(f"{n:<22} {r.circuit.string:<32} RMS {r.rms_pct:.3f} %  "
                  f"{'valid' if r.valid else 'CHECK: ' + '; '.join(r.flags)}")
    return Results(fits, {n: lin_kk(spectra[n]) for n in names}, dict(spectra), sel, searches, title)
