"""
Kramers–Kronig validation with the linear (Lin-KK) test of Schönleber et al.,
Electrochim. Acta 131 (2014) 20.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class KKResult:
    M: int
    mu: float
    c: float
    res_re: np.ndarray
    res_im: np.ndarray
    Z_kk: np.ndarray
    f: np.ndarray

    @property
    def rms(self):
        return float(np.sqrt(np.mean(np.concatenate([self.res_re, self.res_im]) ** 2)))

    @property
    def max(self):
        return float(np.max(np.abs(np.concatenate([self.res_re, self.res_im]))))

    @property
    def verdict(self):
        return "PASS" if self.max < 0.01 else "OK" if self.max < 0.02 else "CHECK"

    @property
    def failed(self):
        """Boolean mask of points whose KK residual exceeds 2 % (likely artefacts or drift)."""
        return np.hypot(self.res_re, self.res_im) > 0.02

    @property
    def failed_frequencies(self):
        return self.f[self.failed]

    def __repr__(self):
        extra = ""
        if self.failed.any():
            ff = self.failed_frequencies
            extra = f", {int(self.failed.sum())} point(s) > 2 % at {ff.min():.3g}–{ff.max():.3g} Hz"
        return (f"KKResult({self.verdict}: RMS {100 * self.rms:.2f} %, max {100 * self.max:.2f} %, "
                f"M={self.M}, mu={self.mu:.2f}{extra})")


def _lin_kk(f, Z, c, max_M, robust=True):
    w = 2 * np.pi * f
    wt = np.abs(Z)
    b = np.concatenate([Z.real / wt, Z.imag / wt])
    best = None
    m_min = max(3, int(np.ceil(np.log10(f.max() / f.min()))))      # at least ~1 RC element per decade
    for M in range(m_min, min(max_M, len(f)) + 1):
        tau = np.logspace(np.log10(1 / w.max()), np.log10(1 / w.min()), M)
        A = np.column_stack([np.ones_like(w, complex), 1j * w, 1 / (1j * w)] +
                            [1 / (1 + 1j * w * t) for t in tau])
        Aw = np.vstack([A.real / wt[:, None], A.imag / wt[:, None]])
        x = np.linalg.lstsq(Aw, b, rcond=None)[0]
        if robust:                       # Cauchy re-weighting: artefacts cannot distort the whole test
            for _ in range(4):
                r = Aw @ x - b
                d = np.hypot(r[:len(f)], r[len(f):])
                sc = max(1.4826 * np.median(d), 1e-6) * 2.5
                sw = np.sqrt(1 / (1 + (d / sc) ** 2))
                sw2 = np.concatenate([sw, sw])
                x = np.linalg.lstsq(Aw * sw2[:, None], b * sw2, rcond=None)[0]
        Rk = x[3:]
        pos, neg = Rk[Rk >= 0].sum(), -Rk[Rk < 0].sum()
        mu = 1 - neg / pos if pos > 0 else 0.0
        best = (M, mu, A @ x)
        if mu <= c:
            break
    M, mu, Zkk = best
    return KKResult(M, mu, c, (Z.real - Zkk.real) / wt, (Z.imag - Zkk.imag) / wt, Zkk, f)


def lin_kk(spectrum, c=0.85, max_M=100, auto_relax=True, robust=True):
    """
    Linear Kramers–Kronig test on the non-excluded points of ``spectrum``.

    The mu criterion (Schönleber 2014) can stop at too few RC elements because mu is not
    monotonic in M; if the residuals then exceed 2 %, c is relaxed (0.75, 0.65, 0.5) and the
    best result is kept. Residuals below ~1 % indicate linear, causal, stable data.
    With ``robust=True`` the linear fit is re-weighted (Cauchy), so that artefacts at a few
    frequencies show up as failures at those frequencies (``failed_frequencies``) instead of
    distorting the whole test.
    """
    sp = spectrum.used
    res = _lin_kk(sp.f, sp.Z, c, max_M, robust)
    if auto_relax and res.max > 0.02:
        for c2 in (0.75, 0.65, 0.5):
            r = _lin_kk(sp.f, sp.Z, c2, max_M, robust)
            if r.max < res.max:
                res = r
            if res.max <= 0.02:
                break
    return res
