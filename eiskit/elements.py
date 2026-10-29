"""
Circuit elements: impedance and analytic derivatives with respect to each parameter.

Every element function takes the angular frequency ``w`` (array) and its parameter
vector ``p`` and returns ``(Z, dZ)`` where ``dZ`` has shape ``(len(w), n_params)``.
Analytic derivatives make fitting several times faster than finite differences.
"""
import numpy as np

# name suffix, unit, kind ("log" = positive, fitted in log10; "n" = CPE exponent; "lin" = signed)
ELEMENTS = {
    "R":   [("", "Ohm", "log")],
    "C":   [("", "F", "log")],
    "L":   [("", "H", "log")],
    "Ls":  [("", "H", "lin")],
    "CPE": [("_Q", "S*s^n", "log"), ("_n", "-", "n")],
    "W":   [("_sigma", "Ohm*s^-0.5", "log")],
    "Wo":  [("_R", "Ohm", "log"), ("_tau", "s", "log")],
    "Ws":  [("_R", "Ohm", "log"), ("_tau", "s", "log")],
    "G":   [("_R", "Ohm", "log"), ("_tau", "s", "log")],
    "TLM": [("_Rion", "Ohm", "log"), ("_Rct", "Ohm", "log"), ("_Q", "S*s^n", "log"), ("_n", "-", "n")],
    "TLMQ": [("_Rion", "Ohm", "log"), ("_Q", "S*s^n", "log"), ("_n", "-", "n")],
}

DESCRIPTIONS = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "Ls": "signed inductance (high-frequency artefact correction, not a cell property)",
    "CPE": "constant-phase element, Z = 1/[Q (jw)^n]",
    "W": "semi-infinite Warburg, Z = sigma (1 - j)/sqrt(w)",
    "Wo": "finite-space (reflective) Warburg, Z = R coth(sqrt(jw tau))/sqrt(jw tau)",
    "Ws": "finite-length (transmissive) Warburg, Z = R tanh(sqrt(jw tau))/sqrt(jw tau)",
    "G": "Gerischer, Z = R/sqrt(1 + jw tau)",
    "TLM": "de Levie transmission line (porous electrode), interface Rct || CPE: "
           "Z = sqrt(Rion Zi) coth(sqrt(Rion/Zi))",
    "TLMQ": "de Levie transmission line, blocking interface (CPE only)",
}

_BIG = 20.0  # |Re x| above which tanh/coth are replaced by their limits (avoids overflow)


def _tanh_sech2(x):
    big = np.abs(x.real) > _BIG
    xs = np.where(big, 0, x)
    t = np.where(big, np.sign(x.real), np.tanh(xs))
    s2 = np.where(big, 0, 1 / np.cosh(xs) ** 2)
    return t, s2


def element(etype, w, p):
    jw = 1j * w
    n = len(w)
    if etype == "R":
        return np.full(n, p[0], complex), np.ones((n, 1), complex)
    if etype == "C":
        Z = 1 / (jw * p[0])
        return Z, (-Z / p[0])[:, None]
    if etype in ("L", "Ls"):
        return jw * p[0], jw[:, None].astype(complex)
    if etype == "CPE":
        Z = 1 / (p[0] * jw ** p[1])
        return Z, np.column_stack([-Z / p[0], -Z * np.log(jw)])
    if etype == "W":
        g = (1 - 1j) / np.sqrt(w)
        return p[0] * g, g[:, None]
    if etype in ("Wo", "Ws"):
        R, tau = p
        x = np.sqrt(jw * tau)
        t, s2 = _tanh_sech2(x)
        if etype == "Ws":
            h = t / x
            dh = (s2 * x - t) / x ** 2
        else:                                   # coth(x)/x ; d/dx = (-csch^2 x * x - coth x)/x^2
            c = 1 / t
            csch2 = c ** 2 - 1
            h = c / x
            dh = (-csch2 * x - c) / x ** 2
        Z = R * h
        return Z, np.column_stack([h, R * dh * x / (2 * tau)])
    if etype == "G":
        R, tau = p
        s = np.sqrt(1 + jw * tau)
        Z = R / s
        return Z, np.column_stack([1 / s, -Z * jw / (2 * (1 + jw * tau))])
    if etype in ("TLM", "TLMQ"):
        return _tlm(etype, w, np.asarray(p, float))
    raise ValueError(f"unknown element {etype}")


def _tlm_z(etype, w, p):
    jw = 1j * w
    if etype == "TLM":
        Rion, Rct, Q, n = p
        Zi = 1 / (1 / Rct + Q * jw ** n)
    else:
        Rion, Q, n = p
        Zi = 1 / (Q * jw ** n)
    x = np.sqrt(Rion / Zi)
    big = np.abs(x.real) > _BIG
    coth = np.where(big, np.sign(x.real), 1 / np.tanh(np.where(big, 1, x)))
    return np.sqrt(Rion * Zi) * coth


def _tlm(etype, w, p):
    Z = _tlm_z(etype, w, p)
    d = np.empty((len(w), len(p)), complex)
    for i in range(len(p)):                           # central differences (relative step)
        h = 1e-6 * max(abs(p[i]), 1e-12)
        pp, pm = p.copy(), p.copy()
        pp[i] += h; pm[i] -= h
        d[:, i] = (_tlm_z(etype, w, pp) - _tlm_z(etype, w, pm)) / (2 * h)
    return Z, d
