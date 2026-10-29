"""Run with:  python -m pytest tests"""
import numpy as np
import pytest

import eiskit as ek

TRUE = [2.5, 0.3, 1e-3, 0.9, 1.4, 0.1, 0.85]


def synthetic(noise=0.002, seed=1, corrupt=False):
    f = np.logspace(5, -2, 71)
    Z = ek.Circuit("R0-p(R1,CPE1)-p(R2,CPE2)")(TRUE, f)
    rng = np.random.default_rng(seed)
    Z = Z + noise * abs(Z) * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
    if corrupt:
        Z[[20, 35, 50]] *= np.array([1.06, 0.93 + 0.05j, 1.08])        # spikes
        Z[:6] += 1j * 2 * np.pi * f[:6] * 4e-7                         # inductive artefact
    return ek.Spectrum(f, Z, "synthetic")


@pytest.mark.parametrize("text,canonical", [
    ("R0-p(R1,CPE1)", "R0-p(R1,CPE1)"), ("R-p(R,CPE)", "R0-p(R1,CPE1)"), ("(R)p(CPE-Ws)", "p(R0,CPE1-Ws1)"),
    ("R+Q/(R+W)", "R0-p(CPE1,R1-W1)"), ("R(Q[RW])", "R0-p(CPE1,R1-W1)"), ("R(QR)(QR)", "R0-p(CPE1,R1)-p(CPE2,R2)"),
    ("Rs-p(Rct,CPEdl)", "Rs-p(Rct,CPEdl)"), ("L-R-(R||Q)", "L1-R0-p(R1,CPE1)")])
def test_notation(text, canonical):
    assert ek.canonical(text) == canonical


def test_notation_errors():
    for bad in ["p(CPE-Ws)", "R-p(R1,", "R1-R1", "Rx"]:
        with pytest.raises(ek.NotationError):
            ek.canonical(bad)


def test_literature_library():
    assert len(ek.LITERATURE) >= 50
    assert len({ek.Circuit(c).string for _, _, c, _ in ek.LITERATURE}) == len(ek.LITERATURE)


def test_jacobian_matches_finite_differences():
    w = 2 * np.pi * np.logspace(-2, 5, 30)
    for s in ["R0-p(R1,CPE1)-p(R2-Ws2,CPE2)", "Ls0-R0-p(C1,R1-p(CPE2,R2-Wo2))", "R0-G1-p(R2-W2,C2)", "R0-TLM1"]:
        c = ek.Circuit(s)
        p = np.linspace(0.3, 1.2, c.n_params)
        p[c.kinds == "n"] = 0.8
        _, J = c.impedance(p, w, jacobian=True)
        for i in range(c.n_params):
            h = 1e-7 * max(abs(p[i]), 1e-3)
            dp = np.zeros_like(p); dp[i] = h
            fd = (c.impedance(p + dp, w) - c.impedance(p - dp, w)) / (2 * h)
            assert np.max(np.abs(fd - J[:, i])) <= 1e-4 * np.abs(fd).max()


def test_fit_recovers_parameters():
    r = ek.fit(synthetic(), "R0-p(R1,CPE1)-p(R2,CPE2)")
    assert r.valid and np.allclose(r.p, TRUE, rtol=0.05)


def test_robust_fit_ignores_bad_points():
    sp = synthetic(corrupt=True)
    plain = ek.fit(sp, "R0-p(R1,CPE1)-p(R2,CPE2)", robust=False)
    robust = ek.fit(sp, "R0-p(R1,CPE1)-p(R2,CPE2)", robust=True)
    assert np.max(np.abs(robust.p / TRUE - 1)) < 0.1 < np.max(np.abs(plain.p / TRUE - 1))
    assert set(np.where(robust.outliers)[0]) == {0, 1, 2, 3, 4, 5, 20, 35, 50}


def test_text_input():
    sp = ek.from_text("Freq\tZ'\t-Z''\n1000\t2.0\t0.5\n100\t2.5\t1.0\n10\t3.0\t0.4")
    assert np.allclose(sp.Z.imag, [-0.5, -1.0, -0.4])
    sp = ek.from_text("f;Zre;Zim\n1000;2,0;-0,5\n100;2,5;-1,0\n10;3,0;-0,4")
    assert np.allclose(sp.Z.real, [2.0, 2.5, 3.0])
    sp = ek.from_columns("1000\n100\n10", "2\n2.5\n3", "0.5\n1\n0.4")
    assert np.allclose(sp.Z.imag, [-0.5, -1.0, -0.4])


def test_kk_passes_on_valid_data():
    assert ek.lin_kk(synthetic()).max < 0.01


def test_drt_finds_two_processes():
    d = ek.drt(synthetic())
    pk = d.peaks[d.peaks.major]
    assert len(pk) == 2 and abs(d.R_inf - 2.5) < 0.05
    assert np.allclose(sorted(pk.R_Ohm), [0.3, 1.4], rtol=0.15)


def test_fit_all_and_tables():
    data = {"A": synthetic(seed=1), "B": synthetic(seed=2)}
    res = ek.fit_all(data, "R-p(R,CPE)-p(R,CPE)", verbose=False)
    T = res.tables()
    assert list(T["comparison"].columns) == ["Element", "A", "B"]
    assert len(T["S2"]) == 2
