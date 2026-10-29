# eiskit 2.0

**Equivalent-circuit fitting and distribution-of-relaxation-times (DRT) analysis of electrochemical impedance spectra, as a Python package and as an offline web app.**

eiskit compares impedance spectra of several samples (materials, electrodes, conditions). For each sample it:
- checks the data with the Kramers–Kronig test,
- fits equivalent circuits robustly, without you having to delete odd points,
- searches a library of more than 60 circuits from the literature, or uses a circuit you type in almost any notation,
- computes the DRT, and
- writes publication-ready figures and tables.

## Two ways to use it

| | Offline web app | Python package |
|---|---|---|
| Install | none: double-click `eiskit_app.html` | `pip install .` |
| Data input | paste a table, paste each column, or upload CSV/TXT/XLSX | files, pasted text, columns, interactive prompt |
| Output | figures (SVG/PNG), tables (copy, CSV, Excel) | figures (SVG/TIFF/PNG/PDF), Excel, Word report |
| Runs | in any current browser, offline; data never leave the computer | Python ≥ 3.9 |

The app is in `eiskit/app/eiskit_app.html`, and `python -m eiskit app` opens it. It contains its own JavaScript fitting engine, so it needs neither Python nor an internet connection. Its results agree with the Python package to within numerical precision.

## Quick start (Python)

```python
import eiskit as ek

data = ek.read_spectra("my_samples.xlsx")              # {sample name: Spectrum}
data["NF/Ni3N"] = ek.from_text(pasted_text, "NF/Ni3N") # or paste data
# data["X"] = ek.prompt_spectrum()                     # or answer prompts

res = ek.fit_all(data, "Rs-p(Rct,CPEdl)")              # one circuit for all samples
res = ek.fit_all(data, "search-common")                # or: best common literature circuit
print(res)

ek.save_figure(ek.plot_nyquist(res, circuit_inset="Rs-p(Rct,CPEdl)"), "nyquist")   # .svg + .tiff
ek.save_figure(ek.plot_bode(res), "bode")
ek.save_figure(ek.plot_drt({n: ek.drt(s) for n, s in data.items()}), "drt")
res.to_excel("results.xlsx"); res.to_docx("tables.docx")
```

Single spectra:
```python
sp = data["NF/Ni3N"]
ek.lin_kk(sp)                                   # Kramers–Kronig check; lists failing frequencies
r = ek.fit(sp, "R(QR)(QR)")                     # Boukamp CDC works too
print(r.summary()); r.table()                   # values, SD, 95 % CI
s = ek.search(sp)                               # rank the literature circuits
d = ek.drt(sp); d.peaks                         # processes: f, tau, resistance
```

Command line:
```bash
python -m eiskit app                                    # open the offline web app
python -m eiskit circuits                               # list the literature circuits
python -m eiskit parse "(R)p(CPE-Ws)"                   # show how a notation is read
python -m eiskit fit    a.csv b.xlsx --circuit "R-p(R,CPE)"
python -m eiskit search a.csv b.xlsx --common
python -m eiskit drt    a.csv b.xlsx
```

## Data layout

Use one row per frequency, with three columns: **frequency (Hz), Z′ (Ω), Z″ or −Z″ (Ω)**.

```
Frequency (Hz)   Z' (Ohm)   -Z'' (Ohm)
100000           2.549      0.383
10000            2.561      0.081
```

- **Headers** are optional.
- **Separators:** tabs, commas, semicolons or spaces. Decimal commas are understood.
- **Sign of the imaginary column:** taken from the header (`-Z''`, `-Im(Z)` …) or otherwise from the data. You can also set it with `fmt="z"` or `fmt="minus_z"`.
- **|Z| and phase:** use `fmt="polar"` for files with columns f, |Z| and phase.
- **Several samples in one file:** put the three-column blocks side by side, with each sample's name in the cell above its block.

## Robust fitting – no need to delete points

Real spectra contain spikes, drift and instrument artefacts. With `robust=True` (the default) the fit follows the *trend* of the data:
1. A soft-L1 fit is made first.
2. It is refined with a Cauchy loss whose scale is estimated from the median deviation.
3. Points that deviate by more than 4 robust standard deviations are down-weighted, reported in `result.outliers`, drawn hollow in plots, and excluded from the statistics.

**Synthetic test** (true circuit plus 3 spikes and 6 inductive artefact points): an ordinary fit is off by up to 60 % in one parameter. The robust fit finds exactly the 9 bad points and recovers every parameter within 2.5 %.

**Real CoMoO₄ data**, with no highlighting or cropping: the robust fit reproduces the charge-transfer resistances previously obtained from hand-cleaned data to within 0.3 %.

The Kramers–Kronig test is also robust. Instead of failing as a whole, it lists the frequencies that violate KK (`kk.failed_frequencies`), which are usually a few glitches or the top of the instrument's range.

## Circuit notation

You can write circuits in any of these forms:

| Meaning | Write |
|---|---|
| series | `A-B` · `A+B` · `AB` · `[A B]` |
| parallel | `p(A,B)` · `A/B` · `A\|B` · `A\|\|B` · `(A)p(B)` |
| grouping | `( … )` |
| Boukamp CDC (detected automatically) | `R(QR)(QR)`, `R(Q[RW])`: `( )` = parallel, `[ ]` = series |

- **Precedence:** parallel binds more strongly than series, so `R1+Q1/R2` = R1 + (Q1 ∥ R2).
- **Labels:** these are optional; numbers are added automatically. You can use common subscripts (`Rs`, `Rct`, `Rsei`, `Rf`, `Cdl`, `CPEdl`) or `_anything` (`R_pore`).
- **Checking the reading:** the interpretation is always shown in plain form. For example, `(R)p(CPE-Ws)` → `[R0 ∥ (CPE1 + Ws1)]`.

| Element | Impedance | Parameters |
|---|---|---|
| `R` | R | R (Ω) |
| `C` | 1/(jωC) | C (F) |
| `L` | jωL | L (H) |
| `Ls` | jωL, L may be < 0 | high-frequency artefact correction |
| `CPE` / `Q` | 1/[Q(jω)ⁿ] | Q (S·sⁿ) = CPE-T, n = CPE-P (0.3–1) |
| `W` | σ(1−j)/√ω | σ |
| `Ws` / `T` | R·tanh√(jωτ)/√(jωτ) | R (Ω), τ (s): finite-length (transmissive) diffusion |
| `Wo` / `O` | R·coth√(jωτ)/√(jωτ) | R (Ω), τ (s): finite-space (reflective) diffusion |
| `G` | R/√(1+jωτ) | R (Ω), τ (s): Gerischer |
| `TLM` | √(R_ion Z_i)·coth√(R_ion/Z_i), Z_i = R_ct ∥ CPE | de Levie porous electrode |
| `TLMQ` | as TLM, with Z_i = CPE | blocking porous electrode |

## Circuit library

`ek.literature_table()` lists **64 named circuits**, each with its typical application and the meaning of its elements. They include:
- the Randles family, with semi-infinite, finite-length or finite-space diffusion;
- blocking electrodes;
- Voigt circuits with 2–4 time constants;
- battery models (SEI, charge transfer, diffusion, intercalation capacitance);
- the brick-layer model;
- coated-metal and nested porous-film circuits;
- the Armstrong–Henderson / Harrington–Conway adsorbed-intermediate model;
- Maxwell branches and surface-state models for photoelectrodes;
- inductive loops, cable inductance, Gerischer and SOFC MIEC electrodes, de Levie transmission lines, and stray capacitance.

`search()` fits all of them. `fit_all(..., "search-common")` picks the single circuit that works best for every sample together, so that parameters can be compared between samples. The generated libraries `"extended"` (about 290 circuits) and `"exhaustive"` (about 900) are also available.

## Methods

**Fitting.** Complex non-linear least squares on real and imaginary parts together, with modulus weighting (1/|Z|).
- Python uses scipy's trust-region least squares with an analytic Jacobian.
- The web app uses a projected Levenberg–Marquardt method with the same loss functions.
- Parameters stay physical: R, C, Q, τ > 0, and 0.3 ≤ n ≤ 1.
- Several starting points are tried.

**Uncertainties.** Standard errors come from s²(JᵀJ)⁻¹; the 95 % confidence interval is p ± 1.96·SD.

**Validity checks.** A fit is marked not valid if any parameter:
- has a relative error above 100 %,
- has no influence on the fit,
- sits at a bound, or
- is a resistance that is effectively infinite or negligible.

Warnings (not failures) are given for n < 0.5, strongly correlated parameters (|r| > 0.98) and time constants outside the measured frequency window.

**Model selection.** Valid fits are ranked first, then by AICc.

**Effective capacitance.** C_eff = Q^(1/n)·R^((1−n)/n) (Hsu & Mansfeld, *Corrosion* 57 (2001) 747).

**Kramers–Kronig test.** Lin-KK (Schönleber et al., *Electrochim. Acta* 131 (2014) 20), with at least one RC element per decade, the μ criterion relaxed if needed, and Cauchy re-weighting.

**DRT.** The impedance is modelled as Z = R∞ + jωL + 1/(jωC) + ∫γ(ln τ)/(1+jωτ) d ln τ with γ ≥ 0. γ is found by Tikhonov regularisation with a second-derivative penalty and non-negative least squares (cf. Wan et al., *Electrochim. Acta* 184 (2015) 483).
- **λ** is chosen as the smoothest DRT whose reconstruction error is within 25 % of the best attainable.
- **Peaks:** each peak's position gives f = 1/(2πτ), and its area gives that process's resistance.
- **Extrapolation:** peaks outside the measured frequency window are reported but not counted.

## Testing

```bash
python -m pytest tests
```
The tests cover notation, the library, the analytic Jacobian, parameter recovery, robust fitting, text input, Kramers–Kronig, DRT, and multi-sample tables.

## Licence

MIT (see `LICENSE`). If you use eiskit in published work, please cite the methods referenced above.
