"""
Figures: fit dashboard, circuit schematic, Nyquist / Bode / DRT comparisons of samples.

All functions return a matplotlib Figure; :func:`save_figure` writes SVG/TIFF/PNG/PDF.
Publication style: Arial, thick frame, outward ticks (see ``STYLE``).
"""
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

STYLE = {"font.family": "sans-serif",
         "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
         "svg.fonttype": "none", "axes.linewidth": 2.5, "xtick.direction": "out", "ytick.direction": "out",
         "xtick.major.size": 6, "ytick.major.size": 6, "xtick.major.width": 2.2, "ytick.major.width": 2.2,
         "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.labelsize": 18}


def save_figure(fig, stem, formats=("svg", "tiff"), dpi=600):
    """Save ``fig`` as ``stem.svg``, ``stem.tiff`` (LZW, ``dpi``), etc. Returns the paths."""
    paths = []
    for fmt in formats:
        p = f"{stem}.{fmt}"
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt in ("tif", "tiff") else {}
        fig.savefig(p, format="tiff" if fmt == "tif" else fmt, dpi=dpi, **kw)
        paths.append(p)
    return paths


def _pow_label(v, _):
    if v <= 0:
        return ""
    e = int(round(np.log10(v)))
    return rf"10$^{{{e}}}$" if np.isclose(v, 10.0 ** e) else ""


def _freq_label(v, _):
    if v <= 0:
        return ""
    e = int(round(np.log10(v)))
    if not np.isclose(v, 10.0 ** e):
        return ""
    return f"{v:g}" if -2 <= e <= 2 else rf"10$^{{{e}}}$"


# ---------------------------------------------------------------------------------------
# circuit schematic
# ---------------------------------------------------------------------------------------
_EW, _EH, _GAP = 1.8, 1.1, 0.35


def _size(node):
    if node[0] == "e":
        return _EW, _EH
    s = [_size(c) for c in node[1]]
    if node[0] == "s":
        return sum(a for a, _ in s), max(b for _, b in s)
    return max(a for a, _ in s) + 2 * _GAP, sum(b for _, b in s)


def _element(ax, etype, name, x, y):
    lw, a, b = 1.5, x + 0.45, x + _EW - 0.45
    m = (a + b) / 2
    if etype in ("C", "CPE"):
        ax.plot([x, m - 0.08], [y, y], "k", lw=lw); ax.plot([m + 0.08, x + _EW], [y, y], "k", lw=lw)
        for xx in (m - 0.08, m + 0.08):
            if etype == "C":
                ax.plot([xx, xx], [y - 0.25, y + 0.25], "k", lw=lw + 0.6)
            else:
                ax.plot([xx + 0.1, xx, xx + 0.1], [y + 0.25, y, y - 0.25], "k", lw=lw + 0.4)
    elif etype in ("L", "Ls"):
        ax.plot([x, a], [y, y], "k", lw=lw); ax.plot([b, x + _EW], [y, y], "k", lw=lw)
        d = (b - a) / 4
        for i in range(4):
            ax.add_patch(mpl.patches.Arc((a + d * (i + 0.5), y), d, 0.4, theta1=0, theta2=180, lw=lw))
    else:
        ax.plot([x, a], [y, y], "k", lw=lw); ax.plot([b, x + _EW], [y, y], "k", lw=lw)
        ax.add_patch(Rectangle((a, y - 0.16), b - a, 0.32, fill=etype != "R", fc="#dde6f5", ec="k", lw=lw))
        if etype != "R":
            ax.text(m, y, etype, ha="center", va="center", fontsize=7 if len(etype) > 2 else 8)
    ax.text(m, y + 0.33, name, ha="center", va="bottom", fontsize=9)


def _draw(ax, node, x, y):
    w, h = _size(node)
    if node[0] == "e":
        _element(ax, node[1], node[2], x, y)
    elif node[0] == "s":
        for c in node[1]:
            _draw(ax, c, x, y); x += _size(c)[0]
    else:
        cy, ys = y + h / 2, []
        for c in node[1]:
            cw, ch = _size(c); yc = cy - ch / 2; ys.append(yc)
            _draw(ax, c, x + _GAP, yc)
            ax.plot([x + _GAP + cw, x + w - _GAP], [yc, yc], "k", lw=1.5)
            cy -= ch
        for xx in (x + _GAP, x + w - _GAP):
            ax.plot([xx, xx], [min(ys), max(ys)], "k", lw=1.5)
        ax.plot([x, x + _GAP], [y, y], "k", lw=1.5); ax.plot([x + w - _GAP, x + w], [y, y], "k", lw=1.5)


def draw_circuit(circuit, ax=None, title=None):
    """Draw the schematic of a :class:`~eiskit.circuit.Circuit` (or circuit string)."""
    from .circuit import Circuit
    c = circuit if isinstance(circuit, Circuit) else Circuit(circuit)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2.2))
    w, h = _size(c.tree)
    ax.plot([-0.3, 0], [0, 0], "k", lw=1.5); ax.plot([w, w + 0.3], [0, 0], "k", lw=1.5)
    ax.plot([-0.3, w + 0.3], [0, 0], "ko", ms=5, ls="none")
    _draw(ax, c.tree, 0, 0)
    ax.set_xlim(-0.6, w + 0.6); ax.set_ylim(-h / 2 - 0.3, h / 2 + 0.5)
    ax.set_aspect("equal"); ax.axis("off")
    if title is None:
        title = c.string
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


# ---------------------------------------------------------------------------------------
# single-fit dashboard
# ---------------------------------------------------------------------------------------
def plot_fit(res, spectrum=None, kk=None):
    """Nyquist, Bode |Z| and phase, residuals, circuit and parameter table for one fit."""
    f, Z = res.f, res.Z
    fd = np.logspace(np.log10(f.min()), np.log10(f.max()), 300)
    Zs = res.predict(fd)
    fig = plt.figure(figsize=(15, 8))
    g = fig.add_gridspec(2, 3)
    out = res.outliers if res.outliers is not None else np.zeros(len(f), bool)
    fig.suptitle(f"{res.name}   {res.circuit.string}   RMS {res.rms_pct:.3f} %"
                 + (f"   ({res.n_outliers} outlier(s) down-weighted)" if res.n_outliers else ""), fontsize=12)
    a = fig.add_subplot(g[0, 0])
    a.plot(Z.real[~out], -Z.imag[~out], "o", c="C0", ms=5, label="data")
    if out.any():
        a.plot(Z.real[out], -Z.imag[out], "o", mfc="white", mec="C0", ms=5, label="outlier (down-weighted)")
    if spectrum is not None and spectrum.exclude.any():
        ex = spectrum.exclude
        a.plot(spectrum.Z.real[ex], -spectrum.Z.imag[ex], "x", c="0.6", label="excluded")
    a.plot(Zs.real, -Zs.imag, "r-", label="fit"); a.set_aspect("equal", adjustable="datalim")
    a.set(xlabel="Z' / Ω", ylabel="-Z'' / Ω", title="Nyquist"); a.legend(fontsize=8)
    a = fig.add_subplot(g[0, 1]); a.loglog(f, np.abs(Z), "o", ms=4); a.loglog(fd, np.abs(Zs), "r-")
    a.set(xlabel="f / Hz", ylabel="|Z| / Ω", title="Bode |Z|")
    a = fig.add_subplot(g[0, 2]); a.semilogx(f, -np.angle(Z, deg=True), "o", ms=4)
    a.semilogx(fd, -np.angle(Zs, deg=True), "r-"); a.set(xlabel="f / Hz", ylabel="-phase / °", title="Bode phase")
    a = fig.add_subplot(g[1, 0])
    a.semilogx(f, 100 * res.res_re, "o-", ms=3, label="Δ real")
    a.semilogx(f, 100 * res.res_im, "s-", ms=3, label="Δ imag")
    if kk is not None:
        a.semilogx(kk.f, 100 * kk.res_im, ":", c="0.5", label="KK Δ imag")
    a.axhline(0, c="k", lw=0.6); a.set(xlabel="f / Hz", ylabel="residual / % of |Z|", title="Residuals")
    a.legend(fontsize=7)
    draw_circuit(res.circuit, fig.add_subplot(g[1, 1]))
    a = fig.add_subplot(g[1, 2]); a.axis("off")
    txt = ["Parameter        Value        ±SD      err%"]
    for nm, v, s, e, u in zip(res.circuit.names, res.p, res.sd, res.rel, res.circuit.units):
        txt.append(f"{nm:<12}{v:>11.4g} {s:>10.3g} {e:>7.1f}  {u}")
    txt += ["", f"chi2 = {res.chi2:.3g}   red. chi2 = {res.red_chi2:.3g}",
            f"RMS = {res.rms_pct:.3f} %   R2 = {res.r2:.5f}", f"AICc = {res.aicc:.1f}"]
    txt += [f"{d['element']}: tau = {d['tau']:.3g} s, f_c = {d['f_c']:.3g} Hz" +
            (f", C_eff = {d['C_eff']:.3g} F" if "C_eff" in d else "") for d in res.derived]
    a.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=8, transform=a.transAxes)
    for ax_ in fig.axes[:4]:
        ax_.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_search(search_result, n=6):
    """Nyquist + phase of the ``n`` best circuits of a search."""
    fits = search_result.fits[:n]
    cols = 3; rows = int(np.ceil(len(fits) / cols))
    fig, axes = plt.subplots(2 * rows, cols, figsize=(5 * cols, 5.2 * rows),
                             gridspec_kw={"height_ratios": [2, 1] * rows}, squeeze=False)
    for i in range(rows * cols):
        a1, a2 = axes[2 * (i // cols), i % cols], axes[2 * (i // cols) + 1, i % cols]
        if i >= len(fits):
            a1.axis("off"); a2.axis("off"); continue
        r = fits[i]
        a1.plot(r.Z.real, -r.Z.imag, "o", ms=3, mfc="none"); a1.plot(r.Zfit.real, -r.Zfit.imag, "r-")
        a1.set_aspect("equal", adjustable="datalim")
        a1.set_title(f"#{i + 1} {r.circuit.string}\nRMS {r.rms_pct:.3f} %  AICc {r.aicc:.0f}" +
                     ("" if r.valid else "  ✗"), fontsize=8)
        a2.semilogx(r.f, -np.angle(r.Z, deg=True), "o", ms=3, mfc="none")
        a2.semilogx(r.f, -np.angle(r.Zfit, deg=True), "r-")
        for a in (a1, a2):
            a.tick_params(labelsize=7); a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------
# comparison figures (publication style)
# ---------------------------------------------------------------------------------------
PALETTE = ["#E8413C", "#F39C3D", "#4AA83B", "#27A9E1", "#9B72CF", "#8C564B", "#E377C2", "#7F7F7F",
           "#BCBD22", "#17BECF"]
MARKERS = ["D", "o", "<", "*", "s", "^", "v", "p", "h", ">"]


def _items(obj):
    """Accept {name: FitResult}, {name: Spectrum}, Results, or a single object."""
    from .data import Spectrum
    from .fitting import FitResult, Results
    if isinstance(obj, Results):
        return {n: (obj.spectra[n], r) for n, r in obj.fits.items()}
    if isinstance(obj, (Spectrum, FitResult)):
        obj = {getattr(obj, "name", "") or "sample": obj}
    out = {}
    for n, v in obj.items():
        out[n] = (None, v) if isinstance(v, FitResult) else (v, None)
    return out


def _data(sp, r):
    if r is not None:
        return r.f, r.Z, (r.outliers if r.outliers is not None else np.zeros(len(r.f), bool))
    u = sp.used
    return u.f, u.Z, np.zeros(len(u.f), bool)


def _marker(i, ms):
    m = MARKERS[i % len(MARKERS)]
    return m, ms * (1.5 if m == "*" else 1.0)


def plot_nyquist(obj, palette=None, smooth=True, equal_aspect=True, circuit_inset=None,
                 legend_loc="upper left", size_cm=(12, 10), ms=6, zoom_to_trend=True):
    """
    Nyquist plot comparing samples: data as markers, fits as lines in the same colour
    (outliers of robust fits drawn hollow). ``obj``: Results, {name: FitResult} or
    {name: Spectrum}. ``circuit_inset``: a circuit (any notation) drawn as an inset.
    ``zoom_to_trend``: axis limits follow the fitted points, so that far-off artefact
    points (down-weighted outliers) do not shrink the plot.
    """
    items = _items(obj)
    pal = palette or PALETTE
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54))
        keep_x, keep_y = [], []
        for i, (n, (sp, r)) in enumerate(items.items()):
            col = pal[i % len(pal)]
            m, msz = _marker(i, ms)
            f, Z, out = _data(sp, r)
            keep_x += list(Z.real[~out]); keep_y += list(-Z.imag[~out])
            if r is not None:
                fd = np.logspace(np.log10(f[~out].min()), np.log10(f[~out].max()), 400) if smooth else f
                Zl = r.predict(fd)
                ax.plot(Zl.real, -Zl.imag, "-", c=col, lw=1.6, zorder=2)
            ax.plot(Z.real[~out], -Z.imag[~out], m, ls="none", ms=msz, mfc=col, mec=col, zorder=3,
                    label=n)
            if out.any():
                ax.plot(Z.real[out], -Z.imag[out], m, ls="none", ms=msz, mfc="white", mec=col, zorder=3)
        ax.set_xlabel("Z' (Ω)"); ax.set_ylabel("-Z'' (Ω)")
        if zoom_to_trend and keep_x:
            x0, x1, y0, y1 = min(keep_x), max(keep_x), min(min(keep_y), 0), max(keep_y)
            px, py = 0.06 * (x1 - x0), 0.08 * (y1 - y0)
            ax.set_xlim(x0 - px, x1 + px); ax.set_ylim(y0 - py, y1 + py)
            if equal_aspect:                    # figure shaped to the data (no empty bands)
                ratio = (y1 - y0 + 2 * py) / (x1 - x0 + 2 * px)
                wid = size_cm[0] / 2.54
                fig.set_size_inches(wid, min(max(wid * ratio * 0.82 + 1.0, 0.55 * wid), 1.25 * wid))
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box" if zoom_to_trend else "datalim")
        ax.tick_params(top=False, right=False)
        ax.legend(frameon=False, fontsize=11, loc=legend_loc, handletextpad=0.3)
        fig.tight_layout()
        if circuit_inset:                       # inset in the upper right of the plot area
            fig.canvas.draw()
            bb = ax.get_position()
            ins = fig.add_axes([bb.x0 + 0.52 * bb.width, bb.y0 + 0.66 * bb.height, 0.46 * bb.width, 0.32 * bb.height])
            draw_circuit(circuit_inset, ins, title="")
    return fig


def plot_bode(obj, palette=None, smooth=True, size_cm=(13, 15), ms=5):
    """
    Standard Bode plot: |Z| (log scale) and -phase versus frequency, one colour per sample,
    data as markers and fits as lines.
    """
    items = _items(obj)
    pal = palette or PALETTE
    with mpl.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54))
        for i, (n, (sp, r)) in enumerate(items.items()):
            col = pal[i % len(pal)]
            m, msz = _marker(i, ms)
            f, Z, out = _data(sp, r)
            for ax, y in ((a1, np.abs(Z)), (a2, -np.angle(Z, deg=True))):
                ax.plot(f[~out], y[~out], m, ls="none", ms=msz, mfc=col, mec=col, label=n, zorder=3)
                if out.any():
                    ax.plot(f[out], y[out], m, ls="none", ms=msz, mfc="white", mec=col, zorder=3)
            if r is not None:
                fd = np.logspace(np.log10(f.min()), np.log10(f.max()), 400) if smooth else f
                Zl = r.predict(fd)
                a1.plot(fd, np.abs(Zl), "-", c=col, lw=1.4, zorder=2)
                a2.plot(fd, -np.angle(Zl, deg=True), "-", c=col, lw=1.4, zorder=2)
        a1.set_xscale("log"); a1.set_yscale("log")
        a1.set_ylabel("|Z| (Ω)"); a2.set_ylabel("-Phase (°)"); a2.set_xlabel("Frequency (Hz)")
        a2.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
        a2.xaxis.set_major_formatter(FuncFormatter(_pow_label))
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        a1.yaxis.set_major_formatter(plain); a1.yaxis.set_minor_formatter(
            FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "2345" else ""))
        for ax in (a1, a2):
            ax.tick_params(top=False, right=False, which="both")
        a1.legend(frameon=False, fontsize=10)
        fig.tight_layout()
    return fig


def plot_drt(drts, palette=None, x="frequency", annotate=True, show_extrapolation=False, size_cm=(14, 10)):
    """
    DRT of several samples: gamma vs frequency (``x="frequency"``) or time constant
    (``x="tau"``). ``drts``: a DRTResult or {name: DRTResult}. Major peaks are marked (▼);
    only the measured frequency window is drawn solid.
    """
    from .drt import DRTResult
    if isinstance(drts, DRTResult):
        drts = {drts.name or "sample": drts}
    pal = palette or PALETTE
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(size_cm[0] / 2.54, size_cm[1] / 2.54))
        for i, (n, d) in enumerate(drts.items()):
            col = pal[i % len(pal)]
            xv = d.f_tau if x == "frequency" else d.tau
            inside = (d.f_tau >= d.f.min() * 0.999) & (d.f_tau <= d.f.max() * 1.001)
            g_in = np.where(inside, d.gamma, np.nan)
            ax.plot(xv, g_in, "-", c=col, lw=2, label=n)
            if show_extrapolation:
                ax.plot(xv, np.where(inside, np.nan, d.gamma), "--", c=col, lw=1, alpha=0.6)
            ax.fill_between(xv, np.nan_to_num(g_in), color=col, alpha=0.12, lw=0)
            if annotate and len(d.peaks):
                pk = d.peaks[d.peaks["major"]]
                ax.plot(pk["f_Hz"] if x == "frequency" else pk["tau_s"], pk["gamma_Ohm"], "v", c=col, ms=7)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
        ax.xaxis.set_major_formatter(FuncFormatter(_freq_label)); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Frequency (Hz)" if x == "frequency" else "τ (s)")
        ax.set_ylabel("γ (Ω)")
        ax.set_ylim(bottom=0)
        ax.tick_params(top=False, right=False)
        ax.legend(frameon=False, fontsize=11)
        fig.tight_layout()
    return fig


def plot_drt_check(d):
    """DRT of one spectrum with the reconstructed spectrum (Nyquist) and its residuals."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    inside = (d.f_tau >= d.f.min() * 0.999) & (d.f_tau <= d.f.max() * 1.001)
    axs[0].semilogx(d.f_tau, np.where(inside, d.gamma, np.nan), "-", lw=2)
    axs[0].semilogx(d.f_tau, np.where(inside, np.nan, d.gamma), "--", lw=1, c="C0", alpha=0.6,
                    label="extrapolated (outside data)")
    axs[0].axvspan(d.f.min(), d.f.max(), color="0.93", zorder=0)
    axs[0].legend(fontsize=7)
    if len(d.peaks):
        for _, pk in d.peaks.iterrows():
            axs[0].plot(pk.f_Hz, pk.gamma_Ohm, "v" if pk.major else "x", c="C3", ms=8)
            if pk.major:
                axs[0].annotate(f"{pk.f_Hz:.3g} Hz\n{pk.R_Ohm:.3g} Ω", (pk.f_Hz, pk.gamma_Ohm),
                                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axs[0].set(xlabel="f = 1/(2πτ) / Hz", ylabel="γ / Ω", title=f"DRT  (λ = {d.lam:.2g})")
    axs[0].set_ylim(0, max(d.gamma.max(), 1e-12) * 1.35)
    axs[1].plot(d.Z.real, -d.Z.imag, "o", mfc="none", label="data")
    axs[1].plot(d.Z_drt.real, -d.Z_drt.imag, "r-", label="DRT reconstruction")
    axs[1].set_aspect("equal", adjustable="datalim"); axs[1].legend(fontsize=8)
    axs[1].set(xlabel="Z' / Ω", ylabel="-Z'' / Ω", title="Nyquist")
    r = (d.Z - d.Z_drt) / np.abs(d.Z)
    axs[2].semilogx(d.f, 100 * r.real, "o-", ms=3, label="Δ real")
    axs[2].semilogx(d.f, 100 * r.imag, "s-", ms=3, label="Δ imag")
    axs[2].axhline(0, c="k", lw=0.6); axs[2].legend(fontsize=8)
    axs[2].set(xlabel="f / Hz", ylabel="residual / %", title=f"Reconstruction RMS {d.rms_pct:.2f} %")
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle(d.name)
    fig.tight_layout()
    return fig
