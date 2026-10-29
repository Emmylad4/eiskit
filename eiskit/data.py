"""
Impedance data: the :class:`Spectrum` container and all ways of getting data in.

* :func:`read_spectra`  – CSV / TXT / XLSX with one or several samples side by side
* :func:`read_files`    – several files, one sample each (name = file name)
* :func:`from_text`     – text pasted from Excel, Origin, an instrument export ...
* :func:`from_columns`  – separate lists (or pasted columns) of f, Z', Z''
* :func:`prompt_spectrum` – asks for the data interactively in a console / notebook

Expected data layout
--------------------
Three columns per sample, one row per frequency::

    Frequency (Hz)   Z' (Ohm)   -Z'' (Ohm)
    100000           2.549      0.383
    79432.8          2.551      0.306
    ...

* The imaginary column may be Z'' (negative for capacitive data) or -Z''; the sign is
  taken from the header ("-Z''", "-Im") or, without a header, from the data.
* ``fmt="polar"`` reads frequency, |Z|, phase (degrees) instead.
* Several samples: blocks of three columns next to each other, optionally separated by an
  empty column, with the sample name in the cell above each block.
"""
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Spectrum:
    f: np.ndarray
    Z: np.ndarray
    name: str = ""
    exclude: np.ndarray = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.f = np.asarray(self.f, float)
        self.Z = np.asarray(self.Z, complex)
        if len(self.f) != len(self.Z):
            raise ValueError("frequency and impedance arrays have different lengths")
        self.exclude = (np.zeros(len(self.f), bool) if self.exclude is None
                        else np.asarray(self.exclude, bool))
        ok = np.isfinite(self.f) & np.isfinite(self.Z) & (self.f > 0)
        o = np.argsort(self.f[ok])[::-1]                      # high -> low frequency
        self.f, self.Z, self.exclude = self.f[ok][o], self.Z[ok][o], self.exclude[ok][o]
        _, first = np.unique(self.f, return_index=True)       # drop duplicate frequencies
        if len(first) != len(self.f):
            keep = np.sort(first)
            self.f, self.Z, self.exclude = self.f[keep], self.Z[keep], self.exclude[keep]
            o = np.argsort(self.f)[::-1]
            self.f, self.Z, self.exclude = self.f[o], self.Z[o], self.exclude[o]

    # ---- views ---------------------------------------------------------------------
    @property
    def used(self):
        m = ~self.exclude
        return Spectrum(self.f[m], self.Z[m], self.name, None, dict(self.meta))

    @property
    def phase(self):
        """Phase angle in degrees (negative for capacitive behaviour)."""
        return np.degrees(np.arctan2(self.Z.imag, self.Z.real))

    @property
    def modulus(self):
        return np.abs(self.Z)

    @property
    def label(self):
        return self.name

    def __len__(self):
        return len(self.f)

    def __repr__(self):
        ex = f", {int(self.exclude.sum())} excluded" if self.exclude.any() else ""
        return f"Spectrum('{self.name}', {len(self)} points{ex}, {self.f.min():.3g}–{self.f.max():.3g} Hz)"

    # ---- editing (each returns a new Spectrum) ------------------------------------
    def _copy(self, ex):
        return Spectrum(self.f, self.Z, self.name, ex, dict(self.meta))

    def crop(self, fmin=None, fmax=None):
        """Exclude points outside [fmin, fmax]."""
        ex = self.exclude.copy()
        if fmin is not None:
            ex |= self.f < fmin
        if fmax is not None:
            ex |= self.f > fmax
        return self._copy(ex)

    def exclude_frequencies(self, freqs, rtol=0.01):
        ex = self.exclude.copy()
        for fx in np.atleast_1d(freqs):
            ex |= np.abs(self.f / fx - 1) < rtol
        return self._copy(ex)

    def exclude_inductive_tail(self):
        """Exclude the contiguous high-frequency points with Z'' > 0 (cable inductance)."""
        ex = self.exclude.copy()
        i = 0
        while i < len(self.f) and self.Z.imag[i] > 0:
            ex[i] = True
            i += 1
        return self._copy(ex)

    def include_all(self):
        return self._copy(None)

    def to_frame(self):
        return pd.DataFrame({"f_Hz": self.f, "Zre_Ohm": self.Z.real, "minusZim_Ohm": -self.Z.imag,
                             "absZ_Ohm": self.modulus, "minus_phase_deg": -self.phase,
                             "excluded": self.exclude})


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float, np.floating)):
        return float(v) if np.isfinite(v) else None
    s = str(v).strip().replace("\u2212", "-")
    if not s:
        return None
    if re.fullmatch(r"[-+]?\d+,\d+([eE][-+]?\d+)?", s):          # decimal comma
        s = s.replace(",", ".")
    try:
        x = float(s)
        return x if np.isfinite(x) else None
    except ValueError:
        return None


def _imag_is_minus(header, values):
    h = (header or "").replace(" ", "").lower()
    if h.startswith("-") or "-z''" in h or "-im" in h or "-zim" in h or "-z\"" in h or "minus" in h:
        return True
    if h and ("z''" in h or "zim" in h or "im(z)" in h or "imag" in h or 'z"' in h):
        return False
    return bool(np.median(values) > 0)                     # most data are capacitive


def _make(f, a, b, name, fmt, imag_header="", exclude=None, meta=None):
    f, a, b = (np.asarray(x, float) for x in (f, a, b))
    if fmt == "polar":                                       # |Z|, phase in degrees
        ph = np.radians(b)
        if np.median(b) > 0:                                 # given as -phase
            ph = -ph
        Z = a * np.cos(ph) + 1j * a * np.sin(ph)
    else:
        minus = {"minus_z": True, "z": False}.get(fmt, None)
        if minus is None:
            minus = _imag_is_minus(imag_header, b)
        Z = a + 1j * (-b if minus else b)
    return Spectrum(f, Z, name, exclude, meta or {})


# ---------------------------------------------------------------------------------------
# pasted text and columns
# ---------------------------------------------------------------------------------------
def from_text(text, name="Sample", fmt="auto"):
    """
    Spectrum from pasted text: one row per frequency, columns frequency, Z', Z'' (or -Z'').
    Separators: tab, comma, semicolon or spaces. Header lines are allowed (and used to read
    the sign of the imaginary part). ``fmt``: "auto", "z" (column is Z''), "minus_z"
    (column is -Z'') or "polar" (|Z| and phase in degrees).
    """
    rows, header = [], ""
    for line in str(text).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t")
        elif ";" in line:
            parts = line.split(";")                         # European CSV (decimal commas allowed)
        elif line.count(",") >= 2:
            parts = line.split(",")                         # ordinary CSV
        else:
            parts = line.split()                            # spaces (decimal commas allowed)
        vals = [_num(p) for p in parts if p.strip() != ""]
        if len(vals) >= 3 and all(v is not None for v in vals[:3]):
            rows.append(vals[:3])
        elif not rows:
            header = line
    if len(rows) < 3:
        raise ValueError("need at least 3 rows with three numbers (frequency, Z', Z'')")
    a = np.array(rows)
    hdr_parts = re.split(r"\t|;|,(?!\d)|\s{2,}", header) if header else []
    imag_hdr = hdr_parts[2] if len(hdr_parts) >= 3 else header
    return _make(a[:, 0], a[:, 1], a[:, 2], name, fmt, imag_hdr, meta={"source": "pasted text"})


def _as_array(x):
    if isinstance(x, str):
        return np.array([v for v in (_num(t) for t in re.split(r"[\s,;]+", x.strip())) if v is not None])
    return np.asarray(x, float)


def from_columns(frequency, real, imaginary, name="Sample", imaginary_is="auto"):
    """
    Spectrum from three separate columns (lists, arrays or pasted text, one value per line).
    ``imaginary_is``: "auto", "z" (values are Z'') or "minus_z" (values are -Z'').
    """
    f, a, b = _as_array(frequency), _as_array(real), _as_array(imaginary)
    if not (len(f) == len(a) == len(b)):
        raise ValueError(f"columns have different lengths: {len(f)}, {len(a)}, {len(b)}")
    return _make(f, a, b, name, imaginary_is, meta={"source": "columns"})


def prompt_spectrum(name=None):
    """
    Ask for one spectrum interactively (works in a terminal and in Jupyter):
    either a file path, or the data pasted as three columns and finished with an empty line.
    """
    name = name or input("Sample name: ").strip() or "Sample"
    print("Paste the data (columns: frequency, Z', Z'' or -Z''; header optional),\n"
          "then press Enter on an empty line.  Or type a file path and press Enter.")
    first = input().strip().strip('"')
    if first and os.path.exists(first):
        sp = next(iter(read_spectra(first).values()))
        sp.name = name
        return sp
    lines = [first]
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return from_text("\n".join(lines), name)


# ---------------------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------------------
def _is_highlight(cell):
    fill = cell.fill
    if fill is None or fill.fill_type in (None, "none"):
        return False
    col = fill.fgColor
    if col is not None and col.type == "rgb" and isinstance(col.rgb, str):
        try:
            r, g, b = (int(col.rgb[-6:][i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return True
        return (r, g, b) != (255, 255, 255) and r > 180 and g > 180 and b < 170
    if col is not None and col.type == "indexed":
        return col.indexed in (5, 13, 26, 43, 51)
    return fill.fill_type == "solid"


def _grid(path, sheet, use_highlights):
    p = str(path).lower()
    if p.endswith((".xlsx", ".xlsm")):
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
        rows = list(ws.iter_rows())
        grid = [[c.value for c in r] for r in rows]
        fills = [[_is_highlight(c) if use_highlights else False for c in r] for r in rows]
    elif p.endswith(".xls"):
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        grid, fills = df.where(pd.notna(df), None).values.tolist(), None
    else:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.read().splitlines()
        sep = "\t" if any("\t" in l for l in lines[:20]) else ";" if any(";" in l for l in lines[:20]) else \
              "," if any("," in l for l in lines[:20]) else None
        grid = [(l.split(sep) if sep else l.split()) for l in lines]
        fills = None
    n = max((len(r) for r in grid), default=0)
    grid = [list(r) + [None] * (n - len(r)) for r in grid]
    if fills is not None:
        fills = [list(r) + [False] * (n - len(r)) for r in fills]
    return grid, fills


def _blocks(grid):
    """Find the start column of every 3-column data block and its first data row."""
    ncol = len(grid[0]) if grid else 0
    starts = []
    for c in range(ncol - 2):
        # first row where this column and the next two are numeric
        for r in range(min(len(grid), 60)):
            if all(_num(grid[r][c + k]) is not None for k in range(3)):
                prev_numeric = c > 0 and _num(grid[r][c - 1]) is not None
                if not prev_numeric and (not starts or c >= starts[-1][0] + 3):
                    starts.append((c, r))
                break
    return starts


def read_spectra(path, sheet=0, fmt="auto", use_highlights=True, names=None):
    """
    Read every sample in a CSV / TXT / XLSX file. Returns ``{name: Spectrum}``.

    Each sample is a block of three columns (frequency, Z', Z'' or -Z''); several blocks may
    sit side by side. The sample name is taken from the top-most text cell above each block
    (otherwise the file name, numbered if needed); ``names`` overrides them. Yellow-filled
    cells in .xlsx files are marked as excluded when ``use_highlights`` is True (robust
    fitting usually makes manual exclusion unnecessary).
    """
    grid, fills = _grid(path, sheet, use_highlights)
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    out = {}
    blocks = _blocks(grid)
    if not blocks:
        raise ValueError(f"no three-column numeric data found in {path}")
    for k, (c, r0) in enumerate(blocks):
        texts = [str(grid[r][c]).strip() for r in range(r0) if grid[r][c] not in (None, "")
                 and _num(grid[r][c]) is None]
        imag_hdr = next((str(grid[r][c + 2]) for r in range(r0 - 1, -1, -1)
                         if grid[r][c + 2] not in (None, "") and _num(grid[r][c + 2]) is None), "")
        header_words = ("freq", "hz", "z'", "zre", "real", "ohm")
        labels = [t for t in texts if not any(w in t.lower() for w in header_words)]
        name = labels[0] if labels else (stem if len(blocks) == 1 else f"{stem} {k + 1}")
        rows, hl = [], []
        for r in range(r0, len(grid)):
            vals = [_num(grid[r][c + j]) for j in range(3)]
            if None in vals:
                if all(v is None for v in vals):
                    break
                continue
            rows.append(vals)
            hl.append(bool(fills and any(fills[r][c + j] for j in range(3))))
        a = np.array(rows)
        m = re.search(r"@\s*([-+]?\d*\.?\d+)\s*V", name)
        meta = {"source": str(path), **({"potential_V": float(m.group(1))} if m else {})}
        if names and k < len(names):
            name = names[k]
        while name in out:
            name += "'"
        out[name] = _make(a[:, 0], a[:, 1], a[:, 2], name, fmt, imag_hdr, np.array(hl), meta)
    return out


def read_spectrum(path, **kw):
    """Read the first sample of a file."""
    return next(iter(read_spectra(path, **kw).values()))


def read_files(paths, names=None, **kw):
    """Read several files (one sample each); names default to the file names."""
    out = {}
    for i, p in enumerate(paths):
        sp = read_spectrum(p, **kw)
        sp.name = names[i] if names else os.path.splitext(os.path.basename(p))[0]
        out[sp.name] = sp
    return out
