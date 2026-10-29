"""
eiskit – equivalent-circuit fitting and DRT analysis of electrochemical impedance spectra.

Quick start::

    import eiskit as ek
    data = ek.read_spectra("samples.xlsx")              # {sample name: Spectrum}
    ek.lin_kk(data["NF/Ni3N"])                          # Kramers–Kronig validity
    res = ek.fit(data["NF/Ni3N"], "Rs-p(Rct,CPEdl)")    # any notation; robust by default
    s = ek.search(data["NF/Ni3N"])                      # rank >60 literature circuits
    allres = ek.fit_all(data, "R-p(R,CPE)")             # compare samples
    ek.plot_nyquist(allres); ek.plot_bode(allres); ek.plot_drt({n: ek.drt(s) for n, s in data.items()})

Data can also be pasted: ``ek.from_text(text, "name")``, ``ek.from_columns(f, zre, zim)``
or entered interactively with ``ek.prompt_spectrum()``. ``python -m eiskit app`` opens the
offline web app.
"""
__version__ = "2.0.0"

from .circuit import LITERATURE, Circuit, circuit_library, literature_table
from .data import Spectrum, from_columns, from_text, prompt_spectrum, read_files, read_spectra, read_spectrum
from .drt import DRTResult, drt
from .fitting import FitResult, Results, SearchResult, fit, fit_all, search
from .notation import NotationError, canonical, describe
from .plotting import (draw_circuit, plot_bode, plot_drt, plot_drt_check, plot_fit, plot_nyquist,
                       plot_search, save_figure)
from .report import comparison_table, tables, to_docx, to_excel
from .validation import KKResult, lin_kk

__all__ = ["Circuit", "LITERATURE", "circuit_library", "literature_table", "canonical", "describe",
           "NotationError", "Spectrum", "read_spectra", "read_spectrum", "read_files", "from_text",
           "from_columns", "prompt_spectrum", "lin_kk", "KKResult", "fit", "search", "fit_all", "FitResult",
           "SearchResult", "Results", "drt", "DRTResult", "plot_fit", "plot_search", "plot_nyquist",
           "plot_bode", "plot_drt", "plot_drt_check", "draw_circuit", "save_figure", "tables",
           "comparison_table", "to_excel", "to_docx", "open_app"]


def open_app():
    """Open the offline web app (eiskit_app.html) in the default browser."""
    import os
    import webbrowser
    path = os.path.join(os.path.dirname(__file__), "app", "eiskit_app.html")
    webbrowser.open("file://" + os.path.abspath(path))
    return path
