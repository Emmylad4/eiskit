"""
Command line:  python -m eiskit <command> [FILE ...] [options]

  app                                   open the offline web app in the browser
  circuits                              list the literature circuits
  parse   "NOTATION"                    show how a circuit notation is read
  fit     FILE [FILE ...] --circuit "Rs-p(Rct,CPEdl)"
  search  FILE [FILE ...] [--library literature|extended|all] [--common]
  drt     FILE [FILE ...]

Each FILE may hold one or several samples (3-column blocks side by side). Results (Excel,
Word, figures) are written to --out (default: eiskit_results).
"""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eiskit", description="EIS circuit fitting and DRT")
    ap.add_argument("command", choices=["app", "circuits", "parse", "fit", "search", "drt"])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--circuit")
    ap.add_argument("--library", default="literature", choices=["literature", "extended", "exhaustive", "all"])
    ap.add_argument("--common", action="store_true", help="one common circuit for all samples")
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--out", default="eiskit_results")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import eiskit as ek

    if a.command == "app":
        print("opening", ek.open_app()); return
    if a.command == "circuits":
        print(ek.literature_table().to_string()); return
    if a.command == "parse":
        for s in a.inputs:
            c = ek.Circuit(s)
            print(f"{s}  ->  {c.string}   [{c.explain()}]   parameters: {', '.join(c.names)}")
        return
    if not a.inputs:
        sys.exit("give at least one data file")
    data = {}
    for p in a.inputs:
        data.update(ek.read_spectra(p, use_highlights=False))
    os.makedirs(a.out, exist_ok=True)
    if a.command == "drt":
        drts = {n: ek.drt(s) for n, s in data.items()}
        for n, d in drts.items():
            print(n, d); print(d.peaks.to_string())
            d.to_frame().to_csv(os.path.join(a.out, f"drt_{n}.csv".replace("/", "_")), index=False)
        ek.save_figure(ek.plot_drt(drts), os.path.join(a.out, "DRT"))
    else:
        circuit = a.circuit if a.command == "fit" else ("search-common" if a.common else "search")
        res = ek.fit_all(data, circuit, robust=not a.no_robust,
                         search_kw={"circuits": a.library, "n_jobs": a.jobs})
        print(res)
        res.to_excel(os.path.join(a.out, "EIS_fit_results.xlsx"))
        figs = {}
        for n, r in res.fits.items():
            p = os.path.join(a.out, "fit_" + "".join(ch if ch.isalnum() else "_" for ch in n))
            ek.save_figure(ek.plot_fit(r, res.spectra[n], res.kk[n]), p, ("png",), dpi=150)
            figs[f"Fit of {n}."] = p + ".png"
        try:
            res.to_docx(os.path.join(a.out, "EIS_tables.docx"), figs)
        except ImportError:
            print("python-docx not installed: Word report skipped (pip install python-docx)")
        ek.save_figure(ek.plot_nyquist(res), os.path.join(a.out, "Nyquist"))
        ek.save_figure(ek.plot_bode(res), os.path.join(a.out, "Bode"))
    print("results in", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
