"""
Equivalent circuits.

A circuit can be written in any notation understood by :mod:`eiskit.notation`
(native ``R0-p(R1,CPE1)``, ``Rs-p(Rct,CPEdl)``, ``R+Q/R``, ``(R)p(CPE-Ws)``, Boukamp
CDC ``R(QR)`` ...). It is converted to the canonical form ``-`` = series, ``p(A,B)`` =
parallel. :data:`LITERATURE` lists >60 named circuits from the EIS literature.
"""
from itertools import combinations_with_replacement

import numpy as np

from .elements import ELEMENTS, element
from .notation import describe, parse, to_string

class Circuit:
    """An equivalent circuit. ``Circuit("R0-p(R1,CPE1)")``"""

    def __init__(self, notation, description=""):
        self.notation = notation
        self.tree = parse(notation)
        self.string = to_string(self.tree)
        self.description = description
        self.names, self.units, kinds = [], [], []
        self.elements, self.start = [], {}
        self._index(self.tree, kinds)
        self.kinds = np.array(kinds)
        if len({n for _, n in self.elements}) != len(self.elements):
            raise ValueError(f"duplicate element names in '{notation}'")

    def __repr__(self):
        return f"Circuit('{self.string}')"

    def explain(self):
        """Plain-language reading of the circuit, e.g. 'R0 + [R1 ∥ CPE1]'."""
        return describe(self.string)

    @property
    def n_params(self):
        return len(self.names)

    def _index(self, node, kinds):
        if node[0] == "e":
            _, etype, name = node
            self.start[name] = len(self.names)
            self.elements.append((etype, name))
            for suf, unit, kind in ELEMENTS[etype]:
                self.names.append(name + suf)
                self.units.append(unit)
                kinds.append(kind)
        else:
            for c in node[1]:
                self._index(c, kinds)

    # ---- evaluation --------------------------------------------------------------------
    def _eval(self, node, p, w, k):
        if node[0] == "e":
            etype, name = node[1], node[2]
            s = self.start[name]
            m = len(ELEMENTS[etype])
            Z, d = element(etype, w, p[s:s + m])
            J = np.zeros((len(w), k), complex)
            J[:, s:s + m] = d
            return Z, J
        parts = [self._eval(c, p, w, k) for c in node[1]]
        if node[0] == "s":
            return sum(z for z, _ in parts), sum(j for _, j in parts)
        Y = sum(1 / z for z, _ in parts)
        Z = 1 / Y
        J = (Z ** 2)[:, None] * sum(j / (z ** 2)[:, None] for z, j in parts)
        return Z, J

    def impedance(self, p, w, jacobian=False):
        """Impedance at angular frequencies ``w``; with ``jacobian=True`` also dZ/dp."""
        with np.errstate(all="ignore"):
            Z, J = self._eval(self.tree, np.asarray(p, float), np.asarray(w, float), self.n_params)
        return (Z, J) if jacobian else Z

    def __call__(self, p, f):
        """Impedance at frequencies ``f`` in Hz."""
        return self.impedance(p, 2 * np.pi * np.asarray(f, float))

    # ---- helpers ---------------------------------------------------------------------
    def n_time_constants(self):
        """Number of dispersive elements (C, CPE, Warburg, Gerischer) — used for DRT pre-filtering."""
        return sum(t in ("C", "CPE", "W", "Wo", "Ws", "G", "TLM", "TLMQ") for t, _ in self.elements)

    def series_blocks(self):
        """Top-level R||C or R||CPE blocks: list of (cap_type, R_name, cap_name)."""
        top = self.tree[1] if self.tree[0] == "s" else [self.tree]
        out = []
        for node in top:
            if node[0] == "p" and len(node[1]) == 2 and all(b[0] == "e" for b in node[1]):
                kinds = sorted(b[1] for b in node[1])
                if kinds in (["CPE", "R"], ["C", "R"]):
                    R = next(b[2] for b in node[1] if b[1] == "R")
                    X = next(b[2] for b in node[1] if b[1] != "R")
                    out.append((kinds[0], R, X))
        return out


# ---------------------------------------------------------------------------------------
# initial guesses
# ---------------------------------------------------------------------------------------
def initial_guess(circ, f, Z, reverse=False):
    """Data-driven starting values for every parameter of ``circ``."""
    w = 2 * np.pi * np.asarray(f)
    zmax = np.max(np.abs(Z))
    Rs = max(np.min(Z.real), 1e-4 * zmax)
    Rpol = max(Z.real[np.argmin(f)] - Rs, 0.05 * zmax)
    wpk = w[np.argmax(-Z.imag)]
    top = circ.tree[1] if circ.tree[0] == "s" else [circ.tree]
    series_R = [n[2] for n in top if n[0] == "e" and n[1] == "R"]
    for nm in ("R0", "Rs"):
        if nm in circ.start and nm not in series_R:
            series_R.append(nm)
    n_other = sum(1 for t, n in circ.elements
                  if (t == "R" and n not in series_R) or t in ("Wo", "Ws", "G"))
    share = Rpol / max(n_other, 1)
    caps = [n for t, n in circ.elements if t in ("C", "CPE") and not n.endswith("9")]
    spread = np.linspace(1.5, -1.5, len(caps)) if len(caps) > 1 else np.zeros(1)
    if reverse:
        spread = spread[::-1]
    wcap = {n: wpk * 10 ** e for n, e in zip(caps, spread)}
    p = np.zeros(circ.n_params)
    for t, n in circ.elements:
        s = circ.start[n]
        if t in ("C", "CPE") and n.endswith("9"):                   # stray capacitance across the cell
            p[s] = 0.05 / (w.max() * max(Z.real.max(), 1e-6))
            if t == "CPE":
                p[s + 1] = 0.95
        elif t == "R":
            p[s] = Rs / max(len(series_R), 1) if n in series_R else share
        elif t == "C":
            p[s] = 1 / (wcap[n] * share)
        elif t == "CPE":
            p[s], p[s + 1] = 1 / (wcap[n] * share), 0.85
        elif t == "L":
            ind = Z.imag > 0
            p[s] = np.median(Z.imag[ind] / w[ind]) if ind.any() else 1e-7
        elif t == "Ls":
            k = max(3, len(f) // 15)
            p[s] = np.median(Z.imag[:k] / w[:k])
        elif t == "W":
            p[s] = max(abs(Z.imag[np.argmin(f)]) * np.sqrt(w.min()), 1e-3 * share)
        elif t in ("Wo", "Ws"):
            p[s], p[s + 1] = share, 1 / w.min()
        elif t == "G":
            p[s], p[s + 1] = share, 1 / wpk
        elif t == "TLM":
            p[s:s + 4] = share, share, 1 / (wpk * share), 0.85
        elif t == "TLMQ":
            p[s:s + 3] = share, 1 / (wpk * share), 0.85
    return p


# ---------------------------------------------------------------------------------------
# circuit libraries
# ---------------------------------------------------------------------------------------
# name, field of application, circuit, meaning of the elements
LITERATURE = [
    # --- blocking electrodes / capacitors ---------------------------------------------------
    ("Ideal blocking electrode", "capacitors, inert electrodes", "R0-C1", "R0 electrolyte; C1 double layer"),
    ("Blocking electrode (CPE)", "capacitors, rough/porous electrodes", "R0-CPE1", "CPE1 non-ideal double layer"),
    ("Blocking porous electrode (de Levie)", "supercapacitors, porous carbons", "R0-TLMQ1",
     "TLMQ1 pore ionic resistance + distributed double layer"),
    ("Contact + blocking porous electrode", "supercapacitors", "R0-p(R1,CPE1)-TLMQ2",
     "R1/CPE1 current-collector contact; TLMQ2 porous electrode"),
    # --- Randles family -----------------------------------------------------------------------
    ("Simplified Randles", "electrocatalysis, corrosion", "R0-p(R1,C1)", "R1 charge transfer; C1 double layer"),
    ("Simplified Randles (CPE)", "electrocatalysis (HER/OER/ORR), corrosion", "R0-p(R1,CPE1)",
     "R1 charge transfer; CPE1 double layer"),
    ("Randles (semi-infinite diffusion)", "redox couples, sensors", "R0-p(R1-W1,C1)", "W1 Warburg diffusion"),
    ("Modified Randles (CPE + Warburg)", "sensors, biosensors, batteries", "R0-p(R1-W1,CPE1)", ""),
    ("Randles, finite-length diffusion", "thin films, membranes, batteries", "R0-p(R1-Ws1,CPE1)",
     "Ws1 transmissive diffusion layer"),
    ("Randles, finite-space diffusion", "thin films, intercalation, batteries", "R0-p(R1-Wo1,CPE1)",
     "Wo1 reflective (blocked) diffusion"),
    ("Randles, finite-length diffusion (ideal C)", "thin films", "R0-p(R1-Ws1,C1)", ""),
    ("Randles, finite-space diffusion (ideal C)", "intercalation", "R0-p(R1-Wo1,C1)", ""),
    ("Randles with external Warburg", "batteries, fuel cells", "R0-p(R1,CPE1)-W2",
     "diffusion outside the interfacial arc"),
    ("Randles with external Warburg (ideal C)", "batteries", "R0-p(R1,C1)-W2", ""),
    ("Charge transfer + capacitive tail", "pseudocapacitors, intercalation", "R0-p(R1,CPE1)-CPE2",
     "CPE2 low-frequency (pseudo)capacitance"),
    ("Charge transfer + ideal capacitive tail", "pseudocapacitors", "R0-p(R1,CPE1)-C2", ""),
    ("Charge transfer + finite-length diffusion", "batteries, membranes", "R0-p(R1,CPE1)-Ws2", ""),
    ("Charge transfer + blocked diffusion", "batteries at low frequency", "R0-p(R1,CPE1)-Wo2", ""),
    ("Randles with Gerischer", "coupled reaction-diffusion", "R0-p(R1-G1,CPE1)", ""),
    # --- two time constants -------------------------------------------------------------------
    ("Two-time-constant (Voigt, ideal)", "films, SEI, coatings", "R0-p(R1,C1)-p(R2,C2)", ""),
    ("Two-time-constant (Voigt, CPE)", "electrocatalysts with surface layer, SEI + charge transfer",
     "R0-p(R1,CPE1)-p(R2,CPE2)", "R1/CPE1 film or surface layer; R2/CPE2 charge transfer + double layer"),
    ("Two-time-constant (ideal + CPE)", "films", "R0-p(R1,C1)-p(R2,CPE2)", ""),
    ("Two-time-constant (CPE + ideal)", "films", "R0-p(R1,CPE1)-p(R2,C2)", ""),
    ("SEI + charge transfer + diffusion", "Li-ion batteries", "R0-p(R1,CPE1)-p(R2-W2,CPE2)",
     "R1 SEI; R2 charge transfer; W2 solid-state diffusion"),
    ("SEI + charge transfer + finite-length diffusion", "Li-ion batteries", "R0-p(R1,CPE1)-p(R2-Ws2,CPE2)", ""),
    ("SEI + charge transfer + blocked diffusion", "Li-ion batteries", "R0-p(R1,CPE1)-p(R2-Wo2,CPE2)", ""),
    ("Battery cell (SEI, CT, Warburg tail)", "Li-ion / Na-ion cells", "R0-p(R1,CPE1)-p(R2,CPE2)-W3", ""),
    ("Battery cell (SEI, CT, blocked diffusion)", "Li-ion cells", "R0-p(R1,CPE1)-p(R2,CPE2)-Wo3", ""),
    ("Battery cell (SEI, CT, intercalation capacitance)", "Li-ion cells", "R0-p(R1,CPE1)-p(R2,CPE2)-CPE3", ""),
    ("Battery cell (SEI, CT, ideal intercalation capacitance)", "Li-ion cells", "R0-p(R1,CPE1)-p(R2,CPE2)-C3", ""),
    ("Two arcs + Gerischer", "SOFC electrodes", "R0-p(R1,CPE1)-p(R2-G2,CPE2)", ""),
    # --- three or more time constants ---------------------------------------------------------
    ("Three-time-constant (CPE)", "solid-state cells, SOFC, multilayers", "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)", ""),
    ("Brick-layer model", "ceramics, solid electrolytes (bulk, grain boundary, electrode)",
     "R0-p(R1,C1)-p(R2,C2)-p(R3,C3)", "R1 bulk; R2 grain boundary; R3 electrode"),
    ("Brick-layer model (CPE)", "ceramics, solid electrolytes", "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-CPE4",
     "CPE4 blocking electrode"),
    ("Three arcs + diffusion", "batteries, fuel cells", "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3-W3,CPE3)", ""),
    ("Four-time-constant (CPE)", "fuel cells, complex interfaces",
     "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-p(R4,CPE4)", ""),
    # --- nested / ladder ---------------------------------------------------------------------
    ("Coated metal / porous film (nested)", "coatings, passive films, corrosion", "R0-p(CPE1,R1-p(R2,CPE2))",
     "CPE1 coating; R1 pore resistance; R2/CPE2 metal interface"),
    ("Coated metal (ideal capacitances)", "coatings", "R0-p(C1,R1-p(R2,C2))", ""),
    ("Adsorbed intermediate (Armstrong-Henderson / Harrington-Conway)", "HER, OER, UPD",
     "R0-p(CPE1,R1-p(R2,C2))", "R1 charge transfer; R2/C2 adsorption resistance and pseudocapacitance"),
    ("Coating with diffusion at the metal", "coatings, corrosion", "R0-p(CPE1,R1-p(R2-W2,CPE2))", ""),
    ("Coating with blocked diffusion", "coatings", "R0-p(CPE1,R1-p(R2-Wo2,CPE2))", ""),
    ("Three-layer ladder", "multilayer coatings, porous films", "R0-p(CPE1,R1-p(CPE2,R2-p(R3,CPE3)))", ""),
    ("Film with blocking inner interface", "passive films", "R0-p(CPE1,R1-CPE2)", ""),
    ("Extra HF arc + nested adsorption model", "electrocatalysts on supports", "R0-p(R1,CPE1)-p(CPE2,R2-p(R3,CPE3))", ""),
    # --- parallel branches (Maxwell) ----------------------------------------------------------
    ("Maxwell-type adsorption branch", "adsorption, electrocatalysis", "R0-p(CPE1,R1,R2-C2)", ""),
    ("Maxwell-type adsorption branch (ideal)", "adsorption", "R0-p(C1,R1,R2-C2)", ""),
    ("Surface-state model (photoelectrodes)", "photoelectrochemistry (hematite, BiVO4)",
     "R0-p(C1,R1,R2-p(C2,R3))", "C1 bulk; R1 bulk transfer; R2 trapping; C2/R3 surface states"),
    ("Surface-state model (CPE)", "photoelectrochemistry", "R0-p(CPE1,R1,R2-p(CPE2,R3))", ""),
    # --- inductive behaviour -------------------------------------------------------------------
    ("Inductive loop (adsorbed intermediate)", "corrosion, dissolution, fuel cells", "R0-p(CPE1,R1,R2-L2)", ""),
    ("Inductive loop (nested)", "corrosion with adsorption", "R0-p(CPE1,R1-p(R2,L2))", ""),
    ("Randles with cable inductance", "high-frequency artefacts", "L0-R0-p(R1,CPE1)", ""),
    ("Two arcs with cable inductance", "high-frequency artefacts", "L0-R0-p(R1,CPE1)-p(R2,CPE2)", ""),
    ("Two arcs with signed HF inductance", "instrument artefact correction", "Ls0-R0-p(R1,CPE1)-p(R2,CPE2)",
     "Ls0 may be negative (not a cell property)"),
    # --- Gerischer, transmission lines, diffusion only ----------------------------------------
    ("Gerischer", "chemical reaction + diffusion, MIEC electrodes", "R0-G1", ""),
    ("SOFC MIEC cathode (Gerischer / ALS)", "solid oxide fuel cells", "R0-p(R1,CPE1)-G2", ""),
    ("Porous electrode with charge transfer (de Levie)", "porous electrodes, fuel-cell catalyst layers",
     "R0-TLM1", "TLM1 pore resistance, charge transfer and double layer distributed along the pores"),
    ("Contact + porous electrode", "porous electrodes", "R0-p(R1,CPE1)-TLM2", ""),
    ("Semi-infinite diffusion only", "diffusion-controlled electrodes", "R0-W1", ""),
    ("Finite-length diffusion only", "membranes, thin layers", "R0-Ws1", ""),
    ("Finite-space diffusion only", "thin films, batteries", "R0-Wo1", ""),
    # --- stray capacitance --------------------------------------------------------------------
    ("Randles with stray capacitance", "high-impedance cells, reference-electrode artefacts", "p(C9,R0-p(R1,CPE1))", ""),
    ("Two arcs with stray capacitance", "high-impedance cells", "p(C9,R0-p(R1,CPE1)-p(R2,CPE2))", ""),
    ("Bulk + interface (solid electrolyte, blocking)", "solid electrolytes", "p(R1,C1)-CPE2",
     "R1/C1 bulk; CPE2 blocking electrodes"),
    ("Bulk + grain boundary + blocking", "solid electrolytes", "p(R1,C1)-p(R2,CPE2)-CPE3", ""),
]


def literature_library():
    """``{canonical circuit: "name – field"}`` for the >60 named literature circuits."""
    return {Circuit(c).string: f"{name} – {field}" for name, field, c, _ in LITERATURE}


def literature_table():
    import pandas as pd
    return pd.DataFrame([{"name": n, "application": f, "circuit": Circuit(c).string, "elements": e}
                         for n, f, c, e in LITERATURE])


def _block(kind, i, cap):
    if kind == "G":
        return f"G{i}"
    return f"p(R{i},{cap}{i})" if kind == "plain" else f"p(R{i}-{kind}{i},{cap}{i})"


def circuit_library(level="literature", max_params=None, caps=("CPE", "C"), max_blocks=None):
    """
    Dictionary ``{circuit_string: description}``.

    level: ``"literature"`` (>60 named circuits, default), ``"extended"`` (~290 generated:
    1-3 R||Q blocks, one diffusion or Gerischer element, optional tail, nested/ladder
    circuits), ``"exhaustive"`` (~900: up to 4 blocks and stray capacitance), or
    ``"all"`` (literature + extended).
    """
    if level == "literature":
        lib = literature_library()
    elif level == "all":
        lib = {**circuit_library("extended"), **literature_library()}
    else:
        exh = level == "exhaustive"
        max_blocks = max_blocks or (4 if exh else 3)
        diff = ["W", "Wo", "Ws"]
        types = [("plain", c) for c in caps] + [("G", None)] + [(d, c) for d in diff for c in caps]
        tails = [None, "CPE", "W", "Wo", "Ws"] + (["C"] if "C" in caps else [])
        lib = {}
        for nb in range(1, max_blocks + 1):
            for combo in combinations_with_replacement(range(len(types)), nb):
                kinds = [types[j] for j in combo]
                nd = sum(k in diff for k, _ in kinds)
                ng = sum(k == "G" for k, _ in kinds)
                if nd > 1 or ng > 1:
                    continue
                for tail in tails:
                    if tail in diff and (nd or ng):
                        continue
                    parts = ["R0"] + [_block(k, i + 1, c) for i, (k, c) in enumerate(kinds)]
                    if tail:
                        parts.append(f"{tail}8")
                    desc = f"{nb} time constant(s)" + (" + diffusion" if nd else "") + \
                           (" + Gerischer" if ng else "") + (f" + {tail} tail" if tail else "")
                    lib["-".join(parts)] = desc
        for c1 in caps:
            for c2 in caps:
                for end in ["", "-W3", "-Wo3", "-Ws3"]:
                    inner = f"R1-p(R2{end},{c2}2)"
                    lib[f"R0-p({c1}1,{inner})"] = "nested: adsorption intermediate / porous layer"
                    lib[f"R0-p(R4,{c1}4)-p({c1}1,{inner})"] = "extra HF arc + nested model"
                    lib[f"R0-p({c1}1,R1-p({c2}2,R2-p(R3{end},{c2}3)))"] = "3-level ladder"
                lib[f"R0-p({c1}1,R1-{c2}2)"] = "film with blocking inner interface"
                lib[f"R0-p({c1}1,R1,R2-{c2}2)"] = "adsorption (Maxwell-type) branch"
        if exh:
            for k in list(lib):
                lib[f"p(C9,{k})"] = lib[k] + " + stray capacitance"
    if max_params:
        lib = {k: v for k, v in lib.items() if Circuit(k).n_params <= max_params}
    return lib
