"""
Circuit notation parser.

Several common notations are accepted and converted to one canonical form
(``-`` = series, ``p(A,B)`` = parallel), which is always shown back to the user so that
the interpretation can be checked.

Accepted syntax
---------------
Elements (numbers optional — they are assigned automatically):
    R  C  L  Ls  CPE (or Q)  W  Ws (or T)  Wo (or O)  G  TLM  TLMQ
Labels may be numbers (R1), common subscripts (Rs, Rct, Rsei, Rf, Cdl, CPEdl) or any text
after an underscore (R_pore, CPE_film).

Series:    ``A-B``   ``A+B``   ``AB`` (written next to each other)   ``[A B]``
Parallel:  ``p(A,B)``   ``A/B``   ``A|B``   ``A||B``   ``(A)p(B)``
Grouping:  ``( ... )``

Parallel binds more strongly than series (as in EC-Lab: ``R1+Q1/R2`` = R1 + (Q1 ∥ R2)).

Boukamp CDC (single letters, ``()`` = parallel, ``[]`` = series) is detected
automatically, e.g. ``R(QR)(QR)`` or ``R(Q[RW])``.

Examples
--------
    "R0-p(R1,CPE1)"          -> R0-p(R1,CPE1)
    "R-p(R,CPE)"             -> R0-p(R1,CPE1)
    "(R)p(CPE-Ws)"           -> p(R0,CPE1-Ws1)
    "R+Q/(R+W)"              -> R0-p(CPE1,R1-W1)
    "R(Q[RW])"  (CDC)        -> R0-p(CPE1,R1-W1)
"""
import re

ELEMENT_TYPES = ["TLMQ", "TLM", "CPE", "Ls", "Wo", "Ws", "W", "R", "C", "L", "G"]
ALIASES = {"Q": "CPE", "O": "Wo", "T": "Ws"}
# labels: digits, "_anything", or common subscripts (Rs, Rct, Cdl, CPEdl, Rsei, Rf ...)
_SUBSCRIPTS = "sei|film|pore|ion|ads|sol|ct|dl|ox|po|gb|ad|el|b|f|s"
_EL = re.compile(r"(TLMQ|TLM|CPE|Ls|Wo|Ws|W|R|C|L|G|Q|O|T)(\d+|_\w+|(?:" + _SUBSCRIPTS + r")(?![a-z]))?")
_CDC_CHARS = set("RCLQWOTG()[]")


class NotationError(ValueError):
    pass


# ---------------------------------------------------------------------------------------
def _tokenize(s):
    toks, i = [], 0
    s = s.replace(" ", "").replace("‖", "||").replace("∥", "||")
    while i < len(s):
        ch = s[i]
        if s.startswith("||", i):
            toks.append(("PAR", "||")); i += 2; continue
        if ch in "-+":
            toks.append(("SER", ch)); i += 1; continue
        if ch in "/|":
            toks.append(("PAR", ch)); i += 1; continue
        if ch in "([":
            toks.append(("LP", ch)); i += 1; continue
        if ch in ")]":
            toks.append(("RP", ch)); i += 1; continue
        if ch == ",":
            toks.append(("COMMA", ch)); i += 1; continue
        if ch == "p":
            prev = toks[-1][0] if toks else None
            if prev in ("EL", "RP"):
                toks.append(("PAR", "p"))                      # infix: (A)p(B)
            elif i + 1 < len(s) and s[i + 1] == "(":
                toks.append(("PFUN", "p("))                   # prefix: p(A,B)
                i += 1
            else:
                raise NotationError(f"unexpected 'p' at position {i} in '{s}'")
            i += 1
            continue
        m = _EL.match(s, i)
        if m:
            t = ALIASES.get(m.group(1), m.group(1))
            toks.append(("EL", (t, (m.group(2) or "").lstrip("_"))))
            i = m.end()
            continue
        raise NotationError(f"cannot read '{s[i:]}' (position {i}) in '{s}'")
    return toks


class _Parser:
    def __init__(self, toks, src):
        self.t, self.i, self.src = toks, 0, src

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, kind=None):
        tok = self.peek()
        if kind and tok[0] != kind:
            raise NotationError(f"expected {kind} but found '{tok[1]}' in '{self.src}'")
        self.i += 1
        return tok

    def series(self):
        items = [self.par()]
        while self.peek()[0] == "SER":
            self.take()
            items.append(self.par())
        return _mk("s", items)

    def par(self):
        items = [self.juxt()]
        while self.peek()[0] == "PAR":
            self.take()
            items.append(self.juxt())
        return _mk("p", items)

    def juxt(self):
        items = [self.atom()]
        while self.peek()[0] in ("EL", "LP", "PFUN"):
            items.append(self.atom())
        return _mk("s", items)

    def atom(self):
        kind, val = self.peek()
        if kind == "EL":
            self.take()
            return ("e", val[0], val[1])
        if kind == "LP":
            self.take()
            node = self.series()
            self.take("RP")
            return node
        if kind == "PFUN":
            self.take()
            branches = [self.series()]
            while self.peek()[0] == "COMMA":
                self.take()
                branches.append(self.series())
            self.take("RP")
            if len(branches) < 2:
                raise NotationError(
                    f"p(...) needs at least two branches separated by commas, e.g. p(R1,CPE1); "
                    f"for '{self.src}' did you mean (A)p(B)?")
            return _mk("p", branches)
        raise NotationError(f"unexpected '{val}' in '{self.src}'" if val else f"incomplete circuit '{self.src}'")


def _mk(kind, items):
    if len(items) == 1:
        return items[0]
    flat = []
    for it in items:                                  # flatten nested series/parallel of same kind
        flat.extend(it[1] if it[0] == kind else [it])
    return (kind, flat)


# ---------------------------------------------------------------------------------------
def _is_cdc(s):
    s = s.replace(" ", "")
    return (bool(s) and set(s) <= _CDC_CHARS and ("(" in s or "[" in s or len(s) <= 6)
            and not re.search(r"\d", s))


def _parse_cdc(s):
    s = s.replace(" ", "")
    pos = 0

    def group(closing, kind):
        nonlocal pos
        items = []
        while pos < len(s) and s[pos] != closing:
            ch = s[pos]
            if ch == "(":
                pos += 1; items.append(group(")", "p"))
            elif ch == "[":
                pos += 1; items.append(group("]", "s"))
            elif ch in "RCLQWOTG":
                items.append(("e", ALIASES.get(ch, ch), "")); pos += 1
            else:
                raise NotationError(f"unexpected '{ch}' in CDC '{s}'")
        if closing:
            if pos >= len(s):
                raise NotationError(f"missing '{closing}' in CDC '{s}'")
            pos += 1
        if not items:
            raise NotationError(f"empty group in CDC '{s}'")
        return _mk(kind, items)

    return group(None, "s")


# ---------------------------------------------------------------------------------------
def _number(tree):
    """Give every unnumbered element a unique number (R from 0, the others from 1)."""
    used = {}

    def collect(n):
        if n[0] == "e":
            if n[2]:
                key = n[1] + n[2]
                if key in used:
                    raise NotationError(f"element {key} appears twice")
                used[key] = True
        else:
            for c in n[1]:
                collect(c)

    collect(tree)
    counter = {}

    def assign(n):
        if n[0] == "e":
            t, num = n[1], n[2]
            if not num:
                k = counter.get(t, 0 if t == "R" else 1)
                while f"{t}{k}" in used:
                    k += 1
                counter[t] = k + 1
                num = str(k)
                used[t + num] = True
            return ("e", t, t + num)
        return (n[0], [assign(c) for c in n[1]])

    return assign(tree)


def to_string(tree):
    if tree[0] == "e":
        return tree[2]
    if tree[0] == "s":
        return "-".join(to_string(c) if c[0] != "s" else f"({to_string(c)})" for c in tree[1])
    return "p(" + ",".join(to_string(c) for c in tree[1]) + ")"


def parse(notation):
    """Parse any supported notation. Returns the element tree ``("e", type, name)`` /
    ``("s", [..])`` / ``("p", [..])`` with every element numbered."""
    s = notation.strip()
    if not s:
        raise NotationError("empty circuit")
    tree = _parse_cdc(s) if _is_cdc(s) else _Parser(_tokenize(s), s).series()
    return _number(tree)


def canonical(notation):
    """Canonical string of a circuit written in any supported notation."""
    return to_string(parse(notation))


def describe(notation):
    """Plain-language description of the parsed circuit (for checking the interpretation)."""
    def d(n, inside_par=False):
        if n[0] == "e":
            return n[2]
        if n[0] == "s":
            txt = " + ".join(d(c) for c in n[1])
            return f"({txt})" if inside_par else txt
        return "[" + " ∥ ".join(d(c, True) for c in n[1]) + "]"
    return d(parse(notation))
