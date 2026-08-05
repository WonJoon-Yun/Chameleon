"""Chameleon-E column data for the two-track main table, from chameleon_e_grid.json.

For each (spec, field) cell returns the Chameleon-E gain formatted like the table's Gain column
(percent, mean [min,max]), or a marker: "same" when the guardrail never switched (Chameleon-E ==
Chameleon by construction), or "TBD" when the cell is E-infeasible (low-LER, search can't resolve).
gen_main_table_v2 imports ecell() to add the column.

Correctness note (verified 2026-07-14, 80 feasible seeds): the 'gain' branch averages evo_gain
across feasible seeds. This is honest because the grid writer sets evo_gain = base/LER(DEPLOYED
frame) on the held-out TEST split, where DEPLOYED = evolutionary frame iff it beat the surrogate on
the independent VAL split by >=MARGIN, else Chameleon's frame (so evo_gain == cham_gain exactly on
non-switched seeds). Guardrail decides on val, reports on test; 0 test-cell regressions measured.

Standalone: python3 scripts/e_column.py   (prints the column for every main-table cell)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

import json, os
import numpy as np
from chameleon._root import rpath
from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()

_J = rpath(_CFG.out_dir + "/chameleon_e_grid.json")


def _load():
    return json.load(open(_J)) if os.path.exists(_J) else {}


def ecell(grid, spec, field):
    """Return (text, kind) for one cell. kind in {'gain','same','infeasible','missing'}."""
    seeds = [v for k, v in grid.items() if k.startswith("%s|%s|" % (spec, field))]
    if not seeds:
        return ("", "missing")
    feas = [v for v in seeds if v.get("feasible")]
    if not feas:
        return ("TBD", "infeasible")                  # decoder search not yet resolved (low LER)
    if not any(v.get("deployed_evo") for v in feas):
        return ("same", "same")                       # E == Chameleon (guardrail kept the blind frame)
    g = [v["evo_gain"] for v in feas]
    pg, plo, phi = round((np.mean(g) - 1) * 100), round((min(g) - 1) * 100), round((max(g) - 1) * 100)
    return ("%+d\\%%\\,[%+d,%+d]" % (pg, plo, phi), "gain")


def main():
    grid = _load()
    ORDER = ["surf2d:3", "surf2d:5", "surf2d:7", "color2d:3", "color2d:5", "color2d:7",
             "BB18", "BB36", "BB72"]
    FIELDS = ["berlin_star", "miami_star", "willow_star", "xyz:10"]
    print("%-11s %-12s  %-22s %s" % ("code", "field", "Chameleon-E column", "kind"))
    for spec in ORDER:
        for field in FIELDS:
            txt, kind = ecell(grid, spec, field)
            print("%-11s %-12s  %-22s %s" % (spec, field, txt or "(none)", kind))


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
