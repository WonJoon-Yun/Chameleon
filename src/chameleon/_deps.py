"""Turn a missing measurement dependency into an instruction.

The artifact has two dependency tiers on purpose: reproducing every reported
result needs numpy and pandas, while re-measuring from scratch needs the
simulator and decoder stack. The README tells an evaluator to install the
minimal tier first, so the likely next step -- running a study script -- lands
on a bare `ModuleNotFoundError: No module named 'panqec'` and a traceback
through code they have never read.

`required` catches exactly that and says which tier is missing, what it is for,
and the one command that fixes it. Any other import error propagates untouched:
a genuinely broken install must not be disguised as a missing package.
"""
import contextlib

#: Packages that belong to the measurement tier (requirements.txt), keyed to
#: what each one provides, so the message can say why it is wanted.
_MEASUREMENT_STACK = {
    "stim": "stabilizer circuit simulation",
    "pymatching": "MWPM decoding (surface codes)",
    "ldpc": "BP+OSD decoding (bivariate-bicycle codes)",
    "chromobius": "color-code decoding",
    "panqec": "code constructions",
    "qldpc": "bivariate-bicycle code constructions",
}


@contextlib.contextmanager
def required(*packages, purpose=None):
    """Wrap imports of the measurement stack so a missing one explains itself.

        with required("panqec", purpose="code constructions"):
            from panqec.codes import RotatedPlanar2DCode

    ``packages`` names what the block imports; the raised message reports
    whichever one was actually missing. A ModuleNotFoundError for a package
    outside the measurement tier is re-raised unchanged -- it is a real bug, not
    a tier mismatch.
    """
    try:
        yield
    except ModuleNotFoundError as exc:
        name = (exc.name or "").split(".")[0]
        if name not in _MEASUREMENT_STACK:
            raise
        what = purpose or _MEASUREMENT_STACK[name]
        raise ModuleNotFoundError(
            "%s is part of the measurement stack (needed here for %s) and is not "
            "installed.\n"
            "\n"
            "    pip install -r requirements.txt        # or requirements-lock.txt "
            "for the exact versions behind the paper\n"
            "\n"
            "This is only needed to re-measure from scratch. Reproducing every "
            "reported result -- `make all` -- runs on numpy and pandas alone and "
            "does not import this package."
            % (name, what),
            name=exc.name, path=exc.path) from exc
