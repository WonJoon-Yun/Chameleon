"""The `--help` contract shared by the study scripts.

Most scripts under `scripts/` take no arguments: everything they need comes from
`ProtocolConfig`, and the only per-run knob is `PROCS`. They still have to answer
`--help`, because that is the first thing anyone types at an unfamiliar script --
and a script that ignores argv answers it by launching a multi-hour measurement
instead. `parse_no_args` gives them a real parser in one line, so the contract
lives here rather than in ten near-identical copies.
"""
import argparse
import os
import sys


def parse_no_args(doc, epilog=None):
    """Handle `--help` for a script that takes no arguments, and reject anything else.

    Pass the module's ``__doc__``: it becomes the help text, so the description
    an evaluator reads is the same one that sits at the top of the file and
    cannot drift from it. Unknown arguments exit 2 with the usage message rather
    than being silently ignored, which is what makes a typo visible instead of
    costing a whole run.

    Returns the parsed (empty) namespace, so a caller can keep the usual shape.
    """
    doc = (doc or "").strip()
    head, _, body = doc.partition("\n")
    p = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description=head,
        epilog=body.strip() + (("\n\n" + epilog) if epilog else ""),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--procs", type=int, default=None,
                   help="worker processes (default: the PROCS environment variable, "
                        "else every core)")
    args = p.parse_args()
    if args.procs is not None:
        os.environ["PROCS"] = str(args.procs)
    return args
