"""Command menu for detuned-interferometer design, rotor Qdot and array studies."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "quantum": ("quantumNoise", "Compare analytic and GWINC detuned quantum noise"),
    "noise-budget": ("plotNoiseBudget", "Plot quantum, thermal and residual-gas noise"),
    "thermal-gas": ("estimateThermalGas", "Report thermal and residual-gas constraints"),
    "qdot": ("extractQdot", "Project a rotor signal onto its Qdot-phase sine"),
    "qdot-sweep": ("sweepQdotArmLength", "Study modeled Qdot SNR versus arm length"),
    "array-generate": ("sourceArray", "Generate phase-aligned rotor arrays (CSV/NPZ)"),
    "array": ("runArray", "Generate or load an array and calculate total-signal SNR"),
    "array-sweep": ("sweepArraySNR", "Study total-signal SNR versus array size"),
    "validate": ("validateResponse", "Check response physics, Qdot projection and arrays"),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # Preserve existing main.py array commands. Empty invocations show the menu.
    if args and args[0].startswith("-") and args[0] not in ("-h", "--help"):
        return import_module("scripts.runArray").main(args)

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, description) in COMMANDS.items():
        subparsers.add_parser(name, help=description, add_help=False)
    selected, remaining = parser.parse_known_args(args)
    if selected.command is None:
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        parser.print_help()
        return
    module_name, _ = COMMANDS[selected.command]
    return import_module(f"scripts.{module_name}").main(remaining)


if __name__ == "__main__":
    main()
