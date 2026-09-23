"""Estimate configured GWINC thermal and residual-gas noise for one detector."""
from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ghe.config import ConfigFileError
from ghe.paths import DETECTOR_CONFIG_FILE, RUNS_DIR, SOURCE_CONFIG_FILE
from ghe.thermal_gas import SPECTRUM_COLUMNS, THERMAL_COMPONENTS, ThermalGasResult, calculate_thermal_gas


def write_report(result: ThermalGasResult, output_dir: Path, *, make_plot: bool = True) -> dict:
    """Write matching CSV columns and a target-frequency JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = result.summary()
    (output_dir / 'thermal_gas.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    with (output_dir / 'thermal_gas.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frequency_hz'] +
                        [f'{name}_{unit}' for name in SPECTRUM_COLUMNS for unit in ('psd', 'asd')])
        for index, frequency in enumerate(result.frequency_hz):
            row = [frequency]
            for name in SPECTRUM_COLUMNS:
                psd = result.spectra_psd[name][index]
                row.extend((psd, np.sqrt(psd)))
            writer.writerow(row)

    plot_path = output_dir / 'thermal_gas.png'
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, layout='constrained')
        fig.suptitle(
            f"{Path(result.metadata['detector_config_path']).name} "
            f"({result.metadata['gwinc_budget_class']} thermal/gas budget)"
        )
        f = result.frequency_hz
        thermal = axes[0]
        for name in THERMAL_COMPONENTS:
            if np.any(result.spectra_psd[name]):
                thermal.loglog(f, np.sqrt(result.spectra_psd[name]), label=name, linewidth=1.1)
        thermal.loglog(f, np.sqrt(result.spectra_psd['ThermalTotal']),
                       label='Thermal total', color='black', linewidth=2.2)
        thermal.set_title('GWINC thermal components')

        gas = axes[1]
        for name, label, style, width in (
            ('GasScattering', 'GWINC gas scattering', '-', 1.2),
            ('GasDamping', 'GWINC gas damping', '-', 1.2),
            ('ResidualGas', 'GWINC residual gas total', '-', 2.1),
            ('ThermalPlusGas', 'Thermal + gas', '-', 2.2),
            ('NotebookScattering', 'Notebook optical comparison', '--', 1.2),
            ('NotebookDampingFreeMass', 'Notebook free-mass damping comparison', '--', 1.2),
        ):
            gas.loglog(f, np.sqrt(result.spectra_psd[name]), label=label,
                       linestyle=style, linewidth=width)
        gas.set_title('Residual gas and paper-model comparisons')
        gas.set_xlabel('Frequency [Hz]')
        for axis in axes:
            axis.axvline(result.target_frequency_hz, color='0.5', linestyle=':',
                         label=f'Target {result.target_frequency_hz:g} Hz')
            axis.axhline(result.gas_constraints['allowance_asd'], color='0.4', linestyle='-.',
                         label='Conditional paper allowance')
            axis.set_ylabel(r'Strain ASD [Hz$^{-1/2}$]')
            axis.grid(True, which='both', alpha=0.15)
            axis.legend(fontsize=7.5, ncol=2)
        fig.savefig(plot_path, dpi=170)
        plt.close(fig)
    else:
        plot_path.unlink(missing_ok=True)
    return summary


def print_summary(summary: dict) -> None:
    meta = summary['metadata']
    print(f"Detector config: {meta['detector_config_path']}")
    print(f"GWINC {meta['gwinc_version']} budget: {meta['gwinc_budget_class']}")
    print(f"Target: {summary['target_frequency_hz']:.10g} Hz")
    print(f"Arm length: {meta['effective_detector']['Infrastructure']['Length']:g} m; "
          f"source distance from R/L: {meta['source_distance_m']:g} m")
    print('Temperatures [K]: ' + ', '.join(
        f'{key}={value:g}' for key, value in meta['temperatures_k'].items()
        if key != 'suspension_stages'
    ))
    print('Suspension stages [K]: ' + ', '.join(
        f'{value:g}' for value in meta['temperatures_k']['suspension_stages']
    ))
    for location, pressures in meta['pressures_pa'].items():
        print(f"{location} total pressure: {meta['pressure_totals_pa'][location]:.8e} Pa "
              + '(' + ', '.join(f'{species}={value:.3e}' for species, value in pressures.items()) + ')')
    print('\nTarget strain ASD [Hz^-1/2]:')
    for key in ('ThermalTotal', 'GasScattering', 'GasDamping', 'ResidualGas',
                'ThermalPlusGas', 'NotebookScattering', 'NotebookDampingFreeMass'):
        print(f"  {key:30s} {summary['target_noise'][key]['asd']:.8e}")
    allowance = summary['reference_allowance']
    print(f"Paper reference allowance: {allowance['gas_asd']:.8e} Hz^-1/2; "
          f"thermal/allowance={allowance['thermal_to_allowance_ratio']:.3g}, "
          f"gas/allowance={allowance['gwinc_gas_to_allowance_ratio']:.3g}")
    optical = summary['notebook_gas_constraints']['optical_pathlength']
    if optical['pressure_ceiling_pa'] is None:
        print('\nConditional beam-tube pressure ceiling unavailable for zero-pressure mixture')
    else:
        print(f"\nConditional beam-tube pressure ceiling: {optical['pressure_ceiling_pa']:.8e} Pa "
              '(full paper gas allowance assigned to optical noise alone)')
        print('Nominal beam-tube pressure / conditional ceiling: '
              f"{allowance['nominal_beamtube_to_conditional_ceiling_ratio']:.3g}")
    for warning in meta['warnings']:
        print('WARNING: ' + warning)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DETECTOR_CONFIG_FILE,
                        help='Complete detector YAML; default is the GHE CE2 silicon hybrid.')
    parser.add_argument('--source-config', type=Path, default=SOURCE_CONFIG_FILE)
    parser.add_argument('--output-dir', type=Path, default=RUNS_DIR / 'thermal-gas')
    parser.add_argument('--no-plot', action='store_true', help='Write only CSV and JSON.')
    args = parser.parse_args(argv)
    try:
        result = calculate_thermal_gas(detector_config=args.config, source_config=args.source_config)
        summary = write_report(result, args.output_dir, make_plot=not args.no_plot)
    except (ConfigFileError, ValueError, ModuleNotFoundError) as exc:
        parser.exit(2, f'Thermal/gas estimate failed: {exc}\n')
    print_summary(summary)
    print(f'\nReport saved to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
