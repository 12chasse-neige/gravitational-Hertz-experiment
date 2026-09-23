"""GWINC thermal and residual-gas estimates from one complete detector YAML.

All exported strain spectra are one-sided power spectral densities (1/Hz).
Independent contributions are added as PSDs before taking square roots for
ASDs. The paper-equivalent gas model is reported separately and is never added
to the GWINC gas or thermal-plus-gas totals.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigFileError, DetectorConfig, _detector_defaults_from_yaml, _read_yaml
from .paths import CONFIG_DIR, DETECTOR_CONFIG_FILE, REPO_ROOT, SOURCE_CONFIG_FILE
from .residual_gas import (
    SPECIES, at, calculate_residual_gas_constraints,
    calculate_residual_gas_spectra, number, physical,
)


# The same columns are exported for both detector families. aLIGO has no
# ITM thermo-refractive budget term, so that one column is exactly zero there.
THERMAL_COMPONENTS = (
    'SuspensionThermal', 'CoatingBrownian', 'CoatingThermoOptic',
    'ITMThermoRefractive', 'SubstrateBrownian', 'SubstrateThermoElastic',
)
GAS_COMPONENTS = tuple(
    mechanism + species for mechanism in ('Scattering', 'Damping') for species in SPECIES
)
SPECTRUM_COLUMNS = (
    *THERMAL_COMPONENTS, 'ThermalTotal', *GAS_COMPONENTS,
    'GasScattering', 'GasDamping', 'ResidualGas', 'ThermalPlusGas',
    'NotebookScattering', 'NotebookDampingFreeMass',
    'NotebookDampingInfiniteFreeMass',
)


def resolve_config_path(path: str | Path) -> Path:
    """Accept absolute paths, repository-relative paths, or names in configs/."""
    requested = Path(path).expanduser()
    candidates = ([requested] if requested.is_absolute() else
                  [requested, REPO_ROOT / requested, CONFIG_DIR / requested])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ConfigFileError(f'Configuration file not found: {path}')


def _json_config(value):
    """Encode GWINC's .nan auto-compute sentinels in strict JSON metadata."""
    if isinstance(value, dict):
        return {key: _json_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_config(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return '.nan' if np.isnan(value) else ('.inf' if value > 0 else '-.inf')
    return value


def _trace_psd(trace, *path) -> np.ndarray:
    value = trace
    for key in path:
        value = value[key]
    return np.asarray(value.psd, dtype=float)


@dataclass
class ThermalGasResult:
    frequency_hz: np.ndarray
    spectra_psd: dict[str, np.ndarray]
    target_frequency_hz: float
    gas_constraints: dict
    metadata: dict

    def summary(self) -> dict:
        """Summarize the exact target row; the complete curves go to CSV."""
        target_index = int(np.flatnonzero(self.frequency_hz == self.target_frequency_hz)[0])
        target = {
            name: {'psd': float(values[target_index]),
                   'asd': float(np.sqrt(values[target_index]))}
            for name, values in self.spectra_psd.items()
        }
        allowance = self.gas_constraints['allowance_asd']
        optical = self.gas_constraints['optical_pathlength']
        return {
            'units': {'frequency': 'Hz', 'psd': '1/Hz', 'asd': '1/sqrt(Hz)',
                      'pressure': 'Pa', 'temperature': 'K', 'number_density': '1/m^3'},
            'target_frequency_hz': self.target_frequency_hz,
            'frequency_samples': len(self.frequency_hz),
            'target_noise': target,
            'reference_allowance': {
                'gas_asd': allowance,
                'thermal_to_allowance_ratio': target['ThermalTotal']['asd'] / allowance,
                'gwinc_gas_to_allowance_ratio': target['ResidualGas']['asd'] / allowance,
                'nominal_beamtube_to_conditional_ceiling_ratio': (
                    optical['pressure_pa'] / optical['pressure_ceiling_pa']
                    if optical['pressure_ceiling_pa'] is not None else None
                ),
                'interpretation': 'Paper allowance is a conditional reference, not a CE2 design limit.',
            },
            'notebook_gas_constraints': self.gas_constraints,
            'metadata': self.metadata,
        }


def calculate_thermal_gas(*, detector_config=DETECTOR_CONFIG_FILE,
                          source_config=SOURCE_CONFIG_FILE) -> ThermalGasResult:
    """Read YAML afresh and evaluate only GWINC thermal and gas budget terms.

    ``+inherit`` selects the GWINC budget family. In GWINC, an omitted marker
    means aLIGO. We instantiate the selected class with the full effective YAML
    rather than relying on ``gwinc.load_budget`` merging: its ``Struct.update``
    skips nulls, which would erase GHE's automatic SR detuning request.
    """
    import gwinc
    from gwinc import Struct
    from gwinc.ifo.CE2silicon import CE2silicon
    from gwinc.ifo.aLIGO import aLIGO

    detector_path = resolve_config_path(detector_config)
    source_path = resolve_config_path(source_config)
    detector = deepcopy(dict(_read_yaml(detector_path)))
    source = deepcopy(dict(_read_yaml(source_path)))
    budget_name = detector.pop('+inherit', 'aLIGO')
    if budget_name not in ('CE2silicon', 'aLIGO'):
        raise ConfigFileError(f'Unsupported GWINC +inherit budget {budget_name!r}; use CE2silicon or aLIGO')

    source_frequency = physical(source, 'Source', 'Rotor', 'AngularVelocity') / np.pi
    auto_phase_requested = at(detector, 'Optics', 'SRM', 'Tunephase') is None
    detector_fields = _detector_defaults_from_yaml(detector, detector_path)
    for key in ('testmass', 'length', 'length_SR', 'wavelength', 'power'):
        number(detector_fields[key], key)
    detector_fields.update(
        c=physical(source, 'Constants', 'SpeedOfLight'),
        hbar=physical(source, 'Constants', 'ReducedPlanckConstant'),
        resonance_frequency_hz=source_frequency,
    )
    resolved = DetectorConfig(**detector_fields)
    if not np.isfinite(resolved.phi_SR):
        raise ConfigFileError('Optics.SRM.Tunephase must resolve to a finite phase')
    detector['Optics']['SRM']['Tunephase'] = float(resolved.phi_SR)

    settings = at(source, 'Noise', 'Estimation')
    samples = number(at(settings, 'FrequencySamples'), 'Noise.Estimation.FrequencySamples')
    if samples != int(samples) or samples < 2:
        raise ConfigFileError('Noise.Estimation.FrequencySamples must be an integer >= 2')
    lower = physical(source, 'Noise', 'FrequencyBand', 'Minimum')
    upper = physical(source, 'Noise', 'FrequencyBand', 'Maximum')
    if upper <= lower:
        raise ConfigFileError('Noise.FrequencyBand.Maximum must exceed Minimum')
    requested_target = at(settings, 'TargetFrequency')
    target = (source_frequency if requested_target is None else
              number(requested_target, 'Noise.Estimation.TargetFrequency'))
    allowance = number(at(settings, 'GasAllowanceASD'), 'Noise.Estimation.GasAllowanceASD')
    # The common grid is independent of the selected detector and includes the
    # exact comparison frequency even if it lies outside the plotting band.
    frequency = np.unique(np.append(np.geomspace(lower, upper, int(samples)), target))

    gas = calculate_residual_gas_spectra(frequency, detector)
    constraints = calculate_residual_gas_constraints(detector, target, allowance)
    if budget_name == 'CE2silicon':
        budget_type = CE2silicon
        selected = ('SuspensionThermal', 'Coating', 'Substrate', 'ResidualGas')
        thermal_paths = {
            'SuspensionThermal': ('SuspensionThermal',),
            'CoatingBrownian': ('Coating', 'CoatingBrownian'),
            'CoatingThermoOptic': ('Coating', 'CoatingThermoOptic'),
            'ITMThermoRefractive': ('Substrate', 'ITMThermoRefractive'),
            'SubstrateBrownian': ('Substrate', 'SubstrateBrownian'),
            'SubstrateThermoElastic': ('Substrate', 'SubstrateThermoElastic'),
        }
    else:
        budget_type = aLIGO
        selected = (*THERMAL_COMPONENTS[:3], *THERMAL_COMPONENTS[4:], 'ResidualGas')
        thermal_paths = {name: (name,) for name in selected if name != 'ResidualGas'}

    # A fresh GWINC budget uses the selected family's calibrations and only the
    # requested terms. This excludes quantum, seismic, and Newtonian noise.
    budget = budget_type(freq=frequency, noises=selected, ifo=Struct(deepcopy(detector)))
    try:
        trace = budget.run()
    except (AttributeError, KeyError) as exc:
        raise ConfigFileError(f'Incomplete GWINC detector configuration {detector_path}: {exc}') from exc
    spectra = {
        name: (_trace_psd(trace, *thermal_paths[name]) if name in thermal_paths
               else np.zeros_like(frequency))
        for name in THERMAL_COMPONENTS
    }
    spectra['ThermalTotal'] = sum(spectra[name] for name in THERMAL_COMPONENTS)
    for mechanism in ('Scattering', 'Damping'):
        for species in SPECIES:
            key = mechanism + species
            spectra[key] = _trace_psd(trace, 'ResidualGas', key)
        spectra['Gas' + mechanism] = sum(spectra[mechanism + species] for species in SPECIES)
    spectra['ResidualGas'] = spectra['GasScattering'] + spectra['GasDamping']
    if not np.allclose(spectra['ResidualGas'], _trace_psd(trace, 'ResidualGas'),
                       rtol=1e-12, atol=0):
        raise ValueError('GWINC residual-gas children do not sum to their parent PSD')
    if budget_name == 'CE2silicon':
        for parent, children in (
            ('Coating', ('CoatingBrownian', 'CoatingThermoOptic')),
            ('Substrate', ('ITMThermoRefractive', 'SubstrateBrownian', 'SubstrateThermoElastic')),
        ):
            if not np.allclose(_trace_psd(trace, parent),
                               sum(spectra[name] for name in children), rtol=1e-12, atol=0):
                raise ValueError(f'GWINC {parent} children do not sum to their parent PSD')
    spectra['ThermalPlusGas'] = spectra['ThermalTotal'] + spectra['ResidualGas']
    spectra['NotebookScattering'] = sum(gas.scattering_psd.values())
    spectra['NotebookDampingFreeMass'] = sum(gas.damping_free_mass_psd.values())
    spectra['NotebookDampingInfiniteFreeMass'] = sum(gas.damping_infinite_psd.values())
    spectra = {name: spectra[name] for name in SPECTRUM_COLUMNS}
    for name, values in spectra.items():
        if values.shape != frequency.shape or not np.all(np.isfinite(values) & (values >= 0)):
            raise ValueError(f'{name} returned an invalid PSD; check the detector configuration')

    stages = at(detector, 'Suspension', 'Stage')
    stage_temperatures = [
        number(stage['Temp'], f'Suspension.Stage.{index}.Temp') if 'Temp' in stage
        else physical(detector, 'Suspension', 'Temp')
        for index, stage in enumerate(stages)
    ]
    geometry_mass = (np.pi * physical(detector, 'Materials', 'MassRadius')**2
                     * physical(detector, 'Materials', 'MassThickness')
                     * physical(detector, 'Materials', 'Substrate', 'MassDensity'))
    warnings = []
    if not np.isclose(geometry_mass, resolved.testmass, rtol=1e-3):
        warnings.append(
            f'Mirror geometry/density imply {geometry_mass:.6g} kg; '
            f'Suspension.Stage[0].Mass is {resolved.testmass:.6g} kg.'
        )
    custom_ce2_sr = (budget_name == 'CE2silicon' and (
        auto_phase_requested
        or physical(detector, 'Optics', 'SRM', 'Transmittance') != 0.02
        or physical(detector, 'Optics', 'SRM', 'CavityLength') != 20
    ))
    if custom_ce2_sr:
        warnings.append('Custom narrowband SRM: thermal and gas estimates are CE2-based, but this is not the official CE2 sensitivity.')
    metadata = {
        'detector_config_path': str(detector_path),
        'source_config_path': str(source_path),
        'gwinc_version': gwinc.__version__,
        'gwinc_budget_class': budget_name,
        'computed_gwinc_terms': list(selected),
        'available_thermal_terms': list(thermal_paths),
        'effective_detector': {'+inherit': budget_name, **_json_config(detector)},
        'custom_ce2_signal_recycling': custom_ce2_sr,
        'source_frequency_hz': source_frequency,
        'source_distance_m': (resolved.length *
                              physical(source, 'Source', 'Placement', 'DetectorDistanceArmLengths')),
        'temperatures_k': {**gas.temperatures_k,
                           'substrate': physical(detector, 'Materials', 'Substrate', 'Temp'),
                           'suspension_stages': stage_temperatures},
        'pressures_pa': gas.pressures_pa,
        'pressure_totals_pa': {location: sum(values.values())
                               for location, values in gas.pressures_pa.items()},
        'geometry_mass_kg': float(geometry_mass),
        'warnings': warnings,
        'assumptions': [
            'Only GWINC thermal and residual-gas terms are included; this is not a detector sensitivity curve.',
            'Independent thermal terms and gas mechanisms are added as PSDs once.',
            'Notebook scattering and free-mass damping are comparisons, not added to GWINC gas.',
            'Uniform molecular partial pressures in each location; independent species.',
            'The gas pressure ceiling allocates the full paper allowance to each mechanism independently.',
            'Null SR tune phase is resolved at the source GW frequency, not the reporting target frequency.',
            'Squeezer settings remain those in the selected detector YAML; they do not enter this selected-term estimate.',
        ],
    }
    return ThermalGasResult(frequency, spectra, target, constraints, metadata)
