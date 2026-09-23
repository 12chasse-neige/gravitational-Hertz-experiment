"""Configured GWINC thermal/gas estimates and stable comparison exports."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import yaml

from ghe.config import ConfigFileError
from ghe.paths import CONFIG_DIR, DETECTOR_CONFIG_FILE, SOURCE_CONFIG_FILE
from ghe.thermal_gas import SPECTRUM_COLUMNS, THERMAL_COMPONENTS, calculate_thermal_gas
from scripts.estimateThermalGas import main, write_report


@pytest.fixture
def source_path(tmp_path):
    source = yaml.safe_load(SOURCE_CONFIG_FILE.read_text())
    source['Noise']['Estimation']['FrequencySamples'] = 12
    path = tmp_path / 'source.yaml'
    path.write_text(yaml.safe_dump(source))
    return path


def test_default_hybrid_is_ce2_except_signal_recycling():
    hybrid = yaml.safe_load(DETECTOR_CONFIG_FILE.read_text())
    ce2 = yaml.safe_load((CONFIG_DIR / 'CE2silicon.yaml').read_text())
    assert hybrid['+inherit'] == ce2['+inherit'] == 'CE2silicon'
    assert hybrid['Squeezer'] == ce2['Squeezer']
    assert hybrid['Infrastructure']['Length'] == 40000
    assert float(hybrid['Laser']['Wavelength']) == 2e-6
    assert hybrid['Materials']['Substrate']['Temp'] == 123
    assert hybrid['Suspension']['Stage'][0]['Temp'] == 123
    assert float(hybrid['Optics']['SRM']['Transmittance']) == 1e-5
    assert hybrid['Optics']['SRM']['Tunephase'] is None
    assert hybrid['Optics']['SRM']['CavityLength'] == 55
    normalized = deepcopy(hybrid)
    normalized['Optics']['SRM'] = ce2['Optics']['SRM']
    # GWINC's YAML auto-compute sentinels are NaNs, which do not compare equal.
    assert yaml.safe_dump(normalized, sort_keys=True) == yaml.safe_dump(ce2, sort_keys=True)


def test_three_configurations_share_schema_and_grid(source_path):
    paths = [DETECTOR_CONFIG_FILE, CONFIG_DIR / 'aLIGO.yaml', CONFIG_DIR / 'CE2silicon.yaml']
    hybrid, aligo, ce2 = [calculate_thermal_gas(detector_config=path, source_config=source_path)
                          for path in paths]
    for result, family in zip((hybrid, aligo, ce2), ('CE2silicon', 'aLIGO', 'CE2silicon')):
        assert result.metadata['gwinc_budget_class'] == family
        assert tuple(result.spectra_psd) == SPECTRUM_COLUMNS
        assert result.target_frequency_hz == 600
        assert len(result.frequency_hz) == 13
        np.testing.assert_array_equal(result.frequency_hz, hybrid.frequency_hz)
        psd = result.spectra_psd
        np.testing.assert_allclose(psd['ThermalTotal'], sum(psd[name] for name in THERMAL_COMPONENTS), rtol=1e-14, atol=0)
        np.testing.assert_allclose(psd['GasScattering'], sum(psd['Scattering'+species] for species in ('H2', 'N2', 'H2O', 'O2')), rtol=1e-14, atol=0)
        np.testing.assert_allclose(psd['GasDamping'], sum(psd['Damping'+species] for species in ('H2', 'N2', 'H2O', 'O2')), rtol=1e-14, atol=0)
        np.testing.assert_allclose(psd['ResidualGas'], psd['GasScattering']+psd['GasDamping'], rtol=1e-14, atol=0)
        np.testing.assert_allclose(psd['ThermalPlusGas'], psd['ThermalTotal']+psd['ResidualGas'], rtol=1e-14, atol=0)
        assert not {'Quantum', 'Seismic', 'Newtonian'} & set(psd)
    assert np.all(aligo.spectra_psd['ITMThermoRefractive'] == 0)
    assert np.all(hybrid.spectra_psd['ITMThermoRefractive'] > 0)
    np.testing.assert_array_equal(hybrid.spectra_psd['ThermalTotal'], ce2.spectra_psd['ThermalTotal'])
    np.testing.assert_array_equal(hybrid.spectra_psd['ResidualGas'], ce2.spectra_psd['ResidualGas'])

    target = hybrid.summary()
    assert target['target_noise']['ThermalTotal']['asd'] == pytest.approx(1.2358717102739732e-26, rel=1e-7)
    assert target['target_noise']['ResidualGas']['asd'] == pytest.approx(2.3322146137678283e-26, rel=1e-7)
    assert target['notebook_gas_constraints']['optical_pathlength']['pressure_ceiling_pa'] == pytest.approx(1.984302553819681e-9, rel=1e-6)
    assert target['metadata']['source_distance_m'] == 60000
    assert target['metadata']['effective_detector']['Squeezer']['AmplitudedB'] == 15
    assert target['metadata']['effective_detector']['Optics']['SRM']['Tunephase'] == pytest.approx(1.5638384282313846)


def test_source_detuning_uses_rotor_frequency_not_reporting_target(source_path):
    source = yaml.safe_load(source_path.read_text())
    source['Source']['Rotor']['AngularVelocity'] *= 1.1
    source['Noise']['Estimation']['TargetFrequency'] = 600
    source_path.write_text(yaml.safe_dump(source))
    changed = calculate_thermal_gas(source_config=source_path)
    assert changed.target_frequency_hz == 600
    assert changed.metadata['source_frequency_hz'] == pytest.approx(660)
    assert changed.metadata['effective_detector']['Optics']['SRM']['Tunephase'] != pytest.approx(1.5638384282313846)


def test_pressure_and_temperature_edits_change_expected_terms(source_path, tmp_path):
    detector = yaml.safe_load(DETECTOR_CONFIG_FILE.read_text())
    path = tmp_path / 'detector.yaml'
    path.write_text(yaml.safe_dump(detector))
    baseline = calculate_thermal_gas(detector_config=path, source_config=source_path)
    for species in ('H2', 'N2', 'H2O', 'O2'):
        detector['Infrastructure']['ResidualGas'][species]['BeamtubePressure'] *= 2
    path.write_text(yaml.safe_dump(detector))
    pressure = calculate_thermal_gas(detector_config=path, source_config=source_path)
    np.testing.assert_allclose(pressure.spectra_psd['GasScattering'], 2*baseline.spectra_psd['GasScattering'], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(pressure.spectra_psd['GasDamping'], baseline.spectra_psd['GasDamping'])
    np.testing.assert_array_equal(pressure.spectra_psd['ThermalTotal'], baseline.spectra_psd['ThermalTotal'])

    detector['Materials']['Substrate']['Temp'] = 150
    path.write_text(yaml.safe_dump(detector))
    substrate = calculate_thermal_gas(detector_config=path, source_config=source_path)
    assert not np.array_equal(substrate.spectra_psd['CoatingBrownian'], pressure.spectra_psd['CoatingBrownian'])
    np.testing.assert_array_equal(substrate.spectra_psd['ResidualGas'], pressure.spectra_psd['ResidualGas'])

    detector['Infrastructure']['Temp'] = 270
    path.write_text(yaml.safe_dump(detector))
    tube = calculate_thermal_gas(detector_config=path, source_config=source_path)
    assert not np.array_equal(tube.spectra_psd['GasScattering'], substrate.spectra_psd['GasScattering'])
    np.testing.assert_array_equal(tube.spectra_psd['GasDamping'], substrate.spectra_psd['GasDamping'])
    np.testing.assert_array_equal(tube.spectra_psd['ThermalTotal'], substrate.spectra_psd['ThermalTotal'])

    for species in ('H2', 'N2', 'H2O', 'O2'):
        detector['Infrastructure']['ResidualGas'][species]['ChamberPressure'] = (
            2 * float(detector['Infrastructure']['ResidualGas'][species]['ChamberPressure'])
        )
    path.write_text(yaml.safe_dump(detector))
    chamber = calculate_thermal_gas(detector_config=path, source_config=source_path)
    np.testing.assert_allclose(chamber.spectra_psd['GasDamping'],
                               2*tube.spectra_psd['GasDamping'], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(chamber.spectra_psd['GasScattering'], tube.spectra_psd['GasScattering'])


def test_cli_exports_csv_json_and_plot_with_exact_target(source_path, tmp_path):
    output = tmp_path / 'report'
    main(['--config', 'configs/detector.yaml', '--source-config', str(source_path),
          '--output-dir', str(output)])
    summary = json.loads((output / 'thermal_gas.json').read_text())
    assert (output / 'thermal_gas.png').is_file()
    with (output / 'thermal_gas.csv').open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['frequency_hz'] + [f'{name}_{unit}' for name in SPECTRUM_COLUMNS for unit in ('psd', 'asd')]
        rows = list(reader)
    assert len(rows) == summary['frequency_samples']
    target_row = next(row for row in rows if float(row['frequency_hz']) == 600)
    for name in SPECTRUM_COLUMNS:
        assert float(target_row[name+'_psd']) == summary['target_noise'][name]['psd']
    assert summary['metadata']['effective_detector']['Suspension']['Stage'][0]['K'] == 36300

    result = calculate_thermal_gas(source_config=source_path)
    write_report(result, output, make_plot=False)
    assert not (output / 'thermal_gas.png').exists()


@pytest.mark.parametrize('field,value', [
    ('FrequencySamples', 1), ('FrequencySamples', 3.5),
    ('TargetFrequency', -1), ('GasAllowanceASD', 0),
])
def test_invalid_estimation_settings(source_path, field, value):
    source = yaml.safe_load(source_path.read_text())
    source['Noise']['Estimation'][field] = value
    source_path.write_text(yaml.safe_dump(source))
    with pytest.raises(ConfigFileError, match=field):
        calculate_thermal_gas(source_config=source_path)


def test_unknown_budget_marker_is_rejected(source_path, tmp_path):
    detector = yaml.safe_load(DETECTOR_CONFIG_FILE.read_text())
    detector['+inherit'] = 'Voyager'
    path = tmp_path / 'detector.yaml'
    path.write_text(yaml.safe_dump(detector))
    with pytest.raises(ConfigFileError, match='Unsupported GWINC'):
        calculate_thermal_gas(detector_config=path, source_config=source_path)
