"""Independent paper benchmarks and physical invariants for gas estimates."""
from copy import deepcopy

import numpy as np
import pytest
import yaml
from scipy.constants import Boltzmann as KB

from ghe.config import ConfigFileError
from ghe.paths import CONFIG_DIR
from ghe.residual_gas import (SPECIES, calculate_residual_gas_constraints,
                              calculate_residual_gas_spectra)


@pytest.fixture
def paper_detector():
    # Preserve the historical 4 km aLIGO-based notebook benchmark after the
    # default detector switches to the 40 km CE2 silicon hybrid.
    d = yaml.safe_load((CONFIG_DIR / 'aLIGO.yaml').read_text())
    d['Infrastructure'].update(Temp=.003838, Length=4000.)
    d['Suspension']['Temp'] = .003838
    d['Suspension']['Stage'][0].update(Mass=39.6, Temp=.003838)
    d['Materials'].update(MassRadius=.17, MassThickness=.20)
    d['Optics']['Curvature'].update(ITM=1934., ETM=2245.)
    d['Infrastructure']['ResidualGas']['SqueezedFilm'] = {'gap': .005}
    # GWINC templates use exponent spellings that plain PyYAML may leave as
    # strings before GWINC has registered its custom YAML number resolver.
    for species in SPECIES:
        for field in ('BeamtubePressure', 'ChamberPressure', 'mass', 'polarizability'):
            d['Infrastructure']['ResidualGas'][species][field] = float(
                d['Infrastructure']['ResidualGas'][species][field]
            )
    return d


def test_paper_benchmarks(paper_detector):
    result = calculate_residual_gas_constraints(paper_detector, 600.)
    optical = result['optical_pathlength']
    assert optical['pressure_ceiling_pa'] == pytest.approx(5.416e-14, rel=.001)
    assert optical['number_density_ceiling_m3'] == pytest.approx(1.022e12, rel=.001)
    assert result['damping_free_mass']['mixture_coefficient_per_pa_hz'] == pytest.approx(1.76581e-50, rel=.001)
    assert result['damping_infinite_free_mass']['mixture_coefficient_per_pa_hz'] == pytest.approx(1.76574e-50, rel=.001)
    assert sum(s['pressure_ceiling_pa'] for s in optical['species'].values()) == pytest.approx(optical['pressure_ceiling_pa'])


def test_pressure_allowance_scaling_and_species_sum(paper_detector):
    baseline = calculate_residual_gas_constraints(paper_detector, 600.)
    for s in SPECIES:
        paper_detector['Infrastructure']['ResidualGas'][s]['BeamtubePressure'] *= 3
        paper_detector['Infrastructure']['ResidualGas'][s]['ChamberPressure'] *= 3
    scaled = calculate_residual_gas_constraints(paper_detector, 600., 9e-27)
    for key in ('optical_pathlength', 'damping_free_mass'):
        assert scaled[key]['psd'] == pytest.approx(3*baseline[key]['psd'], rel=1e-12, abs=0)
        assert scaled[key]['pressure_ceiling_pa'] == pytest.approx(4*baseline[key]['pressure_ceiling_pa'], rel=1e-12)
        assert scaled[key]['psd'] == sum(e['psd'] for e in scaled[key]['species'].values())


def test_quadrature_convergence(paper_detector):
    a = calculate_residual_gas_spectra([1, 600, 5000], paper_detector, quadrature_rtol=1e-7)
    b = calculate_residual_gas_spectra([1, 600, 5000], paper_detector, quadrature_rtol=1e-11)
    for s in SPECIES:
        np.testing.assert_allclose(a.scattering_psd[s], b.scattering_psd[s], rtol=1e-7, atol=0)


def test_four_mirror_normalization_and_frequency_scaling(paper_detector):
    d = paper_detector
    d['Infrastructure']['ResidualGas']['SqueezedFilm'] = {}
    f = np.array([300., 600.])
    result = calculate_residual_gas_spectra(f, d)
    gas = d['Infrastructure']['ResidualGas']['H2']
    r, h, m, t, L = .17, .2, 39.6, .003838, 4000
    beta = np.pi*r*r*gas['ChamberPressure']/np.sqrt(KB*t/gas['mass'])*np.sqrt(8/np.pi)*(1+h/(2*r)+np.pi/4)
    expected = 4*(4*KB*t*beta)/(m**2*(2*np.pi*f)**4*L**2)
    np.testing.assert_allclose(result.damping_free_mass_psd['H2'], expected, rtol=1e-12, atol=0)
    assert result.damping_free_mass_psd['H2'][0]/result.damping_free_mass_psd['H2'][1] == pytest.approx(16)


def test_split_film_and_excess_equivalence(paper_detector):
    gas = paper_detector['Infrastructure']['ResidualGas']
    common = calculate_residual_gas_spectra([1, 600], paper_detector)
    gas['SqueezedFilm'] = {'ITM': {'gap': .005}, 'ETM': {'gap': .005}}
    split = calculate_residual_gas_spectra([1, 600], paper_detector)
    gas['SqueezedFilm'] = {'ITM': {'gap': .005}}
    half = calculate_residual_gas_spectra([1, 600], paper_detector)
    for s in SPECIES:
        np.testing.assert_allclose(common.damping_per_pa[s], split.damping_per_pa[s], atol=0)
        np.testing.assert_allclose(half.damping_per_pa[s], (common.damping_per_pa[s]+common.damping_infinite_per_pa[s])/2, atol=0)
    # A directly specified amplitude enhancement applies equally to all species.
    gas['SqueezedFilm'] = {'ExcessDamping': 3, 'DiffusionTime': .1}
    direct = calculate_residual_gas_spectra([1, 600], paper_detector)
    for s in SPECIES:
        np.testing.assert_allclose(direct.damping_per_pa[s]/direct.damping_infinite_per_pa[s], 1+8/(1+(2*np.pi*np.array([1,600])*.1)**2))


def test_independent_locations_zero_and_temperature_precedence(paper_detector):
    baseline = calculate_residual_gas_spectra([600], paper_detector)
    for s in SPECIES:
        paper_detector['Infrastructure']['ResidualGas'][s]['BeamtubePressure'] = 0
    result = calculate_residual_gas_constraints(paper_detector, 600)
    assert result['optical_pathlength']['psd'] == 0
    assert result['optical_pathlength']['pressure_ceiling_pa'] is None
    assert result['damping_free_mass']['pressure_ceiling_pa'] is not None
    paper_detector['Suspension']['Temp'] = 290  # Stage.Temp wins.
    unchanged = calculate_residual_gas_spectra([600], paper_detector)
    for s in SPECIES:
        np.testing.assert_array_equal(baseline.damping_per_pa[s], unchanged.damping_per_pa[s])


@pytest.mark.parametrize('keys,value,match', [
    (('Infrastructure','Temp'), 0, 'Infrastructure.Temp'),
    (('Infrastructure','ResidualGas','H2','mass'), -1, 'H2.mass'),
    (('Infrastructure','ResidualGas','H2','polarizability'), float('nan'), 'polarizability'),
    (('Infrastructure','ResidualGas','H2','ChamberPressure'), -1, 'ChamberPressure'),
    (('Infrastructure','ResidualGas','SqueezedFilm'), {'gap': 0}, 'gap'),
    (('Infrastructure','ResidualGas','SqueezedFilm'), {'ExcessDamping': 2}, 'DiffusionTime'),
    (('Infrastructure','ResidualGas','SqueezedFilm'), {'ExcessDamping': .5, 'DiffusionTime': 1}, 'ExcessDamping'),
    (('Infrastructure','ResidualGas','SqueezedFilm'), {'ITM': {}, 'gap': .005}, 'cannot mix'),
    (('Infrastructure','ResidualGas','SqueezedFilm'), None, 'mapping'),
    (('Optics','Curvature','ITM'), 100, 'unstable'),
])
def test_invalid_fields(paper_detector, keys, value, match):
    node = paper_detector
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    with pytest.raises(ConfigFileError, match=match):
        calculate_residual_gas_spectra([600], paper_detector)


@pytest.mark.parametrize('frequencies', [[], [0], [-1], [float('nan')], [[600]]])
def test_invalid_frequency(paper_detector, frequencies):
    with pytest.raises(ConfigFileError, match='frequency_hz'):
        calculate_residual_gas_spectra(frequencies, paper_detector)


def test_gwinc_matched_models_and_calibration(paper_detector):
    from gwinc import Struct
    from gwinc.ifo.noises import arm_cavity, dhdl
    from gwinc.noise.residualgas import residual_gas_scattering_arm, residual_gas_damping_test_mass
    d = deepcopy(paper_detector)
    d['Infrastructure']['Temp'] = 290.
    f = np.array([50., 600., 1000.])
    ifo = Struct(d)
    cavity = arm_cavity(ifo)
    ours = calculate_residual_gas_spectra(f, d)
    position = np.linspace(0, 4000, 20001)
    chi = -1/(39.6*(2*np.pi*f)**2)
    dhdl_sq, sinc_sq = dhdl(f, 4000.)
    for s in SPECIES:
        species = ifo.Infrastructure.ResidualGas[s]
        arm_psd = residual_gas_scattering_arm(f, ifo, cavity, species, species.BeamtubePressure*np.ones_like(position), position)
        # GWINC removes the finite-wavelength calibration only for scattering.
        calibrated = 2*arm_psd/sinc_sq*dhdl_sq
        np.testing.assert_allclose(ours.scattering_psd[s], calibrated, rtol=1e-7, atol=0)
        displacement = residual_gas_damping_test_mass(f, ifo, species, Struct(tst_suscept=chi), Struct(gap=.005))
        calibrated_damping = 4*displacement*dhdl_sq
        np.testing.assert_allclose(ours.damping_free_mass_psd[s]*sinc_sq, calibrated_damping, rtol=1e-12, atol=0)
