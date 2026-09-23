"""Notebook-equivalent residual gas estimates (one-sided strain PSD, 1/Hz).

Optical phase noise includes two independent arms. Damping includes four
independent mirrors and uses a free-mass susceptibility, NOT the suspension
response used by the full GWINC budget. Pressures are uniform within each
location; beam tubes and chambers may have different mixtures/temperatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from scipy.integrate import quad
from scipy.constants import Boltzmann as KB

from .config import ConfigFileError

SPECIES = ('H2', 'N2', 'H2O', 'O2')


def number(value, field: str, *, zero: bool = False) -> float:
    """Validate physical scalars without accepting booleans or nonfinite values."""
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ConfigFileError(f'{field} must be a finite number') from exc
    if isinstance(value, bool) or not np.isfinite(result) or (result < 0 if zero else result <= 0):
        raise ConfigFileError(f'{field} must be finite and {"nonnegative" if zero else "positive"}')
    return result


def at(document, *keys):
    """Read a required YAML path, retaining the field name in errors."""
    value = document
    try:
        for key in keys:
            value = value[key]
    except (KeyError, TypeError, IndexError) as exc:
        raise ConfigFileError(f'Missing required field {".".join(map(str, keys))}') from exc
    return value


def physical(document, *keys, zero=False):
    return number(at(document, *keys), '.'.join(map(str, keys)), zero=zero)


def _film_geometry(value, path):
    if not isinstance(value, Mapping):
        raise ConfigFileError(f'{path} must be a mapping (use {{}} for no film)')
    if not value:
        return {}
    if set(value) == {'gap'}:
        return {'gap': number(value['gap'], path + '.gap')}
    if set(value) == {'ExcessDamping', 'DiffusionTime'}:
        excess = number(value['ExcessDamping'], path + '.ExcessDamping')
        if excess < 1:
            raise ConfigFileError(f'{path}.ExcessDamping must be >= 1 (amplitude ratio)')
        return {'ExcessDamping': excess,
                'DiffusionTime': number(value['DiffusionTime'], path + '.DiffusionTime')}
    raise ConfigFileError(f'{path}: specify gap OR ExcessDamping with DiffusionTime')


def film_geometries(gas):
    path = 'Infrastructure.ResidualGas.SqueezedFilm'
    film = gas.get('SqueezedFilm', {})
    if isinstance(film, Mapping) and ('ITM' in film or 'ETM' in film):
        if set(film) - {'ITM', 'ETM'}:
            raise ConfigFileError(f'{path}: cannot mix common and ITM/ETM geometry')
        return tuple(_film_geometry(film.get(mirror, {}), path + '.' + mirror)
                     for mirror in ('ITM', 'ETM'))
    common = _film_geometry(film, path)
    return common, common


@dataclass
class GasSpectra:
    frequency_hz: np.ndarray
    scattering_psd: dict[str, np.ndarray]
    damping_free_mass_psd: dict[str, np.ndarray]
    damping_infinite_psd: dict[str, np.ndarray]
    # Per-unit-pressure coefficients allow meaningful limits even when an
    # individual species has zero pressure in the configured mixture.
    scattering_per_pa: dict[str, np.ndarray]
    damping_per_pa: dict[str, np.ndarray]
    damping_infinite_per_pa: dict[str, np.ndarray]
    temperatures_k: dict[str, float]
    pressures_pa: dict[str, dict[str, float]]
    beam: dict[str, float]


def calculate_residual_gas_spectra(frequency_hz, detector, *, quadrature_rtol=1e-9) -> GasSpectra:
    """Evaluate notebook models at positive frequencies (Hz).

    Each species array has shape (number of frequencies,). The dimensionless
    beam integral is integrated in z/z_R and scaled by its peak exponential.
    This avoids absolute quadrature tolerances swallowing the tiny cryogenic
    optical-pathlength signal. A relative-only error target checks convergence.
    """
    f = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    if f.ndim != 1 or not f.size or not np.all(np.isfinite(f) & (f > 0)):
        raise ConfigFileError('frequency_hz must be a nonempty positive finite 1D array')
    rtol = number(quadrature_rtol, 'quadrature_rtol')
    if not 1e-12 <= rtol <= 1e-3:
        raise ConfigFileError('quadrature_rtol must be between 1e-12 and 1e-3')
    L = physical(detector, 'Infrastructure', 'Length')
    wavelength = physical(detector, 'Laser', 'Wavelength')
    r1 = physical(detector, 'Optics', 'Curvature', 'ITM')
    r2 = physical(detector, 'Optics', 'Curvature', 'ETM')
    g1, g2 = 1-L/r1, 1-L/r2
    if not 0 < g1*g2 < 1:
        raise ConfigFileError('Optics.Curvature and Infrastructure.Length define an unstable arm cavity')
    gc = np.sqrt(g1*g2*(1-g1*g2))
    den = g1-2*g1*g2+g2
    w0 = np.sqrt(L*wavelength/np.pi) * np.sqrt(gc/abs(den))
    zr = np.pi*w0*w0/wavelength
    zw = L*g2*(1-g1)/den
    tube_temp = physical(detector, 'Infrastructure', 'Temp')
    stage = at(detector, 'Suspension', 'Stage', 0)
    chamber_temp = (number(stage['Temp'], 'Suspension.Stage.0.Temp') if 'Temp' in stage
                    else physical(detector, 'Suspension', 'Temp'))
    mass = physical(detector, 'Suspension', 'Stage', 0, 'Mass')
    radius = physical(detector, 'Materials', 'MassRadius')
    thickness = physical(detector, 'Materials', 'MassThickness')
    gas = at(detector, 'Infrastructure', 'ResidualGas')
    if not isinstance(gas, Mapping):
        raise ConfigFileError('Infrastructure.ResidualGas must be a mapping')
    unknown = set(gas) - set(SPECIES) - {'SqueezedFilm'}
    if unknown:
        raise ConfigFileError(f'Infrastructure.ResidualGas unsupported species/fields: {sorted(unknown)}')
    films = film_geometries(gas)
    scattering, damping, infinite = {}, {}, {}
    c_scatter, c_damp, c_inf = {}, {}, {}
    pressures = {'beamtube': {}, 'chamber': {}}
    omega = 2*np.pi*f
    # |chi|^2 = 1/[M^2 (2*pi*f)^4]; displacement / L -> strain.
    susceptibility_sq = 1/(mass**2 * omega**4)
    lower, upper = -zw/zr, (L-zw)/zr
    nearest = np.clip(0., lower, upper)
    peak_shape = np.hypot(1., nearest)
    for species in SPECIES:
        path = ('Infrastructure', 'ResidualGas', species)
        molecule_mass = physical(detector, *path, 'mass')
        alpha = physical(detector, *path, 'polarizability')
        ptube = physical(detector, *path, 'BeamtubePressure', zero=True)
        pchamber = physical(detector, *path, 'ChamberPressure', zero=True)
        pressures['beamtube'][species] = ptube
        pressures['chamber'][species] = pchamber
        v0 = np.sqrt(2*KB*tube_temp/molecule_mass)
        integrals = []
        for frequency in f:
            a = 2*np.pi*frequency*w0/v0
            def integrand(u):
                shape = np.hypot(1., u)
                return np.exp(-a*(shape-peak_shape))/shape
            value, error = quad(integrand, lower, upper, epsabs=0,
                                epsrel=rtol, points=[nearest], limit=250)
            if error > max(5*rtol*abs(value), np.finfo(float).tiny):
                raise RuntimeError(f'Gas optical quadrature did not converge for {species} at {frequency} Hz')
            integrals.append(value * np.exp(-a*peak_shape) * zr/w0)
        # Two-arm optical phase PSD: 8(2*pi*alpha)^2 rho/(v0 L^2) int dz/w.
        c_scatter[species] = (8*(2*np.pi*alpha)**2/(v0*L**2*KB*tube_temp)
                              * np.asarray(integrals))
        scattering[species] = ptube*c_scatter[species]
        thermal_v = np.sqrt(KB*chamber_temp/molecule_mass)
        beta_per_pa = (np.pi*radius**2/thermal_v*np.sqrt(8/np.pi)
                       *(1+thickness/(2*radius)+np.pi/4))
        force_per_pa = 4*KB*chamber_temp*beta_per_pa
        # Two ITMs and two ETMs, with independent force fluctuations.
        force_sum = np.zeros_like(f)
        for film in films:
            extra = np.zeros_like(f)
            if film:
                if 'gap' in film:
                    gap = film['gap']
                    tau = np.sqrt(np.pi/2)*radius**2/(gap*thermal_v*np.log1p((radius/gap)**2))
                    delta = 4*KB*chamber_temp*np.pi*radius**2*tau/gap
                else:
                    tau = film['DiffusionTime']
                    delta = (film['ExcessDamping']**2-1)*force_per_pa
                extra = delta/(1+(omega*tau)**2)
            force_sum += 2*(force_per_pa+extra)
        c_damp[species] = force_sum*susceptibility_sq/L**2
        c_inf[species] = 4*force_per_pa*susceptibility_sq/L**2
        damping[species] = pchamber*c_damp[species]
        infinite[species] = pchamber*c_inf[species]
    return GasSpectra(f, scattering, damping, infinite, c_scatter, c_damp, c_inf,
                      {'beamtube': tube_temp, 'chamber': chamber_temp}, pressures,
                      {'waist_radius_m': float(w0), 'rayleigh_range_m': float(zr),
                       'waist_position_m': float(zw)})


def calculate_residual_gas_constraints(detector, frequency_hz: float,
                                      allowance_asd: float = 4.50e-27) -> dict:
    """Conditional fixed-composition ceilings, independently allocating A^2.

    For mixture fractions x_i=P_i/sum(P_i), K=sum(x_i S_i/P_i), hence
    P_max=A^2/K and rho_max=P_max/(k_B T). These are not statistical bounds
    and the same allowance cannot be spent on all mechanisms simultaneously.
    """
    frequency = number(frequency_hz, 'Noise.Estimation.TargetFrequency')
    allowance = number(allowance_asd, 'Noise.Estimation.GasAllowanceASD')
    gas = calculate_residual_gas_spectra([frequency], detector)
    result = {'frequency_hz': frequency, 'allowance_asd': allowance,
              'allocation': 'Full allowance independently assigned to each mechanism; not simultaneous.',
              'temperatures_k': gas.temperatures_k, 'beam': gas.beam}
    for name, location, coefficients in (
        ('optical_pathlength', 'beamtube', gas.scattering_per_pa),
        ('damping_free_mass', 'chamber', gas.damping_per_pa),
        ('damping_infinite_free_mass', 'chamber', gas.damping_infinite_per_pa),
    ):
        pressures = gas.pressures_pa[location]
        total_pressure = sum(pressures.values())
        entries = {s: {'pressure_pa': pressures[s],
                       'number_density_m3': pressures[s]/(KB*gas.temperatures_k[location]),
                       'coefficient_per_pa_hz': float(coefficients[s][0]),
                       'coefficient_per_number_density': float(coefficients[s][0]*KB*gas.temperatures_k[location]),
                       'psd': float(pressures[s]*coefficients[s][0])}
                   for s in SPECIES}
        item = {'location': location, 'pressure_pa': total_pressure, 'species': entries,
                'psd': sum(e['psd'] for e in entries.values())}
        item['asd'] = float(np.sqrt(item['psd']))
        item['pressure_ceiling_pa'] = None
        item['number_density_ceiling_m3'] = None
        item['mixture_coefficient_per_pa_hz'] = None
        if total_pressure == 0:
            item['status'] = 'unavailable: zero total pressure leaves mixture undefined'
        else:
            fractions = {s: pressures[s]/total_pressure for s in SPECIES}
            coefficient = sum(fractions[s]*coefficients[s][0] for s in SPECIES)
            item['mixture_coefficient_per_pa_hz'] = float(coefficient)
            for s in SPECIES:
                entries[s]['fraction'] = fractions[s]
            if coefficient == 0:
                item['status'] = 'unavailable: coefficient below floating-point range'
            else:
                ceiling = float(allowance**2/coefficient)
                item.update(status='conditional', pressure_ceiling_pa=ceiling,
                            number_density_ceiling_m3=ceiling/(KB*gas.temperatures_k[location]))
                for s in SPECIES:
                    entries[s]['pressure_ceiling_pa'] = fractions[s]*ceiling
                    entries[s]['number_density_ceiling_m3'] = fractions[s]*item['number_density_ceiling_m3']
        result[name] = item
    return result
