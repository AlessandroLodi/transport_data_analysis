"""Transport models for molecular junction and stability-diagram analysis.

Voltages and energies are expressed in volts/electron-volts, temperatures in
kelvin, and returned currents in amperes.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import expi, expit

BOLTZMANN_EV = 8.617e-5
PRECISE_BOLTZMANN_EV = 8.6173303e-5
CONDUCTANCE_QUANTUM = 7.7480917310e-5
CURRENT_CONVERSION = 1.6e-3 / 6.582
DEFAULT_ENERGY_LEVELS = 256


def _validate_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _fermi(
    energy: np.ndarray,
    temperature: float,
    boltzmann_constant: float = BOLTZMANN_EV,
) -> np.ndarray:
    """Numerically stable Fermi-Dirac distribution."""
    _validate_positive(temperature=temperature)
    return expit(-energy / (boltzmann_constant * temperature))


def _marcus_rate(
    energy_delta: np.ndarray,
    reorganisation_energy: float,
    temperature: float,
    broadening: float,
    *,
    reduction: bool,
) -> np.ndarray:
    """Return a Marcus hopping-rate density."""
    _validate_positive(
        reorganisation_energy=reorganisation_energy, temperature=temperature
    )
    beta = 1 / (BOLTZMANN_EV * temperature)
    denominator = 4 * reorganisation_energy / beta + broadening / (
        3 * BOLTZMANN_EV * temperature
    )
    if denominator <= 0:
        raise ValueError("broadening produces a non-positive rate denominator")
    sign = -1 if reduction else 1
    prefactor = np.sqrt(np.pi * beta / reorganisation_energy)
    return prefactor * np.exp(
        -((reorganisation_energy + sign * energy_delta) ** 2) / denominator
    )


def _single_site_current(
    energy: np.ndarray,
    molecular_energy: np.ndarray,
    source_bias: np.ndarray,
    source_coupling: float,
    drain_coupling: float,
    temperature: float,
    broadening: float,
    reorganisation_energy: float,
    spin: float,
) -> np.ndarray:
    """Integrate oxidation and reduction rates for one molecular site."""
    delta = energy - molecular_energy
    reduction = _marcus_rate(
        delta, reorganisation_energy, temperature, broadening, reduction=True
    )
    oxidation = _marcus_rate(
        delta, reorganisation_energy, temperature, broadening, reduction=False
    )
    source_occupation = _fermi(energy + source_bias, temperature)
    drain_occupation = _fermi(energy, temperature)

    oxidise_source = source_coupling**2 * np.trapz(
        (1 - source_occupation) * oxidation, energy, axis=-1
    )
    oxidise_drain = drain_coupling**2 * np.trapz(
        (1 - drain_occupation) * oxidation, energy, axis=-1
    )
    reduce_source = source_coupling**2 * np.trapz(
        source_occupation * reduction, energy, axis=-1
    )
    reduce_drain = drain_coupling**2 * np.trapz(
        drain_occupation * reduction, energy, axis=-1
    )
    if spin == 0:
        reduce_source *= 2
        reduce_drain *= 2
    else:
        oxidise_source *= 2
        oxidise_drain *= 2

    numerator = oxidise_source * reduce_drain - reduce_source * oxidise_drain
    denominator = oxidise_source + oxidise_drain + reduce_source + reduce_drain
    return CURRENT_CONVERSION * np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


class SpectralDensityModel:
    """Build a vibrational spectral density and calculate hopping rates."""

    def __init__(
        self,
        *,
        frequencies: np.ndarray | None = None,
        spectral_density: np.ndarray | None = None,
        energies: np.ndarray | None = None,
        times: np.ndarray | None = None,
        **legacy,
    ) -> None:
        frequencies = legacy.pop("w", frequencies)
        spectral_density = legacy.pop("J", spectral_density)
        energies = legacy.pop("E", energies)
        times = legacy.pop("time", times)
        if legacy:
            names = ", ".join(sorted(legacy))
            raise TypeError(f"unexpected keyword argument(s): {names}")

        self.w = np.asarray(
            frequencies if frequencies is not None else np.linspace(1e-5, 0.03, 2000),
            dtype=float,
        )
        self.J = np.asarray(
            spectral_density if spectral_density is not None else np.zeros(self.w.size),
            dtype=float,
        ).copy()
        self.E = np.asarray(
            energies if energies is not None else np.linspace(-0.1, 0.1, 5000),
            dtype=float,
        )
        self.time = np.asarray(
            times if times is not None else np.linspace(0, 16000, 5000),
            dtype=float,
        )
        if self.w.ndim != 1 or self.J.shape != self.w.shape:
            raise ValueError(
                "frequencies and spectral_density must be matching 1D arrays"
            )
        if self.E.ndim != 1 or self.time.ndim != 1:
            raise ValueError("energies and times must be one-dimensional")

        self.marcus_background = False
        self.modes: list[tuple[float, float]] = []
        self.specden = bool(np.any(self.J))

    def add_fake_damped_mode(self, height: float, width: float, omega: float) -> None:
        """Add the empirical damped-mode profile used by legacy analyses."""
        _validate_positive(width=width, omega=omega)
        self.specden = True
        numerator = (
            height
            * (self.w**3 / omega**3)
            * np.exp(-self.w / (width * omega))
            * np.exp(-1 / width)
        )
        denominator = np.exp(-2 * self.w / (width * omega)) + (self.w - omega) ** 2
        self.J += numerator / denominator

    def add_marcus_background(self, lmbd: float) -> None:
        """Record a classical Marcus reorganisation-energy background."""
        _validate_positive(lmbd=lmbd)
        self.marcus_background = True
        self.lmbd = lmbd

    def add_damped_mode(
        self, omega: float, n: float = 0.02, lambd: float = 5e-3, scale: float = 1
    ) -> None:
        """Add the damped vibrational mode of Roden et al. (JCP, 2012)."""
        _validate_positive(omega=omega, n=n, lambd=lambd)
        self.specden = True
        coupling = 0.9**2 * scale
        exponential_integral = expi(self.w / lambd)
        numerator = coupling * n * self.w**3 * np.exp(-self.w / lambd)
        denominator = np.pi**2 * n**2 * self.w**2 * np.exp(-2 * self.w / lambd)
        denominator += (
            self.w
            - omega
            + n * lambd
            - n * self.w * np.exp(-self.w / lambd) * exponential_integral
        ) ** 2
        self.J += numerator / denominator

    def add_mode(self, omega: float, g: float = -1) -> None:
        """Add an undamped mode, using ``0.9 * omega`` as default coupling."""
        _validate_positive(omega=omega)
        self.modes.append((omega, 0.9 * omega if g <= 0 else g))

    def add_background(self, lmbd: float, wc: float) -> None:
        """Add a smooth super-Ohmic background spectral density."""
        _validate_positive(lmbd=lmbd, wc=wc)
        self.specden = True
        self.J += (lmbd / 2) * (self.w / wc) ** 3 * np.exp(-self.w / wc)

    def calculate_rate_constants_from_B(self) -> None:
        """Fourier-transform the current correlation function ``B``."""
        if not hasattr(self, "B"):
            raise RuntimeError("calculate B before calculating rate constants")
        positive = np.exp(1j * np.outer(self.E, self.time)) * self.B
        negative = np.exp(-1j * np.outer(self.E, self.time)) * self.B
        self.k_red = 2 * np.real(np.trapz(positive, self.time, axis=1))
        self.k_ox = 2 * np.real(np.trapz(negative, self.time, axis=1))

    def calculate_rate_constants(self, Y: float = 0, T: float = 77) -> None:
        """Calculate the bath correlation function and hopping-rate spectra."""
        _validate_positive(T=T)
        if Y < 0:
            raise ValueError("Y must be non-negative")
        beta = 1 / (BOLTZMANN_EV * T)
        self.B = np.empty(self.time.size, dtype=np.complex128)
        for index, value in enumerate(self.time):
            exponent = (
                self.J
                / self.w**2
                * (
                    (np.cos(self.w * value) - 1) / np.tanh(beta * self.w / 2)
                    - 1j * np.sin(self.w * value)
                )
            )
            correlation = np.exp(np.trapz(exponent, self.w)) if self.specden else 1
            for omega, coupling in self.modes:
                mode = (coupling / omega) ** 2 * (
                    (np.cos(omega * value) - 1) / np.tanh(beta * omega / 2)
                    - 1j * np.sin(omega * value)
                )
                correlation *= np.exp(mode)
            self.B[index] = correlation
        self.B *= np.exp(-Y * self.time)
        self.calculate_rate_constants_from_B()

    def calculate_IV(self, Vbs, VS, VD, alpha_source, T, spin=0.5):
        """Calculate a current-voltage trace from precomputed rate spectra."""
        if not hasattr(self, "k_ox"):
            self.calculate_rate_constants(T=T)
        bias = np.asarray(Vbs, dtype=float)
        flat_bias = bias.ravel()
        energy = np.broadcast_to(self.E, (flat_bias.size, self.E.size))
        molecular_energy = (-alpha_source * flat_bias)[:, None]
        source_occupation = _fermi(energy + molecular_energy + flat_bias[:, None], T)
        drain_occupation = _fermi(energy + molecular_energy, T)
        current = self._integrate_rates(
            energy, source_occupation, drain_occupation, VS, VD, spin
        )
        return current.reshape(bias.shape)

    def calculate_IVsVg(
        self, Vbs, Vgs, VS, VD, alpha_source, alpha_gate, Vc, T, spin=0.5
    ):
        """Calculate current over corresponding bias and gate-voltage samples."""
        if not hasattr(self, "k_ox"):
            self.calculate_rate_constants(T=T)
        bias, gate = np.broadcast_arrays(
            np.asarray(Vbs, dtype=float), np.asarray(Vgs, dtype=float)
        )
        flat_bias = bias.ravel()
        molecular_energy = alpha_gate * (Vc - gate.ravel()) - alpha_source * flat_bias
        energy = np.broadcast_to(self.E, (flat_bias.size, self.E.size))
        source_occupation = _fermi(
            energy + molecular_energy[:, None] + flat_bias[:, None], T
        )
        drain_occupation = _fermi(energy + molecular_energy[:, None], T)
        current = self._integrate_rates(
            energy, source_occupation, drain_occupation, VS, VD, spin
        )
        return current.reshape(bias.shape)

    def _integrate_rates(
        self,
        energy,
        source_occupation,
        drain_occupation,
        source_coupling,
        drain_coupling,
        spin,
    ):
        oxidise_source = source_coupling**2 * np.trapz(
            (1 - source_occupation) * self.k_ox, energy, axis=1
        )
        oxidise_drain = drain_coupling**2 * np.trapz(
            (1 - drain_occupation) * self.k_ox, energy, axis=1
        )
        reduce_source = source_coupling**2 * np.trapz(
            source_occupation * self.k_red, energy, axis=1
        )
        reduce_drain = drain_coupling**2 * np.trapz(
            drain_occupation * self.k_red, energy, axis=1
        )
        if spin == 0:
            reduce_source *= 2
            reduce_drain *= 2
        else:
            oxidise_source *= 2
            oxidise_drain *= 2
        numerator = oxidise_source * reduce_drain - reduce_source * oxidise_drain
        denominator = oxidise_source + oxidise_drain + reduce_source + reduce_drain
        return CURRENT_CONVERSION * np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
        )

    def marcus_jortner_rate_constants(self, lmbd0, S, w0, T):
        """Calculate Marcus-Jortner rate spectra using 100 vibrational states."""
        _validate_positive(lmbd0=lmbd0, w0=w0, T=T)
        if S < 0:
            raise ValueError("S must be non-negative")
        self.E = np.linspace(-1, 1, 2000)
        denominator = 4 * lmbd0 * BOLTZMANN_EV * T
        prefactor = 2 * np.sqrt(np.pi / denominator)
        self.k_ox = np.zeros(self.E.size)
        self.k_red = np.zeros(self.E.size)
        for state in range(100):
            weight = prefactor * np.exp(-S) * S**state / factorial(state)
            self.k_ox += weight * np.exp(
                -((lmbd0 + state * w0 + self.E) ** 2 / denominator)
            )
            self.k_red += weight * np.exp(
                -((lmbd0 + state * w0 - self.E) ** 2 / denominator)
            )

    def marcus_rate_constants(self, lmbd, X, T):
        """Calculate classical Marcus rate spectra."""
        self.E = np.linspace(-1, 1, 2000)
        self.k_red = 2 * _marcus_rate(self.E, lmbd, T, X, reduction=True)
        self.k_ox = 2 * _marcus_rate(self.E, lmbd, T, X, reduction=False)


class PhysicsModels:
    """Stateless transport-model functions suitable for fitting routines."""

    @staticmethod
    def spectral_density(Vbs, lam, VL, VR):
        model = SpectralDensityModel()
        model.marcus_rate_constants(lam, 0, 77)
        return model.calculate_IV(np.asarray(Vbs), VL, VR, 0.0, 77)

    @staticmethod
    def improved_marcus(Vbs, lam, VS, VD, alpha_source, T, X, spin=0.5):
        bias = np.asarray(Vbs, dtype=float)
        flat_bias = bias.ravel()
        energy = np.broadcast_to(
            np.linspace(-1, 1, DEFAULT_ENERGY_LEVELS),
            (flat_bias.size, DEFAULT_ENERGY_LEVELS),
        )
        current = _single_site_current(
            energy,
            (-alpha_source * flat_bias)[:, None],
            flat_bias[:, None],
            VS,
            VD,
            T,
            X,
            lam,
            spin,
        )
        return current.reshape(bias.shape).flatten()

    @staticmethod
    def stabdiag_improved_marcus(
        data, lam, VS, VD, Vc, alpha_source, alpha_gate, T, X, spin=0.5
    ):
        gate, bias = np.broadcast_arrays(
            np.asarray(data[0], dtype=float), np.asarray(data[1], dtype=float)
        )
        flat_gate, flat_bias = gate.ravel(), bias.ravel()
        energy = np.broadcast_to(
            np.linspace(-1, 1, DEFAULT_ENERGY_LEVELS),
            (flat_bias.size, DEFAULT_ENERGY_LEVELS),
        )
        molecular_energy = (alpha_gate * (Vc - flat_gate) - alpha_source * flat_bias)[
            :, None
        ]
        return _single_site_current(
            energy, molecular_energy, flat_bias[:, None], VS, VD, T, X, lam, spin
        ).flatten()

    @staticmethod
    def marcus(Vbs, lam, VS, VD, alpha_source, T):
        return PhysicsModels.improved_marcus(Vbs, lam, VS, VD, alpha_source, T, 0)

    @staticmethod
    def stabdiag_marcus(data, lam, VS, VD, Vc, alpha_source, alpha_gate, T):
        return PhysicsModels.stabdiag_improved_marcus(
            data, lam, VS, VD, Vc, alpha_source, alpha_gate, T, 0
        )

    @staticmethod
    def doubledot(
        Vbs,
        lam,
        VS,
        VD,
        alpha_source1,
        alpha_source2,
        T,
        X=0,
        eps0_1=0,
        eps0_2=0,
        HAB=1e-3,
        spin=0.5,
    ):
        return PhysicsModels.doubledotSD(
            Vbs,
            np.zeros_like(Vbs),
            lam,
            VS,
            VD,
            alpha_source1,
            alpha_source2,
            0,
            T,
            X,
            eps0_1,
            eps0_2,
            HAB,
            spin,
        )

    @staticmethod
    def doubledotSD(
        Vbs,
        Vg,
        lam,
        VS,
        VD,
        alpha_source1,
        alpha_source2,
        alpha_gate,
        T,
        X=0,
        eps0_1=0,
        eps0_2=0,
        HAB=1e-3,
        spin=0.5,
    ):
        bias, gate = np.broadcast_arrays(
            np.asarray(Vbs, dtype=float), np.asarray(Vg, dtype=float)
        )
        bias, gate = bias.ravel(), gate.ravel()
        site_1 = eps0_1 - alpha_source1 * bias - alpha_gate * gate
        site_2 = eps0_2 - alpha_source2 * bias - alpha_gate * gate
        beta = 1 / (BOLTZMANN_EV * T)
        denominator = 4 * lam / beta + X / (3 * BOLTZMANN_EV * T)
        prefactor = np.sqrt(np.pi * beta / lam)
        rate_12 = prefactor * np.exp(-((lam - site_1 + site_2) ** 2) / denominator)
        rate_21 = prefactor * np.exp(-((lam - site_2 + site_1) ** 2) / denominator)
        energy = np.broadcast_to(
            np.linspace(-1, 1, DEFAULT_ENERGY_LEVELS),
            (bias.size, DEFAULT_ENERGY_LEVELS),
        )
        source_occupation = _fermi(energy + bias[:, None], T)
        drain_occupation = _fermi(energy, T)
        red_s = _marcus_rate(energy - site_1[:, None], lam, T, X, reduction=True)
        ox_s = _marcus_rate(energy - site_1[:, None], lam, T, X, reduction=False)
        red_d = _marcus_rate(energy - site_2[:, None], lam, T, X, reduction=True)
        ox_d = _marcus_rate(energy - site_2[:, None], lam, T, X, reduction=False)
        oxidise_source = VS**2 * np.trapz(
            (1 - source_occupation) * ox_s, energy, axis=1
        )
        oxidise_drain = VD**2 * np.trapz((1 - drain_occupation) * ox_d, energy, axis=1)
        reduce_source = VS**2 * np.trapz(source_occupation * red_s, energy, axis=1)
        reduce_drain = VD**2 * np.trapz(drain_occupation * red_d, energy, axis=1)
        if spin == 0:
            reduce_source *= 2
            reduce_drain *= 2
        else:
            oxidise_source *= 2
            oxidise_drain *= 2
        hop_12, hop_21 = HAB**2 * rate_12, HAB**2 * rate_21
        numerator = hop_21 * reduce_drain * oxidise_source
        numerator -= hop_12 * oxidise_drain * reduce_source
        denominator = (
            oxidise_source * oxidise_drain
            + hop_12 * oxidise_drain
            + hop_21 * oxidise_source
            + reduce_drain * (oxidise_source + hop_12 + hop_21)
            + reduce_source * (oxidise_drain + hop_12 + hop_21)
        )
        return CURRENT_CONVERSION * np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
        )

    @staticmethod
    def simmons(x, A, phi, alpha, d):
        """Evaluate the legacy Simmons tunnelling-current approximation."""
        beta = 10.25
        currents = []
        for bias in np.asarray(x, dtype=float):
            phi_left = (1 + alpha) * phi + bias / 2
            phi_right = (1 - alpha) * phi - bias / 2
            energies = np.arange(-2 * phi, abs(bias / 2), 0.01)
            left = -(energies - bias / 2) * (energies - bias / 2 < 0)
            right = -(energies + bias / 2) * (energies + bias / 2 < 0)
            barrier = []
            for energy in energies:
                if energy > phi_right:
                    value = (
                        d
                        * (2 / 3)
                        * (phi_left - energy) ** 1.5
                        / (phi_left - phi_right)
                    )
                elif energy > phi_left:
                    value = (
                        d
                        * (2 / 3)
                        * (phi_right - energy) ** 1.5
                        / (phi_right - phi_left)
                    )
                else:
                    value = (
                        d
                        * (2 / 3)
                        * ((phi_right - energy) ** 1.5 - (phi_left - energy) ** 1.5)
                        / (phi_right - phi_left)
                    )
                barrier.append(value)
            currents.append(
                np.trapz((left - right) * np.exp(-beta * np.asarray(barrier)))
            )
        return A * np.asarray(currents)

    @staticmethod
    def lifetime_broadening(Vg, Vc, gamma, alpha):
        gamma_squared = gamma**2
        return (
            CONDUCTANCE_QUANTUM
            * gamma_squared
            / (gamma_squared / 4 + (alpha * (np.asarray(Vg) - Vc)) ** 2)
        )

    @staticmethod
    def sub_threshold_swing_pure(Isd, Vg):
        dx = Vg[0, 1] - Vg[0, 0]
        return np.gradient(np.log10(Isd), axis=1) ** -1 * dx

    @staticmethod
    def thermal_broadening(Vg, T, Vc, Gmax, alpha):
        _validate_positive(T=T)
        argument = alpha * (np.asarray(Vg) - Vc) / (2 * PRECISE_BOLTZMANN_EV * T)
        return Gmax / np.cosh(argument) ** 2

    @staticmethod
    def stabdiag(data, prefactor, Vc, alpha_gate, alpha_source, T):
        gate, bias = np.asarray(data[0]), np.asarray(data[1])
        chemical_potential = alpha_gate * (Vc - gate) - alpha_source * bias
        return prefactor * (
            _fermi(chemical_potential, T, PRECISE_BOLTZMANN_EV)
            - _fermi(chemical_potential + bias, T, PRECISE_BOLTZMANN_EV)
        )

    @staticmethod
    def curved_improved_marcus(
        data, lam, VS, VD, Vc, alpha_source, alpha_gate, alpha_J, J, T, X, spin=0.5
    ):
        gate, bias = np.asarray(data[0]), np.asarray(data[1])
        curved_gate = gate + np.sqrt((alpha_J * bias) ** 2 + 4 * J**2) / (
            2 * alpha_gate
        )
        return PhysicsModels.stabdiag_improved_marcus(
            (curved_gate, bias), lam, VS, VD, Vc, alpha_source, alpha_gate, T, X, spin
        )

    @staticmethod
    def curved_stabdiag(data, prefactor, Vc, alpha_gate, alpha_source, alpha_J, T, J):
        gate, bias = np.asarray(data[0]), np.asarray(data[1])
        chemical_potential = (
            alpha_gate * (Vc - gate)
            - alpha_source * bias
            - np.sqrt((alpha_J * bias) ** 2 + 4 * J**2) / 2
        )
        return prefactor * (
            _fermi(chemical_potential, T, PRECISE_BOLTZMANN_EV)
            - _fermi(chemical_potential + bias, T, PRECISE_BOLTZMANN_EV)
        )


spectral_density_model = SpectralDensityModel
physics_models = PhysicsModels

__all__ = [
    "BOLTZMANN_EV",
    "PhysicsModels",
    "SpectralDensityModel",
    "physics_models",
    "spectral_density_model",
]
