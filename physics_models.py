import numpy as np
from scipy.special import expi
import matplotlib.pyplot as plt


class spectral_density_model:
    nM = np.array([[0, 0], [0, 1]])  # what are these?
    aD = np.array([[0, 0], [1, 0]])
    a = aD.T

    N = 2  # what does N do?

    Tph = 77  # phonon temp, in K
    Te = 77  # electron temp, same as phonon
    kb = 8.617e-5  # boltzmann constant in eV.K-1

    def __init__(self, **kwargs):
        self.betaph = 1 / (self.kb * self.Tph)
        self.w = kwargs.get("w", np.linspace(1e-5, 0.03, 5000))
        self.J = kwargs.get("J", np.zeros(len(self.w)))
        self.modes = []
        self.specden = False

    def add_fake_damped_mode(self, height, width, omega):
        self.specden = True
        num = (
            height
            * (self.w ** 3 / omega ** 3)
            * np.exp(-self.w / (width * omega))
            * np.exp(-omega / (width * omega))
        )
        den = (np.exp(-2 * self.w / (width * omega))) + (self.w - omega) ** 2

        self.J += num / den

    # according to Roden JCP (2012), Eq. (45)
    def add_damped_mode(self, omega, n=0.02, lambd=5e-3, scale=1):
        self.specden = True
        g = 0.9 * omega
        X = (g / omega) ** 2 * scale
        Ei = expi(self.w / lambd)
        J = X * n * (self.w ** 3) * np.exp(-self.w / lambd)
        J2 = np.pi ** 2 * n ** 2 * self.w ** 2 * np.exp(-2 * self.w / lambd)
        J2 += (
            self.w - omega + n * lambd - n * self.w * np.exp(-self.w / lambd) * Ei
        ) ** 2
        J /= J2
        self.J += J

    def add_mode(self, omega, g=-1):
        # if g <= 0:
        #   g = omega*0.9
        self.modes.append([omega, g])

    def add_background(self, lmbd, wc):
        self.specden = True
        self.J += (lmbd / 2) * (self.w / wc) ** 3 * np.exp(-self.w / wc)

    def calculate_rate_constants(self, Y=0, T=77):
        betaph = 1 / (self.kb * T)
        self.time = np.linspace(0, 16000, 10000)
        time = self.time

        # calculate B(t) correlation function
        self.B = np.empty(len(time), dtype=np.complex64)
        for index, t in enumerate(time):
            C1 = (
                (self.J)
                / (self.w ** 2)
                * (
                    (np.cos(self.w * t) - 1) / np.tanh(betaph * self.w / 2)
                    - 1j * np.sin(self.w * t)
                )
            )
            self.B[index] = np.exp(np.trapz(C1, self.w)) if self.specden else 1
            for mode in self.modes:
                omega, g = mode
                D1 = (g ** 2 / omega ** 2) * (
                    (np.cos(omega * t) - 1) / np.tanh(betaph * omega / 2)
                    - 1j * np.sin(omega * t)
                )
                self.B[index] *= np.exp(D1)
        # include lifetime broadening
        self.B *= np.exp(-Y * time)

        # calculate the hopping rates from the correlation function
        self.E = np.linspace(-0.1, 0.1, 2000)
        kred, kox = [np.empty(len(self.E), dtype=np.complex64) for _ in range(2)]
        for index, eps in enumerate(self.E):
            kred[index] = np.trapz(np.exp(1j * time * eps) * self.B, time)
            kox[index] = np.trapz(np.exp(-1j * time * eps) * self.B, time)
        self.k_ox = np.real(kox) * 2
        self.k_red = np.real(kred) * 2

    def calculate_IV(self, Vbs, VS, VD, alpha_source, T, spin=0.5):
        try:
            _ = self.k_ox
        except:
            self.calculate_rate_constants(VS, VD)
        kb = 8.617e-5  # boltzmann constant in eV.K-1

        # prepare 3D arrays of epsilon (mol level) and E for integration
        eps0 = 0
        eps_m = (
            eps0 - alpha_source * Vbs
        )  # eps is raised by Vs and Vg by their corresponding alphas. Vd = 0
        E = np.repeat(self.E[np.newaxis, :], len(eps_m), axis=0)
        eps_m = np.repeat(eps_m[:, np.newaxis], len(self.E), axis=1)
        Vbs = np.repeat(Vbs[:, np.newaxis], len(self.E), axis=1)

        # calculate the Fermi-Dirac distributions
        fS = 1 / (1 + np.exp((E + eps_m + Vbs) / (kb * T)))  # at Vb = Vb V
        fD = 1 / (1 + np.exp((E + eps_m) / (kb * T)))  # at Vb = 0 V

        # calculate the hopping rates
        y_ox_S = VS ** 2 * np.trapz((1 - fS) * self.k_ox, E, axis=1)
        y_ox_D = VD ** 2 * np.trapz((1 - fD) * self.k_ox, E, axis=1)
        y_red_S = VS ** 2 * np.trapz(fS * self.k_red, E, axis=1)
        y_red_D = VD ** 2 * np.trapz(fD * self.k_red, E, axis=1)

        if spin == 0:
            y_red_S *= 2
            y_red_D *= 2
        else:
            y_ox_S *= 2
            y_ox_D *= 2

        # calculate the current
        return (
            (1.6e-3 / 6.582)
            * (y_ox_S * y_red_D - y_red_S * y_ox_D)
            / (y_ox_S + y_ox_D + y_red_S + y_red_D)
        )

    def calculate_IVsVg(
        self, Vbs, Vgs, VS, VD, alpha_source, Vc, alpha_gate, T, spin=0.5
    ):
        try:
            _ = self.k_ox
        except:
            self.calculate_rate_constants(VS, VD)
        kb = 8.617e-5  # boltzmann constant in eV.K-1

        # prepare 3D arrays of epsilon (mol level) and E for integration
        eps0 = alpha_gate * Vc
        eps_m = (
            eps0 - alpha_gate * Vgs - alpha_source * Vbs
        )  # eps is lowered by Vs and Vg by their corresponding alphas. Vd = 0
        E = np.repeat(self.E[np.newaxis, :], eps_m.shape[1], axis=0)
        E = np.repeat(E[np.newaxis, :, :], eps_m.shape[0], axis=0)
        eps_m = np.repeat(eps_m[:, :, np.newaxis], len(self.E), axis=2)
        Vbs = np.repeat(Vbs[:, :, np.newaxis], len(self.E), axis=2)

        # calculate the Fermi-Dirac distributions
        fS = 1 / (1 + np.exp((E + eps_m + Vbs) / (kb * T)))  # at Vb = Vb V
        fD = 1 / (1 + np.exp((E + eps_m) / (kb * T)))  # at Vb = 0 V

        # calculate the hopping rates
        y_ox_S = np.real(2 * VS ** 2 * np.trapz((1 - fS) * self.k_ox, E, axis=2))
        y_ox_D = np.real(2 * VD ** 2 * np.trapz((1 - fD) * self.k_ox, E, axis=2))
        y_red_S = np.real(2 * VS ** 2 * np.trapz(fS * self.k_red, E, axis=2))
        y_red_D = np.real(2 * VD ** 2 * np.trapz(fD * self.k_red, E, axis=2))

        if spin == 0:
            y_red_S *= 2
            y_red_D *= 2
        else:
            y_ox_S *= 2
            y_ox_D *= 2

        # calculate the current
        return (
            (1.6e-3 / 6.582)
            * (y_ox_S * y_red_D - y_red_S * y_ox_D)
            / (y_ox_S + y_ox_D + y_red_S + y_red_D)
        )

    def visualise_is(self):
        plt.plot(self.E, self.i1)
        plt.plot(self.E, self.i2)
        plt.plot(self.E, self.i3)
        plt.plot(self.E, self.i4)
        plt.show()


class physics_models:
    @staticmethod
    def spectral_density(Vbs, lam, VL, VR):
        model = spectral_density_model()
        model.calculate_simple_SD(lam, 0.015)
        return model.single_site(Vbs, VL, VR, 0.0)

    @staticmethod
    def improved_marcus(Vbs, lam, VS, VD, alpha_source, T, X, spin=0.5):
        print(
            ">> Fitting IV trace... lam: {}, VS: {}, VD: {}, X: {}".format(
                lam, VS, VD, X
            )
        )

        kb = 8.617e-5  # boltzmann constant in eV.K-1
        Tph = T
        Te = T
        betaph = 1 / (kb * Tph)

        levels = 256
        E = np.linspace(-1, 1, levels)  # = basically -inf to inf, increase as needed

        # prepare 3D arrays of epsilon (mol level) and E for integration
        eps0 = 0
        eps_m = (
            eps0 - alpha_source * Vbs
        )  # eps is raised by Vs and Vg by their corresponding alphas. Vd = 0
        E = np.repeat(E[np.newaxis, :], len(eps_m), axis=0)
        eps_m = np.repeat(eps_m[:, np.newaxis], levels, axis=1)
        Vbs = np.repeat(Vbs[:, np.newaxis], levels, axis=1)

        # calculate the hopping rate constants
        k_red = (np.pi * betaph / lam) ** 0.5 * np.exp(
            -((lam - (E - eps_m)) ** 2) / (4 * lam / betaph + X / (3 * kb * T))
        )
        k_ox = (np.pi * betaph / lam) ** 0.5 * np.exp(
            -((lam + (E - eps_m)) ** 2) / (4 * lam / betaph + X / (3 * kb * T))
        )

        # calculate the Fermi-Dirac distributions
        fS = 1 / (1 + np.exp((E + Vbs) / (kb * Te)))  # at Vb = Vb V
        fD = 1 / (1 + np.exp((E) / (kb * Te)))  # at Vb = 0 V

        # calculate the hopping rates
        y_ox_S = VS ** 2 * np.trapz((1 - fS) * k_ox, E, axis=1)
        y_ox_D = VD ** 2 * np.trapz((1 - fD) * k_ox, E, axis=1)
        y_red_S = VS ** 2 * np.trapz(fS * k_red, E, axis=1)
        y_red_D = VD ** 2 * np.trapz(fD * k_red, E, axis=1)

        if spin == 0:
            y_red_S *= 2
            y_red_D *= 2
        else:
            y_ox_S *= 2
            y_ox_D *= 2

        # calculate the current
        I = (y_ox_S * y_red_D - y_red_S * y_ox_D) / (
            y_ox_S + y_ox_D + y_red_S + y_red_D
        )

        return I.flatten() * 1.6e-3 / 6.582

    @staticmethod
    def stabdiag_improved_marcus(
        data, lam, VS, VD, Vc, alpha_source, alpha_gate, T, X, spin=0.5
    ):
        # kb = 8.6173303e-5  # eV.K-1
        Vbs = data[1]
        Vg = data[0]

        print(
            ">> Fitting stability diagram... lam: {}, VS: {}, VD: {}, X: {}".format(
                lam, VS, VD, X
            )
        )

        if len(Vbs.shape) == 1:
            Vbs = np.reshape(Vbs, (-1, len(np.unique(Vg))))
            Vg = np.reshape(Vg, (-1, len(np.unique(Vg))))

        kb = 8.617e-5  # boltzmann constant in eV.K-1
        Tph = T
        Te = T
        betaph = 1 / (kb * Tph)

        levels = 256
        E = np.linspace(-1, 1, levels)  # = basically -inf to inf, increase as needed

        # prepare 3D arrays of epsilon (mol level) and E for integration
        eps0 = alpha_gate * Vc
        eps_m = (
            eps0 - alpha_gate * Vg - alpha_source * Vbs
        )  # eps is lowered by Vs and Vg by their corresponding alphas. Vd = 0
        E = np.repeat(E[np.newaxis, :], eps_m.shape[1], axis=0)
        E = np.repeat(E[np.newaxis, :, :], eps_m.shape[0], axis=0)
        eps_m = np.repeat(eps_m[:, :, np.newaxis], levels, axis=2)
        Vbs = np.repeat(Vbs[:, :, np.newaxis], levels, axis=2)

        # calculate the hopping rate constants
        k_red = (np.pi * betaph / lam) ** 0.5 * np.exp(
            -((lam - (E - eps_m)) ** 2) / (4 * lam / betaph + X / (3 * kb * T))
        )
        k_ox = (np.pi * betaph / lam) ** 0.5 * np.exp(
            -((lam + (E - eps_m)) ** 2) / (4 * lam / betaph + X / (3 * kb * T))
        )

        # calculate the Fermi-Dirac distributions
        fS = 1 / (1 + np.exp((E + Vbs) / (kb * Te)))  # at Vb = Vb V
        fD = 1 / (1 + np.exp((E) / (kb * Te)))  # at Vb = 0 V

        # calculate the hopping rates
        y_ox_S = VS ** 2 * np.trapz((1 - fS) * k_ox, E, axis=2)
        y_ox_D = VD ** 2 * np.trapz((1 - fD) * k_ox, E, axis=2)
        y_red_S = VS ** 2 * np.trapz(fS * k_red, E, axis=2)
        y_red_D = VD ** 2 * np.trapz(fD * k_red, E, axis=2)

        if spin == 0:
            y_red_S *= 2
            y_red_D *= 2
        else:
            y_ox_S *= 2
            y_ox_D *= 2

        # calculate the current
        I = (y_ox_S * y_red_D - y_red_S * y_ox_D) / (
            y_ox_S + y_ox_D + y_red_S + y_red_D
        )

        return I.flatten() * 1.6e-3 / 6.582

    @staticmethod
    def marcus(Vbs, lam, VS, VD, alpha_source, T):
        return physics_models.improved_marcus(Vbs, lam, VS, VD, alpha_source, T, 0)

    @staticmethod
    def stabdiag_marcus(data, lam, VS, VD, Vc, alpha_source, alpha_gate, T):
        return physics_models.stabdiag_improved_marcus(
            data, lam, VS, VD, Vc, alpha_source, alpha_gate, T, 0
        )

    @staticmethod
    def simmons(x, A, phi, alpha, d):
        beta = 10.25  # eV nm
        I = []
        for i in range(len(x)):
            phi_L = (1 + alpha) * phi + x[i] / 2
            phi_R = (1 - alpha) * phi - x[i] / 2

            Ex = np.arange(-2 * phi, abs(x[i] / 2), 0.01)
            nL = -(Ex - x[i] / 2) * (Ex - x[i] / 2 < 0)
            nR = -(Ex + x[i] / 2) * (Ex + x[i] / 2 < 0)

            F = []
            for j in range(len(Ex)):
                if Ex[j] > phi_R:
                    F.append(
                        -d * (2 / 3) * (phi_L - Ex[j]) ** (3 / 2) / (phi_R - phi_L)
                    )
                elif Ex[j] > phi_L:
                    F.append(d * (2 / 3) * (phi_R - Ex[j]) ** (3 / 2) / (phi_R - phi_L))
                else:
                    F.append(
                        d
                        * (2 / 3)
                        * ((phi_R - Ex[j]) ** (3 / 2) - (phi_L - Ex[j]) ** (3 / 2))
                        / (phi_R - phi_L)
                    )

            I.append(np.trapz((nL - nR) * np.exp(-beta * np.array(F))))
        return A * (I / np.max(I))

    @staticmethod
    def lifetime_broadening(Vg, Vc, gamma, alpha):
        G0 = 7.7480917310e-5  # S
        gamma2 = gamma * gamma
        return G0 * gamma2 / (gamma2 / 4 + (alpha * (Vg - Vc)) ** 2)

    @staticmethod
    def thermal_broadening(Vg, T, Vc, Gmax, alpha):
        e = 1.602e-19
        # kb = 8.6173303e-8  # eV.K-1
        kb = 1.3806e-23  # J.K-1
        # T = 77 # K
        kb = 8.6173303e-5  # eV.K-1
        return Gmax / (np.cosh((alpha * (Vg - Vc)) / (2 * kb * T)) ** 2)

    @staticmethod
    def stabdiag(data, prefactor, Vc, alpha_gate, alpha_source, T):
        kb = 8.6173303e-5  # eV.K-1
        Vs = data[1]
        Vg = data[0]
        mu = (
            alpha_gate * Vc - alpha_gate * Vg - alpha_source * Vs
        )  # == mu0 - alpha_gate * Vg - alpha_source * Vs
        fd = 1 / (1 + np.exp(mu / (kb * T)))
        fs = 1 / (1 + np.exp((mu + Vs) / (kb * T)))
        return prefactor * (fd - fs)

    @staticmethod
    def curved_stabdiag(data, prefactor, Vc, alpha_gate, alpha_source, alpha_J, T, J):
        kb = 8.6173303e-5  # eV.K-1
        Vs = data[1]
        Vg = data[0]
        mu = (
            alpha_gate * Vc
            - alpha_gate * Vg
            - alpha_source * Vs
            - np.sqrt((alpha_J * Vs) ** 2 + 4 * J ** 2) / 2
        )
        fd = 1 / (1 + np.exp(mu / (kb * T)))
        fs = 1 / (1 + np.exp((mu + Vs) / (kb * T)))
        return prefactor * (fd - fs)