import numpy as np

from transport_data_analysis import PhysicsModels, SpectralDensityModel


def test_legacy_model_regression_values(capsys):
    bias = np.array([-0.1, 0.0, 0.1])

    current = PhysicsModels.marcus(bias, 0.2, 0.01, 0.02, 0.5, 77)
    thermal = PhysicsModels.thermal_broadening(
        np.array([-1.0, 0.0, 1.0]), 77, 0, 1, 0.1
    )
    lifetime = PhysicsModels.lifetime_broadening(
        np.array([-1.0, 0.0, 1.0]), 0, 0.01, 0.1
    )

    np.testing.assert_allclose(
        current, [-3.12521138885096e-10, 0.0, 4.685733544223023e-10]
    )
    np.testing.assert_allclose(
        thermal, [1.1399636926680236e-06, 1.0, 1.1399636926680236e-06]
    )
    np.testing.assert_allclose(
        lifetime,
        [7.728769806483788e-07, 0.00030992366924, 7.728769806483788e-07],
    )
    assert capsys.readouterr().out == ""


def test_spectral_density_model_supports_small_custom_grids():
    model = SpectralDensityModel(
        frequencies=np.linspace(1e-4, 0.03, 32),
        energies=np.linspace(-0.05, 0.05, 21),
        times=np.linspace(0, 100, 40),
    )
    model.add_mode(omega=0.01)
    model.calculate_rate_constants(Y=0.001, T=77)

    assert model.k_ox.shape == (21,)
    assert model.k_red.shape == (21,)
    assert np.isfinite(model.k_ox).all()
    assert np.isfinite(model.k_red).all()
