import numpy as np

from transport_data_analysis import QTLabData, StabilityDiagram
from transport_data_analysis.qtlab_data import QTLab_Data


def test_qtlab_header_sets_axes_and_metadata(tmp_path):
    path = tmp_path / "measurement.dat"
    path.write_text(
        "# QTLab export\n"
        "# Column 1:\n"
        "# name: Vg\n"
        "# Column 2:\n"
        "# name: Isd\n"
        "# temperature: 77\n"
        "0\t1e-9\n"
        "1\t2e-9\n",
        encoding="utf-8",
    )

    data = QTLabData.load_from_file(path)

    assert data.axes == ["Vg", "Isd"]
    assert data.ps("temperature") == {"temperature": 77.0}
    np.testing.assert_array_equal(data["Vg"].values, [0, 1])
    assert QTLab_Data is QTLabData


def test_stability_diagram_reshapes_a_flat_gate_sweep():
    diagram = StabilityDiagram(
        {
            "Vg": np.array([0.0, 1.0, 0.0, 1.0]),
            "Vsd": np.array([-0.1, -0.1, 0.1, 0.1]),
            "Isd": np.array([-1e-9, -2e-9, 1e-9, 2e-9]),
        }
    )

    assert diagram["Vg"].values.shape == (2, 2)
    assert diagram.axes == ("Vg", "Vsd", "Isd")
    np.testing.assert_array_equal(diagram._data["n"], [[0, 0], [1, 1]])
