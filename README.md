# Transport Data Analysis

Small Python toolkit for molecular-junction transport research. It provides:

- array-backed containers for tabular and gridded measurements;
- QTLab and ADWin file loading and stability-diagram helpers;
- plotting helpers for current, conductance, and gate/bias maps;
- Marcus, Marcus-Jortner, spectral-density, and broadening models;
- threshold-based peak detection.

The package uses a `src` layout and keeps acquisition-specific code separate
from the reusable data and physics layers.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the optional global-fitting and development dependencies when needed:

```bash
python -m pip install -e ".[fit,dev]"
```

## Load and transform data

```python
from transport_data_analysis import QTLabData

data = QTLabData.load_from_file("measurement.dat")

# Select one column, derive current with respect to bias, and smooth a copy.
conductance = data["Isd"].derive(x=data["Vsd"], method="savgol")
smoothed = conductance.smooth(method="gaussian2")
```

`Data` stores equally shaped NumPy arrays by column name. `data["Isd"]`
returns a one-column `Data` object; use `.values` to obtain its NumPy array.
Array masks select rows, while transformation methods return independent copies.

## Work with a stability diagram

```python
from transport_data_analysis import StabilityDiagram
from transport_data_analysis.qtlab import Subplot_GVsVg
from transport_data_analysis import Figure

diagram = StabilityDiagram.load_from_file("stability_diagram.dat")
zero_bias = diagram.zero_bias_gate_trace(zero=0.003)

figure = Figure()
figure.add_subplot(Subplot_GVsVg(diagram))
figure.visualise()
```

QTLab files are expected to contain `# Column ... name: ...` header records and
tab-separated values. Pass `axes=(...)` to `load_from_file` for plain delimited
files without a QTLab header.

## Evaluate a transport model

```python
import numpy as np
from transport_data_analysis import PhysicsModels

bias = np.linspace(-0.2, 0.2, 401)
current = PhysicsModels.marcus(
    bias,
    lam=0.2,
    VS=0.01,
    VD=0.02,
    alpha_source=0.5,
    T=77,
)
```

Energies and voltages use eV/V, temperature uses kelvin, and model currents are
returned in amperes. The legacy names `physics_models`, `spectral_density_model`,
`Cyclic_Data`, and the old module paths remain available for existing notebooks.

## Develop

```bash
ruff check .
ruff format --check .
pytest
```

Add regression tests whenever a scientific formula or data transformation is
changed. Tests should use small synthetic arrays so numerical intent is clear.
