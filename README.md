# Kepler — Interactive Newtonian Orbit Laboratory

Kepler is a notebook-first, interactive two-body gravity simulator. It uses
Newtonian mechanics and a velocity-Verlet integrator to explore bound orbits,
escape trajectories, close encounters, and collisions.

## What is included

- Interactive controls for masses, positions, velocities, time step, duration,
  radii, and gravitational constant.
- Saved presets for circular, elliptical, near-escape, escape, and collision
  scenarios.
- Animated trajectories with play/pause controls and a time slider.
- Position, velocity, separation, energy, and angular-momentum diagnostics.
- CSV, JSON, interactive HTML, and (when Kaleido supports it) PNG exports.
- A standalone HTML report containing the trajectory and diagnostic figures.
- Conservation-error summaries to help judge numerical accuracy.

## Run locally

GitHub can display the saved notebook, but it cannot execute its Python kernel or
widgets. Clone the repository and create the isolated environment:

```bash
git clone git@github.com:MLee-Sinphy/Kepler.git
cd Kepler
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open `notebooks/kepler_orbit_lab.ipynb`, choose **Run All**, select a preset or
change its controls, and press **Run simulation**.

Conda users can instead run:

```bash
conda env create -f environment.yml
conda activate kepler-orbits
jupyter lab
```

## Numerical model

Both bodies are integrated in an inertial barycentric frame:

```text
a1 = G m2 (r2 - r1) / |r2 - r1|^3
a2 = G m1 (r1 - r2) / |r2 - r1|^3
```

The velocity-Verlet method is symplectic and generally preserves orbital energy
far better than forward Euler. Collision detection stops the integration when
the center-to-center distance is no larger than the sum of the object radii.
Escape classification uses positive relative specific orbital energy together
with outward radial motion.

The default units are normalized (`G = 1`). Any consistent unit system can be
used by entering the corresponding value of `G`.

## Project layout

```text
notebooks/kepler_orbit_lab.ipynb  Interactive laboratory
src/orbit.py                      Physics, diagnostics, plotting, exports
tests/test_orbit.py               Numerical and classification checks
environment.yml                   Conda environment
requirements.txt                  venv/pip environment
exports/                          Generated files (ignored by Git)
```

## Future web version

The notebook architecture deliberately separates physics from interface code.
It can later be exposed as a Voilà, Panel, or Dash application without rewriting
the simulation engine.
