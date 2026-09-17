# particle-pushers

Numerical integrators for relativistic charged test particles.

`particle-pushers` tracks relativistic charged test particles through
static and time-dependent electromagnetic fields. It provides lab-frame
pushers, which advance 3-vectors in lab time, and comoving-frame
pushers, which advance 4-vectors in proper time or, through a symmetric
time conversion, in lab time. The base methods are second-order
accurate, and fourth-order variants are available by Yoshida
composition. Scripts for seven test fields compare the methods. All
quantities are in natural units with *c* = 1.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Particles and fields](#particles-and-fields)
- [Pushers](#pushers)
- [Lab time and proper time](#lab-time-and-proper-time)
- [Things to know](#things-to-know)
- [Examples](#examples)
- [Package layout](#package-layout)
- [Licence](#licence)
- [References](#references)

## Installation

`particle-pushers` needs Python 3.9 or later, NumPy and SciPy:

```bash
git clone https://github.com/mckeon-ben/particle-pushers.git
cd particle-pushers
pip install .
```

Use `pip install -e .` instead to work on the code in place. The
scripts in `examples/` also need matplotlib and a LaTeX installation;
see [Examples](#examples).

## Quick start

### Lab-frame pusher

```python
import numpy as np
from particle_pushers import Boris, StaticField, Particle

# Uniform magnetic field along z.
field = StaticField(B_func=lambda x: np.array([0., 0., 1.]))

# Lab-frame pushers take 3-vectors.
particle = Particle(
    x=np.array([1., 0., 0.]),
    u=np.array([0., 0.5, 0.]),
    q=1., m=1.
)

sim = Boris(particle, field)
t, x, u = sim.solve((0., 20 * np.pi), N=1000)
```

For fourth order, replace `Boris` with `BorisOrderFour`; nothing else
changes.

### Comoving-frame pusher

Comoving-frame pushers take 4-vectors. The zeroth component of the
4-position is the coordinate time *t*, and the zeroth component of the
4-velocity is the Lorentz factor *γ*.

```python
import numpy as np
from particle_pushers import GordonExact, StaticField, Particle
from particle_pushers import lorentz_gamma

field = StaticField(B_func=lambda x: np.array([0., 0., 1.]))

# 4-vectors [t, x, y, z] and [gamma, u_x, u_y, u_z].
x3 = np.array([1., 0., 0.])
u3 = np.array([0., 0.5, 0.])
x0 = np.array([0., *x3])
u0 = np.array([lorentz_gamma(u3), *u3])

particle = Particle(x=x0, u=u0, q=1., m=1.)

# Steps are in proper time: tau runs over [0, 20 pi].
sim = GordonExact(particle, field)
tau, x, u = sim.solve((0., 20 * np.pi), N=1000)
```

To step in lab time instead, replace `GordonExact` with
`GordonExactLab`. The call is the same, but `solve` then takes N equal
lab-time steps over the interval, so the result can be compared
directly with a lab-frame pusher at the same step size. For fourth
order, use `GordonExactOrderFour` or `GordonExactLabOrderFour`.

## Particles and fields

A `Particle` holds a position `x`, a velocity `u`, a charge `q` and a
mass `m`. The velocity is the spatial part of the 4-velocity,
`u = gamma v`, not *v* itself. Lab-frame pushers use 3-vectors;
comoving-frame pushers use 4-vectors, as in the quick start.
`lorentz_gamma(u)` returns the Lorentz factor, `sqrt(1 + |u|^2)`.

Fields are built from functions of position (`StaticField`) or of
position and time (`TimeDependentField`):

- `E_func` and `B_func`, the electric and magnetic fields, which every
  pusher uses;
- `phi_func`, the scalar potential, needed by `DiscreteGradient` and
  `HairerDiscreteGradient`;
- `A_func`, `A_x_func` and, for time-dependent fields, `phi_t_func`
  and `A_t_func`: the vector potential, its Jacobian and the time
  derivatives, needed by `HairerVariational`.

The base class `Field` is the identically zero field. Every quantity a
field is not given also returns zero; see
[Things to know](#things-to-know).

## Pushers

Every pusher is constructed as `Pusher(particle, field)` and run with
`solve(t_span, N)`, which returns the time grid, positions and
velocities.

### Lab frame

3-vectors, stepped in lab time.

| Method            | 2nd order          | 4th order          | Scheme   |
| ----------------- | ------------------ | ------------------ | -------- |
| Boris             | `Boris`            | `BorisOrderFour`   | explicit |
| Vay               | `Vay`              | `VayOrderFour`     | explicit |
| Higuera–Cary      | `Higuera`          | `HigueraOrderFour` | explicit |
| Lapenta–Markidis  | `Lapenta`          | n/a                | implicit |
| Discrete gradient | `DiscreteGradient` | n/a                | implicit |

`DiscreteGradient` conserves the energy `gamma m + q phi` exactly for
static fields.

### Comoving frame: Gordon–Hafizi

4-vectors, stepped in proper time or, for the `Lab` classes, in lab
time. All four operators are explicit.

| Operator  | Time   | 2nd order            | 4th order                     |
| --------- | ------ | -------------------- | ----------------------------- |
| Exact     | proper | `GordonExact`        | `GordonExactOrderFour`        |
| Exact     | lab    | `GordonExactLab`     | `GordonExactLabOrderFour`     |
| Quadratic | proper | `GordonQuadratic`    | `GordonQuadraticOrderFour`    |
| Quadratic | lab    | `GordonQuadraticLab` | `GordonQuadraticLabOrderFour` |

The exact operator solves the equations of motion exactly in a locally
constant field. The quadratic operator is a rational approximation
that preserves unit determinant and is exact for null fields.

### Comoving frame: Hairer–Lubich–Shi

4-vectors, stepped in proper time, with velocities on a staggered grid.
All three are second order.

| Method            | Class                    | Scheme   |
| ----------------- | ------------------------ | -------- |
| Explicit leapfrog | `HairerExplicit`         | explicit |
| Discrete gradient | `HairerDiscreteGradient` | implicit |
| Variational       | `HairerVariational`      | implicit |

`HairerExplicit` and `HairerDiscreteGradient` preserve the mass shell
`u^mu u_mu = -1` exactly; `HairerDiscreteGradient` also conserves
`gamma m + q phi` exactly for static fields.

## Lab time and proper time

Lab-frame pushers advance in lab time *t*; comoving-frame pushers
advance in proper time *τ*, and lab time accumulates in the zeroth
component of the 4-position. Comparing the two at a common step size
therefore needs the comoving-frame methods to take controlled lab-time
steps.

The `Lab` variants of the Gordon–Hafizi pushers do this. For each lab
step `dt` they solve the trapezoidal relation

```text
dt = dtau (gamma_n + gamma_{n+1}) / 2
```

for the proper-time step `dtau` by fixed-point iteration. The relation
is time-symmetric, so the lab-time step keeps the second-order,
even-power error structure of the underlying proper-time method.

The fourth-order classes compose a symmetric second-order step three
times with Yoshida's triple-jump coefficients. This needs a
time-symmetric base step, which the explicit lab-frame methods and the
Gordon–Hafizi methods provide, in both proper and lab time.

## Things to know

- `solve` updates `particle` in place. Calling it again continues from
  the final state, not the initial one.
- A field returns zero for any quantity it was not given. A pusher that
  needs `phi` or the vector potential therefore runs without error on a
  field that lacks them, but integrates the wrong problem.
- The discrete gradient methods conserve energy exactly only for static
  fields.
- The Hairer–Lubich–Shi pushers return velocities at half-integer
  steps; see the `solve` docstring of those classes for the shapes.
- `t_span` must be increasing, and `N` a positive integer.

## Examples

The scripts in `examples/` integrate the pushers over a fixed lab time
at a sequence of step counts, second and fourth order, and write the
final states to JSON:

| Script                  | Field                                           |
| ----------------------- | ----------------------------------------------- |
| `charged_column.py`     | Radially growing axial field plus a line charge |
| `coulomb_scattering.py` | Coulomb field of a fixed point charge           |
| `electrode_array.py`    | Vacuum field above a periodic electrode plane   |
| `harmonic_well.py`      | Harmonic electrostatic well in an axial field   |
| `magnetic_mirror.py`    | Axisymmetric magnetic mirror                    |
| `planar_undulator.py`   | Planar undulator, an exact vacuum field         |
| `plane_wave.py`         | Linearly polarised monochromatic plane wave     |

`plotting.py` turns the data files into error estimates, observed
orders and convergence figures. The scripts can be run from any
directory: data files always go to `examples/data/` and figures to
`examples/plots/`.

```bash
python examples/magnetic_mirror.py
python examples/plotting.py                    # every data file
python examples/plotting.py magnetic_mirror    # one data file
```

The full step sequences make each script take a while to run.

`plotting.py` needs matplotlib and, by default, a LaTeX installation
with the `helvet` and `sansmath` packages, since it typesets through
LaTeX. Set `USETEX = False` at the top of the script to use
matplotlib's own renderer instead.

## Package layout

```text
pyproject.toml
README.md
LICENSE
particle_pushers/
    __init__.py            public API
    particle.py            Particle
    field.py               Field, StaticField, TimeDependentField
    lorentz.py             lorentz_gamma
    pusher.py              Pusher, PusherOrderFour (base classes)
    lab_frame/
        boris.py           Boris, BorisOrderFour
        vay.py             Vay, VayOrderFour
        higuera.py         Higuera, HigueraOrderFour
        lapenta.py         Lapenta
        discrete_gradient.py
                           DiscreteGradient
    comoving_frame/
        gordon.py          Gordon-Hafizi pushers, proper and lab time
        hairer.py          Hairer-Lubich-Shi pushers
examples/
    <test field>.py        seven simulation scripts
    plotting.py            error estimates and figures
```

## Licence

MIT; see [LICENSE](LICENSE).

## References

- Boris, J.P., 1970. Relativistic Plasma Simulation — Optimization of a
  Hybrid Code. In *Proc. Fourth Conf. Num. Sim. Plasmas* (pp. 3-67).
- Gonzalez, O., 1996. Time integration and discrete Hamiltonian systems.
  *Journal of Nonlinear Science, 6*(5), pp.449-467.
- Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher for
  extreme fields. *Computer Physics Communications, 258*, p.107628.
- Hairer, E., Lubich, C. and Shi, Y., 2023. Leapfrog methods for
  relativistic charged-particle dynamics. *SIAM Journal on Numerical
  Analysis, 61*(6), pp.2844-2858.
- Higuera, A.V. and Cary, J.R., 2017. Structure-preserving second-order
  integration of relativistic charged particle trajectories in
  electromagnetic fields. *Physics of Plasmas, 24*(5).
- Lapenta, G. and Markidis, S., 2011. Particle acceleration and energy
  conservation in particle in cell simulations. *Physics of Plasmas,
  18*(7).
- Vay, J.L., 2008. Simulation of beams or plasmas crossing at
  relativistic velocity. *Physics of Plasmas, 15*(5).
- Yoshida, H., 1990. Construction of higher order symplectic
  integrators. *Physics Letters A, 150*(5-7), pp.262-268.
