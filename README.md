# particle-pushers

A Python package implementing a suite of numerical integrators for tracking relativistic charged test particles in static and time-dependent electromagnetic fields. All quantities are in natural units where *c* = 1.

## Requirements

- Python 3.9+
- NumPy
- SciPy

## Installation

```bash
git clone https://github.com/mckeon-ben/particle-pushers.git
cd particle-pushers
pip install .
```

## Quick Start

### Lab-frame pusher

```python
import numpy as np
from particle_pushers import Boris, StaticField, Particle

# Define a uniform magnetic field in the z-direction.
field = StaticField(B_func=lambda x: np.array([0., 0., 1.]))

# Initialise a particle.
particle = Particle(
    x=np.array([1., 0., 0.]),
    u=np.array([0., 0.5, 0.]),
    q=1., m=1.
)

# Run the Boris pusher.
sim = Boris(particle, field)
t, x, u = sim.solve(t_span=(0., 20 * np.pi), N=1000)
```

### Comoving-frame pusher

Comoving-frame pushers use 4-vectors for position and velocity. The
zeroth component of the 4-position is the coordinate time *t*, and
the zeroth component of the 4-velocity is the Lorentz factor *γ*.

```python
import numpy as np
from particle_pushers import GordonExact, StaticField, Particle
from particle_pushers import lorentz_gamma

# Define a uniform magnetic field in the z-direction.
field = StaticField(B_func=lambda x: np.array([0., 0., 1.]))

# 3-position and 3-velocity.
x3 = np.array([1., 0., 0.])
u3 = np.array([0., 0.5, 0.])

# Construct 4-vectors: [t, x, y, z] and [gamma, u_x, u_y, u_z].
x0 = np.array([0., x3[0], x3[1], x3[2]])
u0 = np.array([lorentz_gamma(u3), u3[0], u3[1], u3[2]])

# Initialise a particle with 4-vectors.
particle = Particle(x=x0, u=u0, q=1., m=1.)

# Run the Gordon-Hafizi exact pusher in proper time.
sim = GordonExact(particle, field)
tau, x, u = sim.solve(t_span=(0., 20 * np.pi), N=1000)
```

## Available Pushers

All pushers are second-order accurate, with lab-frame methods advancing
in lab time and comoving-frame methods advancing in proper time.

### Lab frame

**Explicit**

| Method | Class |
|:---|:---|
| Boris | `Boris` |
| Vay | `Vay` |
| Higuera-Cary | `Higuera` |

**Implicit**

| Method | Class |
|:---|:---|
| Lapenta-Markidis | `Lapenta` |
| Discrete gradient | `DiscreteGradient` |

### Comoving frame

**Gordon-Hafizi**

| Method | Class |
|:---|:---|
| Exact | `GordonExact` |
| Quadratic | `GordonQuadratic` |

**Hairer-Lubich-Shi**

| Method | Class |
|:---|:---|
| Explicit | `HairerExplicit` |
| Discrete gradient | `HairerDiscreteGradient` |
| Variational | `HairerVariational` |

## Field Classes

| Class | Description |
|:---|:---|
| `StaticField` | Position-dependent fields with no time dependence |
| `TimeDependentField` | Fields depending on both position and time |
| `SuperposedField` | Sum of multiple field objects |

## Utilities

| Name | Description |
|:---|:---|
| `Particle` | Relativistic charged test particle |
| `lorentz_gamma` | Lorentz factor *γ* for a relativistic velocity vector |

## References

- Boris, J.P., 1970. Relativistic Plasma Simulation — Optimization of a Hybrid Code. In *Proc. Fourth Conf. Num. Sim. Plasmas* (pp. 3-67).
- Vay, J.L., 2008. Simulation of beams or plasmas crossing at relativistic velocity. *Physics of Plasmas, 15*(5).
- Higuera, A.V. and Cary, J.R., 2017. Structure-preserving second-order integration of relativistic charged particle trajectories in electromagnetic fields. *Physics of Plasmas, 24*(5).
- Lapenta, G. and Markidis, S., 2011. Particle acceleration and energy conservation in particle in cell simulations. *Physics of Plasmas, 18*(7).
- Gonzalez, O., 1996. Time integration and discrete Hamiltonian systems. *Journal of Nonlinear Science, 6*(5), pp.449-467.
- Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher for extreme fields. *Computer Physics Communications, 258*, p.107628.
- Hairer, E., Lubich, C. and Shi, Y., 2023. Leapfrog methods for relativistic charged-particle dynamics. *SIAM Journal on Numerical Analysis, 61*(6), pp.2844-2858.
