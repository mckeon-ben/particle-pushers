# particle-pushers

A Python package implementing a suite of numerical integrators for tracking relativistic charged test particles in static and time-dependent electromagnetic fields. All quantities are in natural units where *c* = 1.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mckeon-ben/particle-pushers.git
cd particle-pushers
pip install -e .
```

## Quick Start

```python
import numpy as np
from particle_pushers import Boris, StaticField, Particle

# Define a uniform magnetic field in the z-direction.
field = StaticField(B_func=lambda x: np.array([0., 0., 1.]))

# Initialise a particle.
particle = Particle(
    x=np.array([1., 0., 0.]),
    u=np.array([0., 1., 0.]),
    q=1., m=1.
)

# Run the Boris pusher for one gyroperiod.
sim = Boris(particle, field)
t, x, u = sim.solve(t_span=(0, 2 * np.pi), N=1000)
```

## Available Pushers

All numerical methods are second-order accurate.

### Lab frame

**Explicit**

| Method | Class |
|---|---|
| Boris | `Boris` |
| Vay | `Vay` |
| Higuera-Cary | `Higuera` |

**Implicit**

| Method | Class |
|---|---|
| Lapenta-Markidis | `Lapenta` |
| Discrete gradient | `DiscreteGradient` |

### Comoving frame

**Gordon-Hafizi**

| Method | Class |
|---|---|
| Exact | `GordonExact` |
| Quadratic | `GordonQuadratic` |

**Hairer-Lubich-Shi**

| Method | Class |
|---|---|
| Explicit | `HairerExplicit` |
| Variational | `HairerVariational` |
| Discrete gradient | `HairerDiscreteGradient` |

## Field Classes

| Class | Description |
|---|---|
| `StaticField` | Position-dependent fields with no time dependence |
| `TimeDependentField` | Fields depending on both position and time |
| `SuperposedField` | Sum of multiple field objects |

## References

- Boris, J.P., 1970. Relativistic plasma simulation-optimization of a hybrid code. In *Proc. Fourth Conf. Num. Sim. Plasmas* (pp. 3-67).
- Vay, J.L., 2008. Simulation of beams or plasmas crossing at relativistic velocity. *Physics of Plasmas, 15*(5).
- Higuera, A.V. and Cary, J.R., 2017. Structure-preserving second-order integration of relativistic charged particle trajectories in electromagnetic fields. *Physics of Plasmas, 24*(5).
- Lapenta, G. and Markidis, S., 2011. Particle acceleration and energy conservation in particle in cell simulations. *Physics of Plasmas, 18*(7).
- Gonzalez, O., 1996. Time integration and discrete Hamiltonian systems. *Journal of Nonlinear Science, 6*(5), pp.449-467.
- Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher for extreme fields. *Computer Physics Communications, 258*, p.107628.
- Hairer, E., Lubich, C. and Shi, Y., 2023. Leapfrog methods for relativistic charged-particle dynamics. *SIAM Journal on Numerical Analysis, 61*(6), pp.2844-2858.
