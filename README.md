# particle-pushers

A Python package implementing a suite of explicit, implicit, and variational integrators for tracking relativistic charged test particles in static and time-dependent electromagnetic fields. All quantities are in natural units where *c* = 1.

## Features

- **Lab-frame pushers** — Boris, Vay, and Higuera-Cary methods, each available in second-order, fourth-order Yoshida, and adaptive variants
- **Implicit lab-frame pushers** — Lapenta-Markidis, Pétri, and discrete gradient methods
- **Comoving-frame pushers** — Gordon-Hafizi exact and quadratic spinor methods, each available in second and fourth order
- **Hairer-Lubich-Shi pushers** — explicit, variational leapfrog, and implicit discrete gradient methods
- **Flexible field interface** — static, time-dependent, and superposed electromagnetic fields with optional scalar and vector potentials

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

### Lab-frame

| Method | Class | Order |
|---|---|---|
| Boris | `Boris` | 2 |
| Boris | `BorisFourthOrder` | 4 |
| Boris (adaptive, fourth-order) | `BorisAdaptiveFourthOrder` | 2/4 |
| Boris (adaptive, sub-cycling) | `BorisAdaptiveSubstep` | 2 |
| Vay | `Vay` | 2 |
| Vay | `VayFourthOrder` | 4 |
| Vay (adaptive, fourth-order) | `VayAdaptiveFourthOrder` | 2/4 |
| Vay (adaptive, sub-cycling) | `VayAdaptiveSubstep` | 2 |
| Higuera-Cary | `Higuera` | 2 |
| Higuera-Cary | `HigueraFourthOrder` | 4 |
| Higuera-Cary (adaptive, fourth-order) | `HigueraAdaptiveFourthOrder` | 2/4 |
| Higuera-Cary (adaptive, sub-cycling) | `HigueraAdaptiveSubstep` | 2 |
| Lapenta-Markidis | `Lapenta` | 2 |
| Pétri | `Petri` | 2 |
| Discrete gradient | `DiscreteGradient` | 2 |

### Comoving-frame

| Method | Class | Order |
|---|---|---|
| Gordon-Hafizi (exact) | `GordonExact` | 2 |
| Gordon-Hafizi (exact) | `GordonExactFourthOrder` | 4 |
| Gordon-Hafizi (quadratic) | `GordonQuadratic` | 2 |
| Gordon-Hafizi (quadratic) | `GordonQuadraticFourthOrder` | 4 |
| Hairer-Lubich-Shi (explicit) | `HairerExplicit` | 2 |
| Hairer-Lubich-Shi (discrete gradient) | `HairerDiscreteGradient` | 2 |
| Hairer-Lubich-Shi (variational) | `HairerVariational` | 2 |

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
- Pétri, J., 2017. A fully implicit numerical integration of the relativistic particle equation of motion. *Journal of Plasma Physics, 83*(2), p.705830206.
- Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher for extreme fields. *Computer Physics Communications, 258*, p.107628.
- Hairer, E., Lubich, C. and Shi, Y., 2023. Leapfrog methods for relativistic charged-particle dynamics. *SIAM Journal on Numerical Analysis, 61*(6), pp.2844-2858.
- Yoshida, H., 1990. Construction of higher order symplectic integrators. *Physics letters A, 150*(5-7), pp.262-268.
