'''
particle_pushers — Relativistic charged particle pushers for electromagnetic field simulation.

A Python package implementing a range of explicit, implicit and variational
integrators for tracking relativistic charged test particles in static and
time-dependent electromagnetic fields. All quantities are in natural units
where c = 1.

Classes
-------
Field
    Abstract base class for electromagnetic fields.
StaticField
    Electromagnetic field with no explicit time dependence.
TimeDependentField
    Electromagnetic field with explicit time dependence.
SuperposedField
    Superposition of multiple electromagnetic fields.
Particle
    Relativistic charged test particle.

Lab-frame pushers
-----------------
Boris, BorisFourthOrder, BorisAdaptiveFourthOrder, BorisAdaptiveSubstep
    Boris leapfrog method and extensions.
Vay, VayFourthOrder, VayAdaptiveFourthOrder, VayAdaptiveSubstep
    Vay method and extensions.
Higuera, HigueraFourthOrder, HigueraAdaptiveFourthOrder, HigueraAdaptiveSubstep
    Higuera-Cary method and extensions.
Lapenta
    Lapenta-Markidis implicit method.
Petri
    Pétri implicit method.
DiscreteGradient
    Discrete gradient implicit method with exact energy conservation.

Comoving-frame pushers
----------------------
GordonHafiziQuadratic, GordonHafiziQuadraticFourthOrder
    Gordon-Hafizi quadratic spinor method and fourth-order extension.
GordonHafiziExact, GordonHafiziExactFourthOrder
    Gordon-Hafizi exact spinor method and fourth-order extension.
HairerExplicit
    Hairer-Lubich-Shi explicit leapfrog method.
HairerDiscreteGradient
    Hairer-Lubich-Shi implicit discrete gradient leapfrog method.
HairerVariational
    Hairer-Lubich-Shi implicit variational leapfrog method.
'''

from .field import StaticField, TimeDependentField, SuperposedField
from .particle import Particle
from .lab_frame import (
    Boris, BorisFourthOrder, BorisAdaptiveFourthOrder, BorisAdaptiveSubstep,
    Vay, VayFourthOrder, VayAdaptiveFourthOrder, VayAdaptiveSubstep,
    Higuera, HigueraFourthOrder, HigueraAdaptiveFourthOrder, HigueraAdaptiveSubstep,
    Lapenta, Petri, DiscreteGradient
)
from .comoving_frame import (
    GordonHafiziExact, GordonHafiziExactFourthOrder,
    GordonHafiziQuadratic, GordonHafiziQuadraticFourthOrder,
    HairerExplicit, HairerVariational, HairerDiscreteGradient
)
