'''
particle_pushers: relativistic charged particle pushers for electromagnetic field simulation.

A Python package implementing various explicit and implicit numerical
integrators for tracking relativistic charged test particles in static
and time-dependent electromagnetic fields. All quantities are in natural
units where c = 1. All pushers are second-order accurate, with lab-frame
methods integrating in lab time and comoving-frame methods integrating
in proper time.

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

Functions
---------
lorentz_gamma
    Lorentz gamma factor for a relativistic velocity vector.

Lab-frame pushers
-----------------
Boris
    Boris leapfrog method.
Vay
    Vay leapfrog method.
Higuera
    Higuera-Cary leapfrog method.
Lapenta
    Lapenta-Markidis implicit method.
DiscreteGradient
    Discrete gradient implicit method with exact energy conservation.

Comoving-frame pushers
----------------------
GordonExact
    Gordon-Hafizi exact unitary method.
GordonQuadratic
    Gordon-Hafizi quadratic unitary method.
HairerExplicit
    Hairer-Lubich-Shi explicit leapfrog method.
HairerDiscreteGradient
    Hairer-Lubich-Shi implicit discrete gradient method.
HairerVariational
    Hairer-Lubich-Shi variational leapfrog method.
'''

from .field import Field, StaticField, TimeDependentField, SuperposedField
from .particle import Particle
from .lorentz import lorentz_gamma
from .lab_frame import Boris, Vay, Higuera, Lapenta, DiscreteGradient
from .comoving_frame import (
    GordonExact, GordonQuadratic,
    HairerExplicit, HairerDiscreteGradient, HairerVariational
)
