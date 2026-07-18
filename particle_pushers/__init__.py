'''
particle_pushers: relativistic charged particle pushers for
electromagnetic field simulation.

A Python package implementing various explicit and implicit numerical
integrators for tracking relativistic charged test particles in static
and time-dependent electromagnetic fields. All quantities are in natural
units where c = 1. Lab-frame methods integrate in lab time and
comoving-frame methods integrate in proper time. The base methods are
second-order accurate; fourth-order variants of the explicit lab-frame
methods and the Gordon-Hafizi methods are provided via Yoshida triple-jump
composition.

Classes
-------
Field
    Abstract base class for electromagnetic fields.
StaticField
    Electromagnetic field with no explicit time dependence.
TimeDependentField
    Electromagnetic field with explicit time dependence.
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
BorisOrderFour
    Fourth-order Boris method via Yoshida composition.
VayOrderFour
    Fourth-order Vay method via Yoshida composition.
HigueraOrderFour
    Fourth-order Higuera-Cary method via Yoshida composition.
LargeStepExplicit
    Large-stepsize modified Boris method for guiding-centre motion.

Comoving-frame pushers
----------------------
GordonExact
    Gordon-Hafizi exact unitary method.
GordonQuadratic
    Gordon-Hafizi quadratic unitary method.
GordonExactLab
    Gordon-Hafizi exact unitary method with lab-time conversion.
GordonQuadraticLab
    Gordon-Hafizi quadratic unitary method with lab-time conversion.
GordonExactOrderFour
    Fourth-order Gordon-Hafizi exact method via Yoshida composition.
GordonQuadraticOrderFour
    Fourth-order Gordon-Hafizi quadratic method via Yoshida composition.
GordonExactLabOrderFour
    Fourth-order Gordon-Hafizi exact method with lab-time conversion.
GordonQuadraticLabOrderFour
    Fourth-order Gordon-Hafizi quadratic method with lab-time conversion.
HairerExplicit
    Hairer-Lubich-Shi explicit leapfrog method.
HairerDiscreteGradient
    Hairer-Lubich-Shi implicit discrete gradient method.
HairerVariational
    Hairer-Lubich-Shi variational leapfrog method.
'''

from .field import Field, StaticField, TimeDependentField
from .particle import Particle
from .lorentz import lorentz_gamma
from .lab_frame import (
    Boris, Vay, Higuera, Lapenta, DiscreteGradient,
    BorisOrderFour, VayOrderFour, HigueraOrderFour,
    LargeStepExplicit
)
from .comoving_frame import (
    GordonExact, GordonQuadratic,
    GordonExactLab, GordonQuadraticLab,
    GordonExactOrderFour, GordonQuadraticOrderFour,
    GordonExactLabOrderFour, GordonQuadraticLabOrderFour,
    HairerExplicit, HairerDiscreteGradient, HairerVariational
)

__all__ = [
    'Field',
    'StaticField',
    'TimeDependentField',
    'Particle',
    'lorentz_gamma',
    'Boris',
    'Vay',
    'Higuera',
    'Lapenta',
    'DiscreteGradient',
    'BorisOrderFour',
    'VayOrderFour',
    'HigueraOrderFour',
    'LargeStepExplicit',
    'GordonExact',
    'GordonQuadratic',
    'GordonExactLab',
    'GordonQuadraticLab',
    'GordonExactOrderFour',
    'GordonQuadraticOrderFour',
    'GordonExactLabOrderFour',
    'GordonQuadraticLabOrderFour',
    'HairerExplicit',
    'HairerDiscreteGradient',
    'HairerVariational'
]
