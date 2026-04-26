'''
Particle dataclass for relativistic charged particle tracking.

All quantities are assumed to be in natural units where c = 1.
'''

import numpy as np
from dataclasses import dataclass


@dataclass
class Particle:
    '''
    Represents a relativistic charged test particle.

    Stores the particle's kinematic state and physical properties.
    The position and velocity may be either 3-vectors for lab-frame
    pushers or 4-vectors for comoving-frame pushers. All quantities
    are in natural units where c = 1.

    Attributes
    ----------
    x : np.ndarray
        Particle position vector. Shape (3,) for lab-frame pushers
        or (4,) for comoving-frame pushers, where the zeroth
        component is the coordinate time t.
    u : np.ndarray
        Particle relativistic velocity vector. Shape (3,) for
        lab-frame pushers or (4,) for comoving-frame pushers,
        where the zeroth component is gamma.
    q : float
        Particle charge.
    m : float
        Particle mass.

    Examples
    --------
    Lab-frame particle at rest:

    >>> p = Particle(x=[1., 0., 0.],
    ...              u=[0., 0., 0.],
    ...              q=1., m=1.)

    Comoving-frame particle at rest:

    >>> p = Particle(x=[0., 1., 0., 0.],
    ...              u=[1., 0., 0., 0.],
    ...              q=1., m=1.)
    '''
    x: np.ndarray
    u: np.ndarray
    q: float
    m: float
