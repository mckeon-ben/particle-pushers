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
        where the zeroth component is the Lorentz factor gamma,
        equal to the zeroth component of the 4-velocity u^0 = gamma.
    q : float
        Particle charge.
    m : float
        Particle rest mass. Must be strictly positive.

    Raises
    ------
    ValueError
        If x and u do not have the same shape, or if m is not
        strictly positive.

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

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.u = np.asarray(self.u, dtype=float)
        self.q = float(self.q)
        self.m = float(self.m)
        if self.x.shape != self.u.shape:
            raise ValueError(
                f'x and u must have the same shape, '
                f'got x: {self.x.shape} and u: {self.u.shape}'
            )
        if self.m <= 0:
            raise ValueError(
                f'm must be strictly positive, got {self.m}'
            )
