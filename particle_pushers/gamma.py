'''
Lorentz factor calculation for relativistic particle pushing.

All quantities are in natural units where c = 1.
'''

import numpy as np


def Gamma(u):
    '''
    Lorentz gamma factor for a relativistic 3-velocity vector.

    In natural units where c = 1 the Lorentz factor is:

        gamma = sqrt(1 + |u|^2)

    Parameters
    ----------
    u : array_like
        Relativistic 3-velocity vector, shape (3,).

    Returns
    -------
    float
        Lorentz gamma factor corresponding to velocity u.

    Examples
    --------
    Particle at rest:

    >>> Gamma([0., 0., 0.])
    1.0

    Highly relativistic particle:

    >>> Gamma([1., 0., 0.])
    1.4142135623730951
    '''
    return np.sqrt(1 + np.linalg.norm(u)**2)
