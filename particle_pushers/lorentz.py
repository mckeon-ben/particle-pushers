'''
Lorentz factor calculation for relativistic particle pushing.

All quantities are in natural units where c = 1.
'''

import numpy as np


def lorentz_gamma(u):
    '''
    Lorentz gamma factor for a relativistic velocity vector.

    In natural units where c = 1 the Lorentz factor is::

        gamma = sqrt(1 + |u|^2)

    Parameters
    ----------
    u : array_like
        Relativistic velocity vector. Typically shape (3,) for
        lab-frame pushers, but any 1D array is accepted.

    Returns
    -------
    np.float64
        Lorentz gamma factor corresponding to velocity u.

    Examples
    --------
    Particle at rest:

    >>> float(lorentz_gamma([0., 0., 0.]))
    1.0

    Highly relativistic particle:

    >>> float(lorentz_gamma([1., 0., 0.]))
    1.4142135623730951
    '''
    u = np.asarray(u)
    return np.sqrt(1 + np.dot(u, u))
