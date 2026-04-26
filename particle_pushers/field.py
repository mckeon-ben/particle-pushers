'''
Electromagnetic field classes for relativistic charged particle tracking.

Provides a base Field class and concrete implementations for static,
time-dependent, and superposed electromagnetic fields. Fields are
defined in terms of the electric and magnetic field vectors, scalar
and vector potentials, and their derivatives. All quantities are
assumed to be in natural units where c = 1.
'''

import numpy as np


class Field:
    '''
    Abstract base class for electromagnetic fields.

    Defines the interface for all field classes. All methods return
    zero by default, allowing subclasses to override only the
    quantities they define. All quantities are in natural units
    where c = 1.

    All spatial quantities are 3-vectors and all temporal quantities
    are scalars unless otherwise stated.
    '''

    def E(self, x, t) -> np.ndarray:
        '''
        Electric field vector.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        np.ndarray
            Electric field vector at (x, t), shape (3,).
            Returns zeros by default.
        '''
        return np.zeros(3)

    def B(self, x, t) -> np.ndarray:
        '''
        Magnetic field vector.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        np.ndarray
            Magnetic field vector at (x, t), shape (3,).
            Returns zeros by default.
        '''
        return np.zeros(3)

    def phi(self, x, t) -> float:
        '''
        Electric scalar potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        float
            Scalar potential at (x, t).
            Returns zero by default.
        '''
        return 0.0

    def A(self, x, t) -> np.ndarray:
        '''
        Magnetic vector potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        np.ndarray
            Vector potential at (x, t), shape (3,).
            Returns zeros by default.
        '''
        return np.zeros(3)

    def phi_t(self, x, t) -> float:
        '''
        Partial time derivative of the scalar potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        float
            Partial time derivative of the scalar potential at (x, t).
            Returns zero by default.
        '''
        return 0.0

    def A_t(self, x, t) -> np.ndarray:
        '''
        Partial time derivative of the vector potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        np.ndarray
            Partial time derivative of the vector potential at (x, t),
            shape (3,). Returns zeros by default.
        '''
        return np.zeros(3)

    def A_x(self, x, t) -> np.ndarray:
        '''
        Spatial Jacobian of the vector potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        np.ndarray
            Spatial Jacobian of the vector potential at (x, t),
            shape (3, 3), where element (i, j) is dA_i/dx_j.
            Returns zeros by default.
        '''
        return np.zeros((3, 3))


class StaticField(Field):
    '''
    Electromagnetic field with no explicit time dependence.

    All field quantities are functions of position only. The partial
    time derivatives of the scalar and vector potentials are identically
    zero by definition.

    Parameters
    ----------
    E_func : callable, optional
        Electric field function with signature E(x) -> array_like, shape (3,).
    B_func : callable, optional
        Magnetic field function with signature B(x) -> array_like, shape (3,).
    phi_func : callable, optional
        Scalar potential function with signature phi(x) -> float.
    A_func : callable, optional
        Vector potential function with signature A(x) -> array_like, shape (3,).
    A_x_func : callable, optional
        Spatial Jacobian of the vector potential with signature
        A_x(x) -> array_like, shape (3, 3).

    Examples
    --------
    Uniform magnetic field in the z-direction:

    >>> B_func = lambda x: [0.0, 0.0, 1.0]
    >>> field = StaticField(B_func=B_func)
    '''

    def __init__(self, E_func=None, B_func=None, phi_func=None,
                 A_func=None, A_x_func=None):
        self._E = E_func
        self._B = B_func
        self._phi = phi_func
        self._A = A_func
        self._A_x = A_x_func

    def E(self, x, t):
        return np.asarray(self._E(x), dtype=float) if self._E is not None else super().E(x, t)

    def B(self, x, t):
        return np.asarray(self._B(x), dtype=float) if self._B is not None else super().B(x, t)

    def phi(self, x, t):
        return np.asarray(self._phi(x), dtype=float) if self._phi is not None else 0.0

    def A(self, x, t):
        return np.asarray(self._A(x), dtype=float) if self._A is not None else np.zeros(3)

    def phi_t(self, x, t):
        return 0.0

    def A_t(self, x, t):
        return np.zeros(3)

    def A_x(self, x, t):
        if self._A_x is not None:
            return np.asarray(self._A_x(x), dtype=float)
        return np.zeros((3, 3))


class TimeDependentField(Field):
    '''
    Electromagnetic field with explicit time dependence.

    All field quantities are functions of both position and time.
    The partial time derivatives of the scalar and vector potentials
    must be supplied explicitly if required by the integrator.

    Parameters
    ----------
    E_func : callable, optional
        Electric field function with signature E(x, t) -> array_like, shape (3,).
    B_func : callable, optional
        Magnetic field function with signature B(x, t) -> array_like, shape (3,).
    phi_func : callable, optional
        Scalar potential function with signature phi(x, t) -> float.
    A_func : callable, optional
        Vector potential function with signature A(x, t) -> array_like, shape (3,).
    phi_t_func : callable, optional
        Partial time derivative of the scalar potential with signature
        phi_t(x, t) -> float.
    A_t_func : callable, optional
        Partial time derivative of the vector potential with signature
        A_t(x, t) -> array_like, shape (3,).
    A_x_func : callable, optional
        Spatial Jacobian of the vector potential with signature
        A_x(x, t) -> array_like, shape (3, 3).

    Examples
    --------
    Plane wave propagating in the z-direction, polarised in x:

    >>> omega, k, E0 = 1.0, 2.0, 0.1
    >>> E_func = lambda x, t: [E0 * np.cos(omega * t - k * x[2]), 0., 0.]
    >>> B_func = lambda x, t: [0., E0 * np.cos(omega * t - k * x[2]), 0.]
    >>> field = TimeDependentField(E_func=E_func, B_func=B_func)
    '''

    def __init__(self, E_func=None, B_func=None, phi_func=None,
                 A_func=None, phi_t_func=None, A_t_func=None, A_x_func=None):
        self._E = E_func
        self._B = B_func
        self._phi = phi_func
        self._A = A_func
        self._phi_t = phi_t_func
        self._A_t = A_t_func
        self._A_x = A_x_func

    def E(self, x, t):
        return np.asarray(self._E(x, t), dtype=float) if self._E is not None else super().E(x, t)

    def B(self, x, t):
        return np.asarray(self._B(x, t), dtype=float) if self._B is not None else super().B(x, t)

    def phi(self, x, t):
        return np.asarray(self._phi(x, t), dtype=float) if self._phi is not None else 0.0

    def A(self, x, t):
        return np.asarray(self._A(x, t), dtype=float) if self._A is not None else np.zeros(3)

    def phi_t(self, x, t):
        return np.asarray(self._phi_t(x, t), dtype=float) if self._phi_t is not None else 0.0

    def A_t(self, x, t):
        return np.asarray(self._A_t(x, t), dtype=float) if self._A_t is not None else np.zeros(3)

    def A_x(self, x, t):
        return np.asarray(self._A_x(x, t), dtype=float) if self._A_x is not None else np.zeros((3, 3))


class SuperposedField(Field):
    '''
    Superposition of multiple electromagnetic fields.

    Combines any number of Field objects by summing their contributions.
    Useful for representing a background field combined with a
    perturbation, or multiple wave modes simultaneously.

    Parameters
    ----------
    *fields : Field
        Any number of Field objects to superpose.

    Examples
    --------
    Background magnetic field plus a plane wave:

    >>> background = StaticField(B_func=lambda x: [0., 0., 1.])
    >>> wave = TimeDependentField(E_func=E_func, B_func=B_func)
    >>> field = SuperposedField(background, wave)
    '''

    def __init__(self, *fields):
        self.fields = fields

    def E(self, x, t): return sum(f.E(x, t) for f in self.fields)
    def B(self, x, t): return sum(f.B(x, t) for f in self.fields)
    def phi(self, x, t): return sum(f.phi(x, t) for f in self.fields)
    def A(self, x, t): return sum(f.A(x, t) for f in self.fields)
    def phi_t(self, x, t): return sum(f.phi_t(x, t) for f in self.fields)
    def A_t(self, x, t): return sum(f.A_t(x, t) for f in self.fields)
    def A_x(self, x, t): return sum(f.A_x(x, t) for f in self.fields)
