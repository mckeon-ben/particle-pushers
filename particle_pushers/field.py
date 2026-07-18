'''
Electromagnetic field classes for relativistic charged particle tracking.

Provides a base Field class and concrete implementations for static
and time-dependent electromagnetic fields. Fields are defined in terms
of the electric and magnetic field vectors, scalar and vector
potentials, and their derivatives. All quantities are assumed to be in
natural units where c = 1.
'''

import numpy as np


class Field:
    '''
    Base class for electromagnetic fields.

    Defines the interface for all field classes. All methods return
    zero by default, allowing subclasses to override only the
    quantities they define. All quantities are in natural units
    where c = 1.

    All spatial quantities are 3-vectors and all temporal quantities
    are scalars unless otherwise stated.

    Notes
    -----
    This class is concrete and may be instantiated directly, in which
    case it represents an identically zero electromagnetic field. It
    is not an abstract base class: it declares no abstract methods and
    subclasses are free to override any subset of its methods.
    '''

    def E(self, x, t=None) -> np.ndarray:
        '''
        Electric field vector.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float, optional
            Lab time. Ignored for static fields.

        Returns
        -------
        np.ndarray
            Electric field vector at (x, t), shape (3,).
            Returns zeros by default.
        '''
        return np.zeros(3)

    def B(self, x, t=None) -> np.ndarray:
        '''
        Magnetic field vector.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float, optional
            Lab time. Ignored for static fields.

        Returns
        -------
        np.ndarray
            Magnetic field vector at (x, t), shape (3,).
            Returns zeros by default.
        '''
        return np.zeros(3)

    def phi(self, x, t=None) -> float:
        '''
        Electric scalar potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float, optional
            Lab time. Ignored for static fields.

        Returns
        -------
        float
            Scalar potential at (x, t).
            Returns zero by default.
        '''
        return 0.0

    def A(self, x, t=None) -> np.ndarray:
        '''
        Magnetic vector potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float, optional
            Lab time. Ignored for static fields.

        Returns
        -------
        np.ndarray
            Vector potential at (x, t), shape (3,).
            Returns zeros by default.
        '''
        return np.zeros(3)

    def phi_t(self, x, t=None) -> float:
        '''
        Partial time derivative of the scalar potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float, optional
            Lab time. Ignored for static fields.

        Returns
        -------
        float
            Partial time derivative of the scalar potential at (x, t).
            Returns zero by default.
        '''
        return 0.0

    def A_t(self, x, t=None) -> np.ndarray:
        '''
        Partial time derivative of the vector potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float, optional
            Lab time. Ignored for static fields.

        Returns
        -------
        np.ndarray
            Partial time derivative of the vector potential at (x, t),
            shape (3,). Returns zeros by default.
        '''
        return np.zeros(3)

    def A_x(self, x, t=None) -> np.ndarray:
        '''
        Spatial Jacobian of the vector potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float, optional
            Lab time. Ignored for static fields.

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
        Vector potential function with signature
        A(x) -> array_like, shape (3,).
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

    def E(self, x, t=None):
        return (np.asarray(self._E(x), dtype=float)
                if self._E is not None else super().E(x, t))

    def B(self, x, t=None):
        return (np.asarray(self._B(x), dtype=float)
                if self._B is not None else super().B(x, t))

    def phi(self, x, t=None):
        return (float(self._phi(x))
                if self._phi is not None else super().phi(x, t))

    def A(self, x, t=None):
        return (np.asarray(self._A(x), dtype=float)
                if self._A is not None else super().A(x, t))

    def phi_t(self, x, t=None):
        return super().phi_t(x, t)

    def A_t(self, x, t=None):
        return super().A_t(x, t)

    def A_x(self, x, t=None):
        return (np.asarray(self._A_x(x), dtype=float)
                if self._A_x is not None else super().A_x(x, t))


class TimeDependentField(Field):
    '''
    Electromagnetic field with explicit time dependence.

    All field quantities are functions of both position and time.
    The partial time derivatives of the scalar and vector potentials
    must be supplied explicitly if required by the integrator.

    Parameters
    ----------
    E_func : callable, optional
        Electric field function with signature
        E(x, t) -> array_like, shape (3,).
    B_func : callable, optional
        Magnetic field function with signature
        B(x, t) -> array_like, shape (3,).
    phi_func : callable, optional
        Scalar potential function with signature phi(x, t) -> float.
    A_func : callable, optional
        Vector potential function with signature
        A(x, t) -> array_like, shape (3,).
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
    Plane wave propagating in the z-direction, polarised in x
    (with omega = k = 1.0 in natural units where c = 1):

    >>> omega, k, E0 = 1.0, 1.0, 0.1
    >>> E_func = lambda x, t: [E0 * np.cos(omega * t - k * x[2]), 0., 0.]
    >>> B_func = lambda x, t: [0., E0 * np.cos(omega * t - k * x[2]), 0.]
    >>> field = TimeDependentField(E_func=E_func, B_func=B_func)
    '''

    def __init__(self, E_func=None, B_func=None, phi_func=None,
                 A_func=None, phi_t_func=None, A_t_func=None,
                 A_x_func=None):
        self._E = E_func
        self._B = B_func
        self._phi = phi_func
        self._A = A_func
        self._phi_t = phi_t_func
        self._A_t = A_t_func
        self._A_x = A_x_func

    def E(self, x, t=None):
        return (np.asarray(self._E(x, t), dtype=float)
                if self._E is not None else super().E(x, t))

    def B(self, x, t=None):
        return (np.asarray(self._B(x, t), dtype=float)
                if self._B is not None else super().B(x, t))

    def phi(self, x, t=None):
        return (float(self._phi(x, t))
                if self._phi is not None else super().phi(x, t))

    def A(self, x, t=None):
        return (np.asarray(self._A(x, t), dtype=float)
                if self._A is not None else super().A(x, t))

    def phi_t(self, x, t=None):
        return (float(self._phi_t(x, t))
                if self._phi_t is not None else super().phi_t(x, t))

    def A_t(self, x, t=None):
        return (np.asarray(self._A_t(x, t), dtype=float)
                if self._A_t is not None else super().A_t(x, t))

    def A_x(self, x, t=None):
        return (np.asarray(self._A_x(x, t), dtype=float)
                if self._A_x is not None else super().A_x(x, t))
