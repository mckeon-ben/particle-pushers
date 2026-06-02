'''
Gordon-Hafizi comoving-frame particle pushers.

Implements the Gordon-Hafizi spinor-based particle pusher for
relativistic charged particle tracking in the particle's comoving
frame. The velocity update is performed via a time evolution operator
acting on the spinor representation of the 4-velocity.

All quantities are in natural units where c = 1.

References
----------
Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher
for extreme fields. Computer Physics Communications, 258, p.107628.
'''

import warnings
import numpy as np
from ..pusher import Pusher, PusherOrderFour


_PAULI = np.array([
    np.eye(2),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]])
])
'''Pauli matrix array used for spinor conversions.'''


def _vector_to_spinor(a):
    '''
    Convert a 4-vector to its spinor representation.

    Parameters
    ----------
    a : array_like
        4-vector to convert, shape (4,).

    Returns
    -------
    np.ndarray
        Spinor representation of a, shape (2, 2).
    '''
    pauli_swap = np.swapaxes(_PAULI, 0, -1)
    pauli_dot_a = np.swapaxes(pauli_swap * a, -1, 0)
    return np.sum(pauli_dot_a, axis=0)


def _spinor_to_vector(A):
    '''
    Convert a spinor to its 4-vector representation.

    Parameters
    ----------
    A : array_like
        Spinor to convert, shape (2, 2).

    Returns
    -------
    np.ndarray
        4-vector representation of A, shape (4,).
    '''
    return np.trace(_PAULI @ A, axis1=1, axis2=2).real / 2


class Gordon(Pusher):
    '''
    Abstract base class for Gordon-Hafizi comoving-frame pushers.

    Implements the spinor-based velocity update common to all
    Gordon-Hafizi methods. The time evolution operator is applied
    to the spinor representation of the 4-velocity, with the specific
    operator determined by the concrete subclass.

    The system is autonomous in proper time — lab time is extracted
    directly from the zeroth component of the 4-position vector.

    Subclasses must implement _compute_time_operator().
    '''

    def _compute_F_spinor(self, x, t):
        '''
        Compute the electromagnetic field spinor and its norm.

        Constructs the complex electromagnetic field 3-vector from
        the electric and magnetic fields and converts it to spinor
        form via the Pauli matrix representation.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        F_spin : np.ndarray
            Electromagnetic field spinor, shape (2, 2).
        norm_F : complex
            Norm of the electromagnetic field 3-vector.
        '''
        E = self.field.E(x, t)
        B = self.field.B(x, t)
        F3 = (1 / 2) * self.q_over_m * (E + 1j * B)
        norm_F = np.sqrt(np.dot(F3, F3))
        F4 = np.hstack([0, F3])
        F_spin = _vector_to_spinor(F4)
        return F_spin, norm_F

    def _compute_time_operator(self, F, norm_F, dt):
        '''
        Compute the time evolution operator for the velocity spinor.

        Must be implemented by all concrete subclasses.

        Parameters
        ----------
        F : np.ndarray
            Electromagnetic field spinor, shape (2, 2).
        norm_F : complex
            Norm of the electromagnetic field spinor.
        dt : float
            Proper time step.

        Returns
        -------
        np.ndarray
            Time evolution operator, shape (2, 2).

        Raises
        ------
        NotImplementedError
            If called on the base class directly.
        '''
        raise NotImplementedError

    def _step(self, x, u, dt):
        '''
        Perform a single Gordon-Hafizi leapfrog step.

        Advances the 4-position and 4-velocity by one proper time
        step using a symmetric leapfrog splitting — half-step position
        update, full velocity update via the time evolution operator,
        then half-step position update.

        Parameters
        ----------
        x : np.ndarray
            Current 4-position, shape (4,).
        u : np.ndarray
            Current 4-velocity, shape (4,).
        dt : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity, shape (4,).
        '''
        # Half-step position update.
        x_mid = x + u * (dt / 2)

        # Compute time evolution operator at midpoint.
        F, norm_F = self._compute_F_spinor(x_mid[1:], x_mid[0])
        time_op = self._compute_time_operator(F, norm_F, dt)
        time_op_dagger = np.conj(time_op.T)

        # Apply time evolution operator to velocity spinor.
        U = _vector_to_spinor(u)
        U_new = time_op @ U @ time_op_dagger
        u_new = _spinor_to_vector(U_new)

        # Half-step position update with updated velocity.
        x_new = x_mid + u_new * (dt / 2)

        return x_new, u_new

    def advance(self, t_n, dt):
        return self._step(self.particle.x, self.particle.u, dt)


class GordonExact(Gordon):
    '''
    Gordon-Hafizi pusher with exact time evolution operator.

    Computes the time evolution operator exactly via the matrix
    exponential using hyperbolic functions. This is the exact solution
    to the equations of motion in a locally constant field.

    Properties
    ----------
    - Second-order accurate in dt
    - Exact for uniform fields
    - Norm-preserving in purely magnetic fields

    Notes
    -----
    Accuracy degrades for rapidly varying fields since the field is
    assumed locally constant over each time step. Not suitable for
    null electromagnetic fields where norm_F = 0.

    References
    ----------
    Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher
    for extreme fields. Computer Physics Communications, 258, p.107628.
    '''

    def _compute_time_operator(self, F, norm_F, dt):
        '''
        Exact time evolution operator via hyperbolic functions.

        Parameters
        ----------
        F : np.ndarray
            Electromagnetic field spinor, shape (2, 2).
        norm_F : complex
            Norm of the electromagnetic field spinor.
        dt : float
            Proper time step.

        Returns
        -------
        np.ndarray
            Exact time evolution operator, shape (2, 2).
        '''
        if norm_F == 0:
            norm_F += np.finfo(float).eps
        return (np.cosh(norm_F * dt) * np.eye(2)
                + np.sinh(norm_F * dt) * (F / norm_F))


class GordonQuadratic(Gordon):
    '''
    Gordon-Hafizi pusher with quadratic approximate time evolution operator.

    Computes the time evolution operator via a Pade-type rational
    approximation to the matrix exponential. The operator is unitary
    by construction, guaranteeing exact norm preservation at every step.

    Properties
    ----------
    - Second-order accurate in dt
    - Norm-preserving by construction
    - Suitable for null electromagnetic fields

    References
    ----------
    Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher
    for extreme fields. Computer Physics Communications, 258, p.107628.
    '''

    def _compute_time_operator(self, F, norm_F, dt):
        '''
        Approximate time evolution operator via Pade approximant.

        Parameters
        ----------
        F : np.ndarray
            Electromagnetic field spinor, shape (2, 2).
        norm_F : complex
            Norm of the electromagnetic field spinor.
        dt : float
            Proper time step.

        Returns
        -------
        np.ndarray
            Approximate time evolution operator, shape (2, 2).
        '''
        F_in = F * dt
        numer_squared = np.linalg.matrix_power(np.eye(2) + F_in / 2, 2)
        denom_squared = np.linalg.matrix_power(np.eye(2) - F_in / 2, -2)
        time_op_squared = numer_squared @ denom_squared
        trace = np.trace(time_op_squared)
        det = np.linalg.det(time_op_squared)
        s = np.sqrt(det)
        t = np.sqrt(trace + 2 * s)
        return (1 / t) * (s * np.eye(2) + time_op_squared)


class GordonExactOrderFour(PusherOrderFour, GordonExact):
    '''
    Fourth-order Gordon-Hafizi pusher with exact time evolution operator.

    Combines the Yoshida fourth-order composition with the exact hyperbolic
    time evolution operator from GordonExact. The exact operator is exact for
    uniform fields at each sub-step, so the only per-step error is the
    field-variation error, which the composition reduces to fourth order.

    Properties
    ----------
    - Fourth-order accurate in dt
    - Exact for uniform fields at each sub-step
    - Preserves the Minkowski norm (unit determinant) to machine precision

    Notes
    -----
    Because the exact operator is exact in a uniform field, the global error
    converges at fourth order down to round-off, making this the preferred
    choice when high accuracy is required.

    This is a comoving-frame method, so _lab_time is set to False: lab time
    rides in the zeroth component of the 4-position and the base _step()
    takes no explicit time argument.
    '''
    _lab_time = False


class GordonQuadraticOrderFour(PusherOrderFour, GordonQuadratic):
    '''
    Fourth-order Gordon-Hafizi pusher with quadratic time evolution operator.

    Combines the Yoshida fourth-order composition with the Pade-type
    approximate time evolution operator from GordonQuadratic.

    Properties
    ----------
    - Fourth-order accurate in dt down to the accuracy floor of the quadratic 
      operator
    - Norm-preserving by construction at each sub-step
    - No special functions or square roots in the kick (faster than the exact
      operator)

    Notes
    -----
    Unlike the exact operator, the quadratic operator is a rational (Pade)
    approximation and is not exact even in a uniform field. It carries an
    intrinsic per-step error that the composition cannot remove, so the global
    error converges at fourth order only until it reaches a floor set by the
    quadratic operator, below which refinement no longer improves accuracy.

    This is a comoving-frame method, so _lab_time is set to False: lab time
    rides in the zeroth component of the 4-position and the base _step()
    takes no explicit time argument.
    '''
    _lab_time = False
