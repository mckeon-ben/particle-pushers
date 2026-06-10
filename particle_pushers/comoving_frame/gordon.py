'''
Gordon-Hafizi comoving-frame particle pushers.

Implements Gordon-Hafizi spinor-based particle pushers for
relativistic charged particle tracking in the particle's comoving
frame. The velocity update is performed via a time evolution operator
acting on the spinor representation of the 4-velocity.

All quantities are in natural units where c = 1.

References
----------
Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher
for extreme fields. Computer Physics Communications, 258, p.107628.
'''

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

    The system is autonomous in proper time; lab time is extracted
    directly from the zeroth component of the 4-position vector.

    Subclasses must implement _compute_time_operator().
    '''

    def _compute_F_spinor(self, x, t):
        '''
        Compute the electromagnetic field spinor and its field invariant.

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
        field_invariant : complex
            Square root of the field invariant. Complex in general.
        '''
        E = self.field.E(x, t)
        B = self.field.B(x, t)
        F_vec = (1 / 2) * self.q_over_m * (E + 1j * B)
        field_invariant = np.sqrt(np.dot(F_vec, F_vec))
        F4 = np.hstack([0, F_vec])
        F_spin = _vector_to_spinor(F4)
        return F_spin, field_invariant

    def _compute_time_operator(self, F, field_invariant, dtau):
        '''
        Compute the time evolution operator for the velocity spinor.

        Must be implemented by all concrete subclasses.

        Parameters
        ----------
        F : np.ndarray
            Electromagnetic field spinor, shape (2, 2).
        field_invariant : complex
            Square root of the field invariant. Complex in general.
        dtau : float
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

    def _step(self, x, u, dtau):
        '''
        Perform a single Gordon-Hafizi leapfrog step.

        Advances the 4-position and 4-velocity by one proper time
        step using a symmetric leapfrog splitting; half-step position
        update, full velocity update via the time evolution operator,
        then half-step position update.

        Parameters
        ----------
        x : np.ndarray
            Current 4-position, shape (4,).
        u : np.ndarray
            Current 4-velocity, shape (4,).
        dtau : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity, shape (4,).
        '''
        # Half-step position update.
        x_mid = x + u * (dtau / 2)

        # Compute time evolution operator at midpoint.
        F, field_invariant = self._compute_F_spinor(x_mid[1:], x_mid[0])
        time_op = self._compute_time_operator(F, field_invariant, dtau)
        time_op_dagger = np.conj(time_op.T)

        # Apply time evolution operator to velocity spinor.
        U = _vector_to_spinor(u)
        U_new = time_op @ U @ time_op_dagger
        u_new = _spinor_to_vector(U_new)

        # Half-step position update with updated velocity.
        x_new = x_mid + u_new * (dtau / 2)

        return x_new, u_new

    def advance(self, t_n, dt):
        # Proper-time step here; LabTimeConversion overrides advance for lab time.
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

    References
    ----------
    Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher
    for extreme fields. Computer Physics Communications, 258, p.107628.
    '''

    def _compute_time_operator(self, F, field_invariant, dtau):
        '''
        Exact time evolution operator via hyperbolic functions.

        Parameters
        ----------
        F : np.ndarray
            Electromagnetic field spinor, shape (2, 2).
        field_invariant : complex
            Square root of the field invariant. Complex in general.
        dtau : float
            Proper time step.

        Returns
        -------
        np.ndarray
            Exact time evolution operator, shape (2, 2).
        '''
        if field_invariant == 0:
            field_invariant += np.finfo(float).eps
        return (np.cosh(field_invariant * dtau) * np.eye(2)
                + np.sinh(field_invariant * dtau) * (F / field_invariant))


class GordonQuadratic(Gordon):
    '''
    Gordon-Hafizi pusher with quadratic approximate time evolution operator.

    Computes the time evolution operator via a Pade-type rational
    approximation to the matrix exponential. The operator guarantees
    exact unit determinant preservation at every step.

    Properties
    ----------
    - Second-order accurate in dt
    - Unit determinant preserved by construction
    - Exact for null electromagnetic fields

    References
    ----------
    Gordon, D.F. and Hafizi, B., 2021. Special unitary particle pusher
    for extreme fields. Computer Physics Communications, 258, p.107628.
    '''

    def _compute_time_operator(self, F, field_invariant, dtau):
        '''
        Approximate time evolution operator via Pade approximant.

        Parameters
        ----------
        F : np.ndarray
            Electromagnetic field spinor, shape (2, 2).
        field_invariant : complex
            Square root of the field invariant. Complex in general.
        dtau : float
            Proper time step.

        Returns
        -------
        np.ndarray
            Approximate time evolution operator, shape (2, 2).
        '''
        F_in = F * dtau
        half_inv_sq = (field_invariant * dtau / 2)**2
        return ((1 + half_inv_sq) * np.eye(2) + F_in) / (1 - half_inv_sq)


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
    - Preserves unit determinant to machine precision

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


class LabTimeConversion:
    '''
    Mixin lifting a comoving-frame Gordon pusher to lab-time stepping.

    The base Gordon ``_step(x, u, dtau)`` treats ``dtau`` as a proper-time
    step and lets lab time accrue passively in the zeroth component of the
    4-position. This mixin instead takes a controlled lab-time step
    ``dt_lab`` and solves for the proper-time step ``dtau`` that realises it,
    so that ``solve((0, T), N)`` advances N equal lab-time steps and is
    directly comparable to the native lab-frame pushers (Boris, etc.).

    The conversion is the symmetric trapezoidal relation

        dt_lab = dtau * (gamma_n + gamma_{n+1}) / 2,

    solved by fixed-point iteration. Because gamma_{n+1} depends on dtau
    through the velocity update, each iterate calls the inherited proper-time
    ``_step`` to obtain the trial endpoint, updates dtau from the trapezoidal
    average, and repeats to convergence. The returned state is taken from
    the final converged ``_step`` so that (x, u) is self-consistent with the
    converged dtau.

    This conversion is time-symmetric: the reverse lab step from the evolved
    state recovers the forward proper step with opposite sign, so the
    composite lab step inherits the second-order, even-power error structure
    of the underlying proper-time leapfrog.

    The mixin is placed before a concrete Gordon pusher in the bases, e.g.
    ``class GordonExactLab(LabTimeConversion, GordonExact)``, so that this
    ``advance`` takes precedence while ``_step`` resolves to the pusher.

    Attributes
    ----------
    conversion_tol : float
        Absolute convergence tolerance on dtau for the fixed-point iteration.
    conversion_max_iter : int
        Maximum number of fixed-point iterations before giving up.
    '''

    conversion_tol = 1e-14
    conversion_max_iter = 100

    def _lab_step(self, x, u, dt_lab):
        '''
        Advance one controlled lab-time step.

        Solves the symmetric trapezoidal conversion for the proper-time
        step dtau and applies the inherited proper-time ``_step``.

        Parameters
        ----------
        x : np.ndarray
            Current 4-position, shape (4,).
        u : np.ndarray
            Current 4-velocity, shape (4,).
        dt_lab : float
            Lab-time step to realise.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity, shape (4,).

        Raises
        ------
        RuntimeError
            If the fixed-point iteration fails to converge within
            ``conversion_max_iter`` iterations.
        '''
        gamma_n = u[0]
        # Leading-order guess dtau = dt_lab / gamma_n, then iterate to converge.
        dtau = dt_lab / gamma_n
        x_new, u_new = self._step(x, u, dtau)
        for _ in range(self.conversion_max_iter):
            # The endpoint gamma factor is the zeroth element of u_new.
            dtau_next = dt_lab / ((gamma_n + u_new[0]) / 2)
            x_new, u_new = self._step(x, u, dtau_next)
            if abs(dtau_next - dtau) < self.conversion_tol:
                return x_new, u_new
            dtau = dtau_next
        raise RuntimeError(
            f'lab-time conversion failed to converge in '
            f'{self.conversion_max_iter} iterations (dt_lab={dt_lab})'
        )

    def advance(self, t_n, dt):
        return self._lab_step(self.particle.x, self.particle.u, dt)


class GordonExactLab(LabTimeConversion, GordonExact):
    '''
    Lab-time-stepped Gordon-Hafizi pusher with exact time evolution operator.

    Combines the symmetric lab-time conversion with the exact hyperbolic
    operator from GordonExact. Takes controlled lab-time steps, so it can be
    compared directly with native lab-frame pushers at a common dt.

    Properties
    ----------
    - Second-order accurate in lab-time dt
    - Velocity operator exact in uniform fields (midpoint-frozen otherwise)
    - Preserves unit determinant at each step
    '''


class GordonQuadraticLab(LabTimeConversion, GordonQuadratic):
    '''
    Lab-time-stepped Gordon-Hafizi pusher with quadratic time evolution operator.

    Combines the symmetric lab-time conversion with the Pade-type operator
    from GordonQuadratic. Takes controlled lab-time steps for direct
    comparison with native lab-frame pushers at a common dt.

    Properties
    ----------
    - Second-order accurate in lab-time dt
    - Unit determinant preserved by construction at each step
    '''


class GordonQuadraticOrderFour(PusherOrderFour, GordonQuadratic):
    '''
    Fourth-order Gordon-Hafizi pusher with quadratic time evolution operator.

    Combines the Yoshida fourth-order composition with the Pade-type
    approximate time evolution operator from GordonQuadratic.

    Properties
    ----------
    - Fourth-order accurate in dt down to the accuracy floor of the quadratic
      operator
    - Unit determinant preserved by construction at each step

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


class GordonExactLabOrderFour(PusherOrderFour, GordonExactLab):
    '''
    Fourth-order lab-time-stepped Gordon-Hafizi pusher (exact operator).

    Lifts the lab-time-converted exact method (GordonExactLab) to fourth order
    by Yoshida triple-jump composition. The composition is routed through the
    conversion-wrapped step (_lab_step), so each of the three weighted
    sub-steps -- including the central backward sub-step -- solves its own
    symmetric lab-time conversion. Because each lab-time sub-step is itself a
    time-symmetric second-order map, the triple-jump cancels its leading
    error and yields fourth-order accuracy in the lab-time step.

    The three sub-steps are composed in controlled lab time, so this method
    is directly comparable at a common dt with a fourth-order lab-frame
    composition (e.g. BorisOrderFour).

    Properties
    ----------
    - Fourth-order accurate in lab-time dt
    - Preserves unit determinant at each sub-step

    Notes
    -----
    The base step (_lab_step) carries lab time in the zeroth component of the
    4-position, so _lab_time is set to False (no explicit time threading); the
    composition is redirected from _step to _lab_step via _substep.
    '''
    _lab_time = False

    def _substep(self, x, u, t_n, dt):
        return self._lab_step(x, u, dt)


class GordonQuadraticLabOrderFour(PusherOrderFour, GordonQuadraticLab):
    '''
    Fourth-order lab-time-stepped Gordon-Hafizi pusher (quadratic operator).

    Lifts the lab-time-converted quadratic method (GordonQuadraticLab) to
    fourth order by Yoshida triple-jump composition, routed through the
    conversion-wrapped step (_lab_step) exactly as in
    GordonExactLabOrderFour. As with the proper-time quadratic fourth-order
    method, the composition converges at fourth order only until it reaches
    the accuracy floor set by the quadratic (Pade) operator.

    Properties
    ----------
    - Fourth-order accurate in lab-time dt down to the quadratic operator floor
    - Unit determinant preserved by construction at each sub-step

    Notes
    -----
    The base step (_lab_step) carries lab time in the zeroth component of the
    4-position, so _lab_time is set to False; the composition is redirected
    from _step to _lab_step via _substep.
    '''
    _lab_time = False

    def _substep(self, x, u, t_n, dt):
        return self._lab_step(x, u, dt)
