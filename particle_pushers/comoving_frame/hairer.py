'''
Hairer-Lubich-Shi comoving-frame particle pushers.

Implements the Hairer-Lubich-Shi family of particle pushers for
relativistic charged particle tracking in the particle's comoving
frame. All methods use a staggered leapfrog scheme in which positions
are stored at integer proper time steps and velocities are stored at
half-integer proper time steps.

The Minkowski metric M = diag(-1, 1, 1, 1) is incorporated explicitly
into all field tensor and Jacobian computations. All quantities are in
natural units where c = 1.

References
----------
Hairer, E., Lubich, C. and Shi, Y., 2023. Leapfrog methods for
relativistic charged-particle dynamics. SIAM Journal on Numerical
Analysis, 61(6), pp.2844-2858.
'''

import warnings
import numpy as np
from abc import abstractmethod
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import fixed_point
from ..pusher import Pusher
from ..field import TimeDependentField


_M_INV = np.diag((-1., 1., 1., 1.))
'''Inverse Minkowski metric tensor.

Since M = diag(-1, 1, 1, 1), the metric is its own inverse.
'''


class Hairer(Pusher):
    '''
    Abstract base class for Hairer-Lubich-Shi comoving-frame pushers.

    Provides the staggered leapfrog integration loop, electromagnetic
    field tensor construction, Cayley transform application, and stagger
    operator shared by all Hairer methods. The system is autonomous in
    proper time; lab time is extracted directly from the zeroth component
    of the 4-position vector.

    Positions are stored at integer proper time steps and velocities
    at half-integer proper time steps. The stagger operator initialises
    the scheme by advancing the velocity by half a time step and the
    position by a full time step before the main iteration begins.

    Subclasses must implement _step().
    '''

    def _compute_F_tensor(self, x, t, E=None, B=None):
        '''
        Construct the electromagnetic field tensor F.

        Builds the 4×4 electromagnetic field tensor from the electric
        and magnetic field vectors. Field values are evaluated from
        self.field if not provided explicitly.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.
        E : array_like, optional
            Precomputed electric field vector, shape (3,).
            If None, evaluated from self.field at (x, t).
        B : array_like, optional
            Precomputed magnetic field vector, shape (3,).
            If None, evaluated from self.field at (x, t).

        Returns
        -------
        np.ndarray
            Electromagnetic field tensor F, shape (4, 4).
        '''
        if E is None:
            E = self.field.E(x, t)
        if B is None:
            B = self.field.B(x, t)
        F = np.zeros((4, 4))
        F[0, 1:] = -E
        F[1:, 0] = E
        for i in range(1, 4):
            F[1:, i] = np.cross(-B, np.eye(3)[:, i - 1])
        return F

    def _stagger(self, x, u, dt):
        '''
        Stagger the position and velocity by half a time step.

        Advances the 4-velocity by a half proper time step and the
        4-position by a full proper time step to initialise the
        staggered leapfrog scheme. The zeroth component of the
        4-velocity is updated to ensure Lorentz invariance.

        Parameters
        ----------
        x : np.ndarray
            Initial 4-position, shape (4,).
        u : np.ndarray
            Initial 4-velocity, shape (4,).
        dt : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            4-position at the first integer step, shape (4,).
        u_new : np.ndarray
            4-velocity at the first half-integer step, shape (4,).
        '''
        F = self._compute_F_tensor(x[1:], x[0])
        u_new = (np.eye(4) + _M_INV @ F * self.q_over_m * dt / 2) @ u
        u_new[0] = np.sqrt(1 + np.dot(u_new[1:], u_new[1:]))
        x_new = x + u_new * dt
        return x_new, u_new

    def solve(self, t_span, N):
        '''
        Integrate the equations of motion over a given proper time
        interval.

        Uses a staggered leapfrog scheme in which positions are stored
        at integer proper time steps and velocities at half-integer
        proper time steps. The stagger operator is applied once before
        the main iteration begins.

        Parameters
        ----------
        t_span : tuple of float
            Integration interval (t_start, t_end) in proper time.
        N : int
            Number of proper time steps.

        Returns
        -------
        t : np.ndarray
            Proper time array at integer steps, shape (N + 1,).
        x_out : np.ndarray
            4-position array at integer steps, shape (N + 1, 4).
        u_out : np.ndarray
            4-velocity array at half-integer steps, shape (N, 4).
        '''
        # Delegate input validation to Pusher.solve before proceeding.
        if not isinstance(N, (int, np.integer)):
            raise TypeError(f'N must be an integer, got {type(N).__name__}')
        if N <= 0:
            raise ValueError(f'N must be a positive integer, got {N}')
        try:
            t_start, t_end = t_span
        except (TypeError, ValueError):
            raise ValueError(
                f't_span must be a two-element sequence, got {t_span!r}')
        if t_start >= t_end:
            raise ValueError(
                f't_span must satisfy t_start < t_end, '
                f'got ({t_start}, {t_end})')

        dt = (t_end - t_start) / N
        t = np.linspace(t_start, t_end, N + 1)

        n_dims = self.particle.x.size
        x_out = np.zeros((N + 1, n_dims))
        u_out = np.zeros((N, n_dims))
        x_out[0] = self.particle.x

        x_out[1], u_out[0] = self._stagger(
            self.particle.x, self.particle.u, dt
        )
        self.particle.x = x_out[1]
        self.particle.u = u_out[0]

        for n in range(1, N):
            x_out[n + 1], u_out[n] = self.advance(t[n], dt)
            self.particle.x = x_out[n + 1]
            self.particle.u = u_out[n]

        return t, x_out, u_out

    def _cayley_apply(self, A, u):
        '''
        Apply the Cayley transform of a matrix to a vector.

        Returns the action of the Cayley transform (I - A)^{-1}(I + A)
        on the vector u, obtained by solving the linear system
        (I - A) w = (I + A) u. The Cayley transform maps elements of a
        quadratic Lie algebra to the corresponding Lie group; it is
        used to construct the velocity update operator for all
        Hairer-Lubich-Shi leapfrog methods.

        Parameters
        ----------
        A : np.ndarray
            Input matrix, shape (4, 4).
        u : np.ndarray
            Vector to transform, shape (4,).

        Returns
        -------
        np.ndarray
            Cayley transform of A applied to u, shape (4,).
        '''
        return np.linalg.solve(np.eye(4) - A, (np.eye(4) + A) @ u)

    @abstractmethod
    def _step(self, x, u, dt):
        '''
        Perform a single integration step.

        Must be implemented by all concrete subclasses.

        Parameters
        ----------
        x : np.ndarray
            Current 4-position, shape (4,).
        u : np.ndarray
            Current 4-velocity at the half-integer step, shape (4,).
        dt : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position at the next integer step, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity at the next half-integer step, shape (4,).
        '''

    def advance(self, t_n, dt):
        '''
        Advance the particle state by one proper time step.

        The system is autonomous in proper time, so ``dt`` is
        interpreted as a proper time increment and ``t_n`` is unused;
        lab time is carried in the zeroth component of the 4-position
        and advances automatically within ``_step``.

        Parameters
        ----------
        t_n : float
            Current lab time. Accepted for interface uniformity with
            the lab-frame pushers and ignored here.
        dt : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position at the next integer step, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity at the next half-integer step,
            shape (4,).
        '''
        return self._step(self.particle.x, self.particle.u, dt)


class HairerExplicit(Hairer):
    '''
    Hairer-Lubich-Shi explicit leapfrog pusher.

    Advances the 4-velocity using the Cayley transform of the
    electromagnetic field tensor, evaluated at the current integer
    4-position. The position is updated using the new half-integer
    4-velocity.

    Notes
    -----
    - Second-order accurate in proper time step dt
    - Preserves the mass shell condition u^mu u_mu = -1 exactly
    - Volume-preserving in phase space
    '''

    def _step(self, x, u, dt):
        '''
        Perform a single explicit Hairer-Lubich-Shi leapfrog step.

        Parameters
        ----------
        x : np.ndarray
            Current 4-position at integer step, shape (4,).
        u : np.ndarray
            Current 4-velocity at half-integer step, shape (4,).
        dt : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position at the next integer step, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity at the next half-integer step, shape (4,).
        '''
        F = self._compute_F_tensor(x[1:], x[0])
        u_new = self._cayley_apply(_M_INV @ F * self.q_over_m * dt / 2, u)
        x_new = x + u_new * dt
        return x_new, u_new


class HairerDiscreteGradient(Hairer):
    '''
    Hairer-Lubich-Shi implicit discrete gradient pusher.

    Extends the explicit Hairer method to include an electric potential
    via the discrete gradient construction, which ensures exact energy
    conservation for static fields. The velocity update is obtained
    implicitly via fixed-point iteration at each step.

    Notes
    -----
    - Second-order accurate in proper time step dt
    - Exactly conserves the Hamiltonian H = gamma*m + q*phi for static fields
    - Preserves the mass shell condition u^mu u_mu = -1 exactly

    Warnings
    --------
    Energy conservation is not guaranteed in time-dependent fields.
    '''

    def solve(self, t_span, N):
        '''
        Integrate the equations of motion over a given proper time
        interval.

        Identical to :meth:`Hairer.solve`, except that a UserWarning is
        emitted first if the field is time-dependent, since the
        discrete gradient energy conservation identity holds only for
        static fields. See :meth:`Hairer.solve` for parameters,
        returns and raised exceptions.

        Warns
        -----
        UserWarning
            If the field is a TimeDependentField instance.
        '''
        if isinstance(self.field, TimeDependentField):
            warnings.warn(
                'Energy conservation is not guaranteed in '
                'time-dependent fields!',
                UserWarning,
                stacklevel=2
            )
        return super().solve(t_span, N)

    def _compute_E_bar(self, x2, x1):
        '''
        Discrete gradient modified electric field.

        Constructs the discrete gradient of the scalar potential,
        which ensures the work done by the electric field between
        x1 and x2 exactly equals the potential energy difference
        phi(x1) - phi(x2)::

            E_bar*(x2 - x1) = phi(x1) - phi(x2)

        Parameters
        ----------
        x2 : np.ndarray
            4-position at the next half-integer step, shape (4,).
        x1 : np.ndarray
            4-position at the previous half-integer step, shape (4,).

        Returns
        -------
        np.ndarray
            Discrete gradient modified electric field, shape (3,).

        Notes
        -----
        The spatial separation is taken from components 1 to 3 of the
        4-positions; the field and potential are sampled at the lab
        time carried in component 0. When x1 and x2 coincide to within
        machine precision the discrete gradient is singular, and the
        field at x1 is returned instead.

        Note the argument order: the later position x2 precedes the
        earlier position x1.
        '''
        x_bar = (x1 + x2) / 2
        delta_x = x2[1:] - x1[1:]
        norm_delta_x = np.linalg.norm(delta_x)
        E_bar_val = self.field.E(x_bar[1:], x_bar[0])
        if norm_delta_x < np.finfo(float).eps:
            return self.field.E(x1[1:], x1[0])
        phi1 = self.field.phi(x1[1:], x1[0])
        phi2 = self.field.phi(x2[1:], x2[0])
        coeff = (phi2 - phi1 + np.dot(E_bar_val, delta_x)) / norm_delta_x**2
        return E_bar_val - coeff * delta_x

    def _step(self, x, u, dt):
        '''
        Perform a single implicit discrete gradient step.

        Solves implicitly for the updated 4-velocity using fixed-point
        iteration, with the electric field replaced by the discrete
        gradient modified field to ensure exact energy conservation.

        Parameters
        ----------
        x : np.ndarray
            Current 4-position at integer step, shape (4,).
        u : np.ndarray
            Current 4-velocity at half-integer step, shape (4,).
        dt : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position at the next integer step, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity at the next half-integer step, shape (4,).

        Raises
        ------
        RuntimeError
            If the fixed-point iteration fails to converge.
        '''
        def iteration(u_k):
            x_prev_half = x - u * (dt / 2)
            x_next_half = x + u_k * (dt / 2)
            E_bar = self._compute_E_bar(x_next_half, x_prev_half)
            F_bar = self._compute_F_tensor(x[1:], x[0], E=E_bar)
            return self._cayley_apply(
                _M_INV @ F_bar * self.q_over_m * dt / 2, u)

        try:
            u_new = fixed_point(func=iteration, x0=u, xtol=1e-12)
        except RuntimeError as e:
            raise RuntimeError(
                "Hairer-Lubich-Shi discrete gradient solver failed to "
                f"converge: {e}") from e

        x_new = x + u_new * dt
        return x_new, u_new


class HairerVariational(Hairer):
    '''
    Hairer-Lubich-Shi variational leapfrog pusher.

    Implements the variational leapfrog integrator derived from the
    discrete Euler-Lagrange equations of a discretised action integral.
    The method adds a correction term involving the Jacobian of the
    4-potential and a finite difference of the 4-potential to the
    explicit Hairer update.

    The velocity update is obtained implicitly via fixed-point
    iteration at each step.

    Notes
    -----
    - Second-order accurate in proper time step dt
    - Preserves the mass shell condition u^mu u_mu = -1 up to O(dt^2)
    - Conserves the Hamiltonian H up to O(dt^2)
    - Derived from a discrete variational principle
    '''

    def _compute_jacobian(self, x, t):
        '''
        Construct the Jacobian of the 4-potential with respect to the
        4-position.

        Assembles the 4×4 matrix partial_mu A_nu from the electric
        field, partial time derivatives and spatial Jacobian of
        the vector potential.

        Parameters
        ----------
        x : array_like
            Spatial position vector, shape (3,).
        t : float
            Lab time.

        Returns
        -------
        A_prime : np.ndarray
            Jacobian matrix partial_mu A_nu, shape (4, 4).
        '''
        E = self.field.E(x, t)
        A_x = self.field.A_x(x, t)
        phi_t = self.field.phi_t(x, t)
        A_t = self.field.A_t(x, t)

        A_prime = np.zeros((4, 4))
        A_prime[0, 0] = -phi_t
        A_prime[1:, 0] = A_t
        A_prime[0, 1:] = E
        A_prime[1:, 1:] = A_x

        return A_prime

    def _step(self, x, u, dt):
        '''
        Perform a single implicit variational leapfrog step.

        Solves implicitly for the updated 4-velocity using fixed-point
        iteration. The update includes the standard Cayley transform
        velocity update plus correction terms from the Jacobian of the
        4-potential with respect to the 4-position and the finite
        difference of the 4-potential between adjacent integer
        positions.

        Parameters
        ----------
        x : np.ndarray
            Current 4-position at integer step, shape (4,).
        u : np.ndarray
            Current 4-velocity at half-integer step, shape (4,).
        dt : float
            Proper time step.

        Returns
        -------
        x_new : np.ndarray
            Updated 4-position at the next integer step, shape (4,).
        u_new : np.ndarray
            Updated 4-velocity at the next half-integer step, shape (4,).

        Raises
        ------
        RuntimeError
            If the fixed-point iteration fails to converge.
        '''
        F = self._compute_F_tensor(x[1:], x[0])
        jac = self._compute_jacobian(x[1:], x[0])
        F_mod = F + jac

        # x_prev, phi_prev, A_prev, and A4_prev depend only on x and u,
        # both of which are fixed during iteration.
        x_prev = x - u * dt
        phi_prev = self.field.phi(x_prev[1:], x_prev[0])
        A_prev = self.field.A(x_prev[1:], x_prev[0])
        A4_prev = np.hstack((-phi_prev, A_prev))

        cay_arg = _M_INV @ F_mod * self.q_over_m * dt / 2
        lu_fac = lu_factor(np.eye(4) - cay_arg)
        num_fac = np.eye(4) + cay_arg

        def iteration(u_k):
            x_next = x + u_k * dt
            phi_next = self.field.phi(x_next[1:], x_next[0])
            A_next = self.field.A(x_next[1:], x_next[0])
            A4_next = np.hstack((-phi_next, A_next))
            A_bar = (A4_next - A4_prev) / 2
            return lu_solve(lu_fac, num_fac @ u - _M_INV @ A_bar)

        try:
            u_new = fixed_point(func=iteration, x0=u, xtol=1e-12)
        except RuntimeError as e:
            raise RuntimeError(
                "Hairer-Lubich-Shi variational solver failed to "
                f"converge: {e}") from e

        x_new = x + u_new * dt
        return x_new, u_new
