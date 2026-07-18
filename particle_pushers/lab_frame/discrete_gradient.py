'''
Discrete gradient lab-frame particle pusher.

Implements the discrete gradient implicit particle pusher for
relativistic charged particle tracking in the lab frame. The
method guarantees exact single-particle energy conservation for
static fields by replacing the standard electric field with a
discrete gradient of the scalar potential. All quantities are in
natural units where c = 1.

References
----------
Gonzalez, O., 1996. Time integration and discrete Hamiltonian
systems. Journal of Nonlinear Science, 6(5), pp.449-467.
'''

import warnings
import numpy as np
from scipy.optimize import fixed_point
from ..pusher import Pusher
from ..field import TimeDependentField
from ..lorentz import lorentz_gamma


class DiscreteGradient(Pusher):
    '''
    Discrete gradient implicit pusher for relativistic charged
    particle tracking.

    A fully implicit second-order method that guarantees exact
    single-particle energy conservation for static fields. The standard
    electric field is replaced by a discrete gradient of the scalar
    potential, constructed such that the work done by the electric field
    between any two positions exactly equals the potential energy difference.
    The magnetic field does no work and is handled separately.

    The velocity update is obtained by fixed-point iteration at each
    time step.

    Notes
    -----
    - Second-order accurate in dt
    - Exactly conserves the single-particle Hamiltonian H = gamma*m + q*phi
      for static fields
    - Requires an explicit scalar potential phi

    Warnings
    --------
    Energy conservation is not guaranteed for time-dependent fields,
    since the discrete gradient identity only holds when the potential
    is time-independent.
    '''

    def solve(self, t_span, N):
        '''
        Integrate the equations of motion over a given time interval.

        Identical to :meth:`Pusher.solve`, except that a UserWarning is
        emitted first if the field is time-dependent, since the
        discrete gradient energy conservation identity holds only for
        static fields. See :meth:`Pusher.solve` for parameters,
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

    def _compute_E_bar(self, x2, x1, t_n, dt):
        '''
        Discrete gradient modified electric field.

        Constructs the discrete gradient of the scalar potential,
        which ensures the work done by the electric field between
        x1 and x2 exactly equals the potential energy difference
        phi(x1) - phi(x2)::

            E_bar*(x2 - x1) = phi(x1) - phi(x2)

        Parameters
        ----------
        x2 : array_like
            Final spatial position vector, shape (3,).
        x1 : array_like
            Initial spatial position vector, shape (3,).
        t_n : float
            Current lab time.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Discrete gradient modified electric field, shape (3,).

        Notes
        -----
        Midpoint field quantities are evaluated at t_mid = t_n + dt / 2.
        When x1 and x2 coincide to within machine precision the discrete
        gradient is singular, and the field at x1 evaluated at t_n is
        returned instead.

        Note the argument order: the final position x2 precedes the
        initial position x1.
        '''
        t_mid = t_n + dt / 2
        x_bar = (x1 + x2) / 2
        delta_x = x2 - x1
        norm_delta_x = np.linalg.norm(delta_x)
        if norm_delta_x < np.finfo(float).eps:
            return self.field.E(x1, t_n)
        E_bar_val = self.field.E(x_bar, t_mid)
        phi1 = self.field.phi(x1, t_mid)
        phi2 = self.field.phi(x2, t_mid)
        coeff = (phi2 - phi1 + np.dot(E_bar_val, delta_x)) / norm_delta_x**2
        return E_bar_val - coeff * delta_x

    def advance(self, t_n, dt):
        '''
        Advance the particle state by one time step.

        Solves implicitly for the updated velocity using fixed-point
        iteration, with the electric field replaced by the discrete
        gradient modified field to ensure exact energy conservation.
        The magnetic field is evaluated at the midpoint position.

        Parameters
        ----------
        t_n : float
            Current lab time.
        dt : float
            Time step.

        Returns
        -------
        x_new : np.ndarray
            Updated particle position, shape (3,).
        u_new : np.ndarray
            Updated particle relativistic 3-velocity, shape (3,).

        Raises
        ------
        RuntimeError
            If the fixed-point iteration fails to converge.
        '''
        x = self.particle.x
        u = self.particle.u
        gamma_u = lorentz_gamma(u)

        def iteration(u_k):
            gamma_k = lorentz_gamma(u_k)
            v_mid = (u_k + u) / (gamma_k + gamma_u)
            x_mid = x + v_mid * (dt / 2)
            x_next = x + v_mid * dt
            E_bar = self._compute_E_bar(x_next, x, t_n, dt)
            B = self.field.B(x_mid, t_n + dt / 2)
            return u + (E_bar + np.cross(v_mid, B)) * self.q_over_m * dt

        try:
            u_new = fixed_point(func=iteration, x0=u, xtol=1e-12)
        except RuntimeError as e:
            raise RuntimeError(
                f"Discrete gradient solver failed to converge: {e}"
            ) from e

        # Position update using average of old and new velocities.
        gamma_new = lorentz_gamma(u_new)
        x_new = x + (u + u_new) / (gamma_u + gamma_new) * dt

        return x_new, u_new
