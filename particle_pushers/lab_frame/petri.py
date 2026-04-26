'''
Pétri lab-frame particle pusher.

Implements the Pétri implicit particle pusher for relativistic
charged particle tracking in the lab frame. The method is a fully
implicit analogue of the Vay pusher, using physical velocity averaging
to correctly capture the E×B drift velocity. All quantities are in
natural units where c = 1.

References
----------
Pétri, J., 2017. A fully implicit numerical integration of the
relativistic particle equation of motion. Journal of Plasma Physics,
83(2), p.705830206.
'''

import numpy as np
from scipy.optimize import fsolve
from ..pusher import Pusher
from ..gamma import Gamma


class Petri(Pusher):
    '''
    Pétri implicit pusher for relativistic charged particle tracking.

    A fully implicit second-order method analogous to the Vay pusher,
    in which the midpoint velocity is computed as the average of the
    physical velocities u/gamma rather than the proper velocities.
    This averaging is designed to correctly capture the E×B drift
    velocity in the implicit setting, inheriting this property from
    the Vay scheme.

    The velocity update is obtained by solving a nonlinear system via
    Newton's method at each time step.

    Properties
    ----------
    - Second-order accurate in dt
    - Unconditionally stable for large time steps
    - Correctly captures the E×B drift velocity

    Notes
    -----
    Unlike the Lapenta-Markidis method which averages proper velocities,
    the Pétri method averages physical velocities u/gamma. This distinction
    affects the conservation properties and drift behaviour of the scheme.
    Neither method guarantees exact single-particle energy conservation
    in the test particle context.

    References
    ----------
    Pétri, J., 2017. A fully implicit numerical integration of the
    relativistic particle equation of motion. Journal of Plasma Physics,
    83(2), p.705830206.
    '''

    def advance(self, t_n, dt):
        '''
        Advance the particle state by one time step.

        Solves implicitly for the updated velocity using Newton's method,
        with the midpoint velocity computed as the average of the physical
        velocities u/gamma at the old and new time steps.

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
        '''
        x = self.particle.x
        u = self.particle.u
        t_mid = t_n + dt / 2
        gamma_u = Gamma(u)

        def residual(u_k):
            gamma_k = Gamma(u_k)
            v_mid = (u_k / gamma_k + u / gamma_u) / 2
            x_mid = x + v_mid * (dt / 2)
            E = self.field.E(x_mid, t_mid)
            B = self.field.B(x_mid, t_mid)
            return u_k - u - (E + np.cross(v_mid, B)) * self.q_over_m * dt

        u_new = fsolve(func=residual, x0=u)

        # Position update using average of old and new physical velocities.
        gamma_new = Gamma(u_new)
        x_new = x + (u / gamma_u + u_new / gamma_new) * (dt / 2)

        return x_new, u_new
