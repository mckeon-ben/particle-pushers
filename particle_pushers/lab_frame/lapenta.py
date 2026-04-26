'''
Lapenta-Markidis lab-frame particle pusher.

Implements the Lapenta-Markidis implicit particle pusher for
relativistic charged particle tracking in the lab frame.
All quantities are in natural units where c = 1.

References
----------
Lapenta, G. and Markidis, S., 2011. Particle acceleration and energy
conservation in particle in cell simulations. Physics of Plasmas, 18(7).
'''

import numpy as np
from scipy.optimize import fsolve
from ..pusher import Pusher
from ..gamma import Gamma


class Lapenta(Pusher):
    '''
    Lapenta-Markidis implicit pusher for relativistic charged particle tracking.

    A fully implicit second-order method in which the velocity update
    is obtained by solving a nonlinear system via Newton's method at
    each time step. The field is evaluated at the midpoint position,
    which is itself a function of the updated velocity, making the
    scheme fully implicit and self-consistent.

    The method is designed for use in fully coupled particle-in-cell
    simulations where exact energy conservation is achieved through
    the self-consistent update of fields and particles. In the test
    particle context used here, energy conservation is approximate
    rather than exact.

    Properties
    ----------
    - Second-order accurate in dt
    - Unconditionally stable for large time steps
    - Exactly energy-conserving in a fully coupled PIC scheme

    Notes
    -----
    Energy conservation is not guaranteed in the test particle context
    since the back-reaction of the particle on the fields is absent.
    Exact energy conservation requires the field solver, current
    deposition and particle pusher to be fully coupled.

    References
    ----------
    Lapenta, G. and Markidis, S., 2011. Particle acceleration and energy
    conservation in particle in cell simulations. Physics of Plasmas, 18(7).
    '''

    def advance(self, t_n, dt):
        '''
        Advance the particle state by one time step.

        Solves implicitly for the updated velocity using Newton's method,
        with the electromagnetic field evaluated at the midpoint position
        and time. The midpoint position depends on the updated velocity,
        making the scheme fully implicit.

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
            v_mid = (u_k + u) / (gamma_k + gamma_u)
            x_mid = x + v_mid * (dt / 2)
            E = self.field.E(x_mid, t_mid)
            B = self.field.B(x_mid, t_mid)
            return u_k - u - (E + np.cross(v_mid, B)) * self.q_over_m * dt

        u_new = fsolve(func=residual, x0=u)

        # Position update using average of old and new velocities.
        gamma_new = Gamma(u_new)
        x_new = x + (u + u_new) / (gamma_u + gamma_new) * dt

        return x_new, u_new
