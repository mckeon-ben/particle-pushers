'''
Vay lab-frame particle pusher.

Implements the Vay leapfrog method for relativistic charged particle
tracking in the lab frame. The Vay method correctly captures the
E×B drift velocity by construction. All quantities are in natural
units where c = 1.

References
----------
Vay, J.L., 2008. Simulation of beams or plasmas crossing at
relativistic velocity. Physics of Plasmas, 15(5).
'''

import numpy as np
from ..pusher import Pusher, PusherOrderFour
from ..lorentz import lorentz_gamma


class Vay(Pusher):
    '''
    Vay leapfrog pusher for relativistic charged particle tracking.

    A second-order explicit leapfrog method that correctly captures
    the E×B drift velocity. The velocity update uses a relativistic
    correction to the rotation angle that accounts for the full
    Lorentz factor after the combined electric and magnetic field
    update.

    Properties
    ----------
    - Second-order accurate in dt
    - Correctly captures the E×B drift velocity
    - Not volume-preserving in general
    '''

    def _step(self, x, u, t_n, dt):
        '''
        Perform a single Vay leapfrog step.

        Parameters
        ----------
        x : np.ndarray
            Current particle position, shape (3,).
        u : np.ndarray
            Current particle relativistic 3-velocity, shape (3,).
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
        x_mid = x + u / lorentz_gamma(u) * (dt / 2)
        t_mid = t_n + dt / 2
        E, B = self.field.E(x_mid, t_mid), self.field.B(x_mid, t_mid)

        # First velocity update before rotation in B-field.
        force = E + np.cross(u / lorentz_gamma(u), B)
        u_half = u + force * self.q_over_m * (dt / 2)

        # Rotating the velocity vector in B-field.
        tau = B * self.q_over_m * (dt / 2)
        u_prime = u_half + E * self.q_over_m * (dt / 2)
        u_star = np.dot(u_prime, tau)
        gamma_prime = lorentz_gamma(u_prime)
        tau_sq = np.dot(tau, tau)
        sigma = gamma_prime**2 - tau_sq

        # Lorentz gamma factor after rotation in B-field.
        gamma_new = np.sqrt(
            (sigma + np.sqrt(sigma**2 + 4 * (tau_sq + u_star**2))) / 2)

        # Scaling B-field for second velocity update.
        t_vec = tau / gamma_new
        s = 1 / (1 + np.dot(t_vec, t_vec))

        # Second velocity update after rotation in B-field.
        u_rot = u_prime + np.dot(u_prime, t_vec) * t_vec
        u_new = s * (u_rot + np.cross(u_prime, t_vec))

        x_new = x_mid + u_new / lorentz_gamma(u_new) * (dt / 2)
        return x_new, u_new

    def advance(self, t_n, dt):
        return self._step(self.particle.x, self.particle.u, t_n, dt)


class VayOrderFour(PusherOrderFour, Vay):
    '''
    Fourth-order Vay pusher via Yoshida triple-jump composition.

    Properties
    ----------
    - Fourth-order accurate in dt
    - Correctly captures the E×B drift velocity (inherited from Vay)
    '''
    pass
