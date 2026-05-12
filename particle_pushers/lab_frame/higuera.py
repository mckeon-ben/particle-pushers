'''
Higuera-Cary lab-frame particle pusher.

Implements the Higuera-Cary leapfrog method for relativistic charged
particle tracking in the lab frame. The Higuera-Cary method is both
volume-preserving and correctly captures the E×B drift velocity,
combining the key properties of the Boris and Vay methods. All
quantities are in natural units where c = 1.

References
----------
Higuera, A.V. and Cary, J.R., 2017. Structure-preserving second-order
integration of relativistic charged particle trajectories in
electromagnetic fields. Physics of Plasmas, 24(5).
'''

import numpy as np
from ..pusher import Pusher
from ..lorentz import lorentz_gamma


class Higuera(Pusher):
    '''
    Higuera-Cary leapfrog pusher for relativistic charged particle tracking.

    A second-order explicit leapfrog method that is both volume-preserving
    and correctly captures the E×B drift velocity. The velocity update
    uses a relativistic correction to the rotation angle computed from
    the half-accelerated momentum, combining the structure-preserving
    property of Boris with the correct drift behaviour of Vay.

    Properties
    ----------
    - Second-order accurate in dt
    - Volume-preserving in phase space
    - Correctly captures the E×B drift velocity
    '''

    def _step(self, x, u, t_n, dt):
        '''
        Perform a single Higuera-Cary leapfrog step.

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
        E, B = self.field.E(x_mid, t_n + dt / 2), self.field.B(x_mid, t_n + dt / 2)

        # First velocity update before rotation in B-field.
        u_minus = u + E * self.q_over_m * (dt / 2)

        # Rotating the velocity vector in B-field.
        gamma_minus = lorentz_gamma(u_minus)
        tau = B * self.q_over_m * (dt / 2)
        u_star = np.dot(u_minus, tau)
        tau_sq = np.dot(tau, tau)
        sigma = gamma_minus**2 - tau_sq

        # Lorentz gamma factor after rotation in B-field.
        gamma_plus = np.sqrt((sigma + np.sqrt(sigma**2 + 4 * (tau_sq + u_star**2))) / 2)

        # Scaling B-field for second velocity update.
        t_vec = tau / gamma_plus
        s = 1 / (1 + np.dot(t_vec, t_vec))

        # Intermediate velocity after rotation in B-field.
        u_plus = s * (u_minus + np.dot(u_minus, t_vec) * t_vec + np.cross(u_minus, t_vec))

        # Second velocity update after rotation in B-field.
        u_new = u_plus + E * self.q_over_m * (dt / 2) + np.cross(u_plus, t_vec)

        x_new = x_mid + u_new / lorentz_gamma(u_new) * (dt / 2)
        return x_new, u_new

    def advance(self, t_n, dt):
        return self._step(self.particle.x, self.particle.u, t_n, dt)
