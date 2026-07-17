'''
Boris lab-frame particle pusher.

Implements the Boris leapfrog method for relativistic charged particle
tracking in the lab frame. All quantities are in natural units where c = 1.

References
----------
Boris, J.P., 1970. Relativistic Plasma Simulation — Optimization of
a Hybrid Code. In Proc. Fourth Conf. Num. Sim. Plasmas (pp. 3-67).
'''

import numpy as np
from ..pusher import Pusher, PusherOrderFour
from ..lorentz import lorentz_gamma


class Boris(Pusher):
    '''
    Boris leapfrog pusher for relativistic charged particle tracking.

    A second-order explicit leapfrog method in which the velocity
    update is split into two electric field kicks straddling a
    magnetic rotation via the Boris rotation operator. The position
    is updated using two symmetric half-steps.

    Properties
    ----------
    - Second-order accurate in dt
    - Volume-preserving in phase space
    - Correct E×B drift to leading order
    '''

    def _step(self, x, u, t_n, dt):
        '''
        Perform a single Boris leapfrog step.

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
        u_minus = u + E * self.q_over_m * (dt / 2)
        t_vec = (B / lorentz_gamma(u_minus)) * self.q_over_m * (dt / 2)
        s = (2 * t_vec) / (1 + np.dot(t_vec, t_vec))
        u_plus = u_minus + np.cross((u_minus + np.cross(u_minus, t_vec)), s)
        u_new = u_plus + E * self.q_over_m * (dt / 2)
        x_new = x_mid + u_new / lorentz_gamma(u_new) * (dt / 2)
        return x_new, u_new

    def advance(self, t_n, dt):
        return self._step(self.particle.x, self.particle.u, t_n, dt)


class BorisOrderFour(PusherOrderFour, Boris):
    '''
    Fourth-order Boris pusher via Yoshida triple-jump composition.

    Properties
    ----------
    - Fourth-order accurate in dt
    - Volume-preserving in phase space (inherited from Boris)
    '''
    pass
