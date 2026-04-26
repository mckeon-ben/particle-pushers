'''
Boris lab-frame particle pushers.

Implements the Boris leapfrog method and its fourth-order Yoshida
extension and adaptive variants for relativistic charged particle
tracking in the lab frame. All quantities are in natural units
where c = 1.

References
----------
Boris, J.P., 1970. Relativistic plasma simulation-optimization of
a hybrid code. In Proc. Fourth Conf. Num. Sim. Plasmas (pp. 3-67).

Yoshida, H., 1990. Construction of higher order symplectic
integrators. Physics letters A, 150(5-7), pp.262-268.
'''

import numpy as np
from ..pusher import Pusher
from ..gamma import Gamma


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

    References
    ----------
    Boris, J.P., 1970. Relativistic plasma simulation-optimization of
    a hybrid code. In Proc. Fourth Conf. Num. Sim. Plasmas (pp. 3-67).
    '''

    def _compute_fields(self, x_mid, t_mid):
        '''
        Evaluate electric and magnetic fields at a given position and time.

        Parameters
        ----------
        x_mid : array_like
            Spatial position vector, shape (3,).
        t_mid : float
            Lab time.

        Returns
        -------
        E : np.ndarray
            Electric field vector, shape (3,).
        B : np.ndarray
            Magnetic field vector, shape (3,).
        '''
        E = self.field.E(x_mid, t_mid)
        B = self.field.B(x_mid, t_mid)
        return E, B

    def _compute_u_minus(self, u, E, dt):
        '''
        First half electric field kick.

        Parameters
        ----------
        u : array_like
            Current relativistic 3-velocity, shape (3,).
        E : array_like
            Electric field vector, shape (3,).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Half-kicked velocity before magnetic rotation, shape (3,).
        '''
        return u + E * self.q_over_m * (dt / 2)

    def _compute_t_vec(self, u_minus, B, dt):
        '''
        Boris rotation vector.

        Computes the rotation vector t = (q/m) * B/gamma * dt/2,
        whose magnitude is the tangent of the half rotation angle
        in the magnetic field.

        Parameters
        ----------
        u_minus : array_like
            Half-kicked velocity before magnetic rotation, shape (3,).
        B : array_like
            Magnetic field vector, shape (3,).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Boris rotation vector, shape (3,).
        '''
        gamma_minus = Gamma(u_minus)
        return (B / gamma_minus) * self.q_over_m * (dt / 2)

    def _compute_half_angle(self, u, E, B, dt):
        '''
        Compute the magnetic rotation half-angle tangent.

        Returns the norm of the Boris rotation vector, which equals
        the tangent of the half rotation angle in the magnetic field.
        Used by adaptive methods to determine whether sub-cycling
        or higher-order composition is required.

        Parameters
        ----------
        u : array_like
            Current relativistic 3-velocity, shape (3,).
        E : array_like
            Electric field vector, shape (3,).
        B : array_like
            Magnetic field vector, shape (3,).
        dt : float
            Time step.

        Returns
        -------
        float
            Tangent of the magnetic rotation half-angle.
        '''
        u_minus = self._compute_u_minus(u, E, dt)
        t_vec = self._compute_t_vec(u_minus, B, dt)
        return np.linalg.norm(t_vec)

    def _step(self, x, u, t_n, dt, E=None, B=None):
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
        E : array_like, optional
            Precomputed electric field vector, shape (3,).
            If None, evaluated from self.field at the midpoint.
        B : array_like, optional
            Precomputed magnetic field vector, shape (3,).
            If None, evaluated from self.field at the midpoint.

        Returns
        -------
        x_new : np.ndarray
            Updated particle position, shape (3,).
        u_new : np.ndarray
            Updated particle relativistic 3-velocity, shape (3,).
        '''
        t_mid = t_n + dt / 2
        x_mid = x + u / Gamma(u) * (dt / 2)
        if E is None or B is None:
            E, B = self._compute_fields(x_mid, t_mid)
        u_minus = self._compute_u_minus(u, E, dt)
        t_vec = self._compute_t_vec(u_minus, B, dt)
        s = (2 * t_vec) / (1 + np.linalg.norm(t_vec)**2)
        u_plus = u_minus + np.cross((u_minus + np.cross(u_minus, t_vec)), s)
        u_new = u_plus + E * self.q_over_m * (dt / 2)
        x_new = x_mid + u_new / Gamma(u_new) * (dt / 2)
        return x_new, u_new

    def advance(self, t_n, dt):
        return self._step(self.particle.x, self.particle.u, t_n, dt)


class BorisFourthOrder(Boris):
    '''
    Fourth-order Boris pusher via Yoshida triple-jump composition.

    Extends the second-order Boris method to fourth-order accuracy
    using the Yoshida composition scheme with three second-order
    steps and carefully chosen coefficients.

    Properties
    ----------
    - Fourth-order accurate in dt
    - Volume-preserving in phase space

    References
    ----------
    Yoshida, H., 1990. Construction of higher order symplectic
    integrators. Physics letters A, 150(5-7), pp.262-268.
    '''

    w1 = 1 / (2 - 2**(1 / 3))
    w0 = -2**(1 / 3) / (2 - 2**(1 / 3))
    coeffs = [w1, w0, w1]

    def advance(self, t_n, dt):
        x, u = self.particle.x, self.particle.u
        for c in self.coeffs:
            x, u = self._step(x, u, t_n, c * dt)
            t_n += c * dt
        return x, u


class BorisAdaptiveFourthOrder(BorisFourthOrder):
    '''
    Adaptive Boris pusher switching between second and fourth order.

    Uses the magnetic rotation half-angle to determine whether to
    apply the standard second-order Boris step or the fourth-order
    Yoshida composition. The fourth-order scheme is used when the
    half-angle exceeds the threshold, indicating that the particle
    is gyrating rapidly relative to the time step.

    Attributes
    ----------
    threshold : float
        Half-angle threshold in radians above which the fourth-order
        scheme is used. Default is 0.05.
    '''

    threshold = 0.05

    def advance(self, t_n, dt):
        x, u = self.particle.x, self.particle.u
        t_mid = t_n + dt / 2
        x_mid = x + u / Gamma(u) * (dt / 2)
        E, B = self._compute_fields(x_mid, t_mid)
        half_angle = self._compute_half_angle(u, E, B, dt)
        if half_angle > self.threshold:
            for c in self.coeffs:
                x, u = self._step(x, u, t_n, c * dt)
                t_n += c * dt
        else:
            x, u = self._step(x, u, t_n, dt, E=E, B=B)
        return x, u


class BorisAdaptiveSubstep(Boris):
    '''
    Adaptive Boris pusher with sub-cycling for rapid gyration.

    Uses the magnetic rotation half-angle to determine whether to
    apply the standard Boris step or subdivide the time step into
    smaller sub-steps. Sub-cycling is used when the half-angle
    exceeds the threshold, reducing the effective time step to
    better resolve the gyration.

    Attributes
    ----------
    threshold : float
        Half-angle threshold in radians above which sub-cycling
        is used. Default is 0.05.
    n_substeps : int
        Number of sub-steps to use when sub-cycling. Default is 4.
    '''

    threshold = 0.05
    n_substeps = 4

    def advance(self, t_n, dt):
        x, u = self.particle.x, self.particle.u
        t_mid = t_n + dt / 2
        x_mid = x + u / Gamma(u) * (dt / 2)
        E, B = self._compute_fields(x_mid, t_mid)
        half_angle = self._compute_half_angle(u, E, B, dt)
        if half_angle > self.threshold:
            dt_sub = dt / self.n_substeps
            for n in range(self.n_substeps):
                x, u = self._step(x, u, t_n, dt_sub)
                t_n += dt_sub
        else:
            x, u = self._step(x, u, t_n, dt, E=E, B=B)
        return x, u
