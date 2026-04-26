'''
Vay lab-frame particle pushers.

Implements the Vay leapfrog method and its fourth-order Yoshida
extension and adaptive variants for relativistic charged particle
tracking in the lab frame. The Vay method correctly captures the
E×B drift velocity by construction. All quantities are in natural
units where c = 1.

References
----------
Vay, J.L., 2008. Simulation of beams or plasmas crossing at
relativistic velocity. Physics of Plasmas, 15(5).

Yoshida, H., 1990. Construction of higher order symplectic
integrators. Physics letters A, 150(5-7), pp.262-268.
'''

import numpy as np
from ..pusher import Pusher
from ..gamma import Gamma


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

    References
    ----------
    Vay, J.L., 2008. Simulation of beams or plasmas crossing at
    relativistic velocity. Physics of Plasmas, 15(5).
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

    def _compute_tau(self, B, dt):
        '''
        Vay rotation scaling vector.

        Computes tau = (q/m) * B * dt/2, used in the relativistic
        rotation angle calculation.

        Parameters
        ----------
        B : array_like
            Magnetic field vector, shape (3,).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray
            Vay rotation scaling vector, shape (3,).
        '''
        return B * self.q_over_m * (dt / 2)

    def _compute_half_angle(self, u, E, B, dt):
        '''
        Compute the magnetic rotation half-angle tangent.

        Returns the norm of the Vay rotation vector t = tau/gamma_new,
        which equals the tangent of the half rotation angle. Used by
        adaptive methods to determine whether sub-cycling or higher-order
        composition is required.

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
        gamma_u = Gamma(u)
        u_half = u + (E + np.cross(u / gamma_u, B)) * self.q_over_m * (dt / 2)
        tau = self._compute_tau(B, dt)
        u_prime = u_half + E * self.q_over_m * (dt / 2)
        u_star = np.dot(u_prime, tau)
        gamma_prime = Gamma(u_prime)
        sigma = gamma_prime**2 - np.linalg.norm(tau)**2
        gamma_new = np.sqrt((sigma + np.sqrt(sigma**2 + 4 * (np.linalg.norm(tau)**2 + u_star**2))) / 2)
        t_vec = tau / gamma_new
        return np.linalg.norm(t_vec)

    def _step(self, x, u, t_n, dt, E=None, B=None):
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

        # First velocity update before rotation in B-field.
        gamma_u = Gamma(u)
        u_half = u + (E + np.cross(u / gamma_u, B)) * self.q_over_m * (dt / 2)

        # Rotating the velocity vector in B-field.
        tau = self._compute_tau(B, dt)
        u_prime = u_half + E * self.q_over_m * (dt / 2)
        u_star = np.dot(u_prime, tau)
        gamma_prime = Gamma(u_prime)
        sigma = gamma_prime**2 - np.linalg.norm(tau)**2

        # Lorentz gamma factor after rotation in B-field.
        gamma_new = np.sqrt((sigma + np.sqrt(sigma**2 + 4 * (np.linalg.norm(tau)**2 + u_star**2))) / 2)

        # Scaling B-field for second velocity update.
        t_vec = tau / gamma_new
        s = 1 / (1 + np.linalg.norm(t_vec)**2)

        # Second velocity update after rotation in B-field.
        u_new = s * (u_prime + np.dot(u_prime, t_vec) * t_vec + np.cross(u_prime, t_vec))

        x_new = x_mid + u_new / Gamma(u_new) * (dt / 2)
        return x_new, u_new

    def advance(self, t_n, dt):
        return self._step(self.particle.x, self.particle.u, t_n, dt)


class VayFourthOrder(Vay):
    '''
    Fourth-order Vay pusher via Yoshida triple-jump composition.

    Extends the second-order Vay method to fourth-order accuracy
    using the Yoshida composition scheme with three second-order
    steps and carefully chosen coefficients.

    Properties
    ----------
    - Fourth-order accurate in dt
    - Correctly captures the E×B drift velocity at each sub-step

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


class VayAdaptiveFourthOrder(VayFourthOrder):
    '''
    Adaptive Vay pusher switching between second and fourth order.

    Uses the magnetic rotation half-angle to determine whether to
    apply the standard second-order Vay step or the fourth-order
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


class VayAdaptiveSubstep(Vay):
    '''
    Adaptive Vay pusher with sub-cycling for rapid gyration.

    Uses the magnetic rotation half-angle to determine whether to
    apply the standard Vay step or subdivide the time step into
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
