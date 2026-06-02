'''
Base class for relativistic charged particle pushers.

Provides the common interface and integration loop shared by all
pusher implementations. All quantities are in natural units where c = 1.
'''

import numpy as np
from abc import ABC, abstractmethod
from .field import Field
from .particle import Particle


class Pusher(ABC):
    '''
    Abstract base class for relativistic charged particle pushers.

    Provides the integration loop and common interface for all pusher
    implementations. Concrete subclasses must implement the advance()
    method, which defines the specific time-stepping algorithm.

    All quantities are in natural units where c = 1.

    Parameters
    ----------
    particle : Particle
        The test particle to be pushed. Position and velocity may be
        3-vectors for lab-frame pushers or 4-vectors for comoving-frame
        pushers.
    field : Field
        The electromagnetic field in which the particle moves.

    Attributes
    ----------
    particle : Particle
        The test particle being pushed.
    field : Field
        The electromagnetic field.
    q_over_m : float
        Charge-to-mass ratio of the particle.

    Notes
    -----
    After solve() returns, self.particle holds the final state from
    the integration. Calling solve() a second time will therefore
    continue from the final state of the previous call rather than
    from the original initial conditions.

    Examples
    --------
    Concrete pushers are instantiated directly rather than through
    the base class:

    >>> field = StaticField(B_func=lambda x: np.array([0., 0., 1.]))
    >>> particle = Particle(x=np.array([1., 0., 0.]),
    ...                     u=np.array([0., 1., 0.]),
    ...                     q=1., m=1.)
    >>> sim = Boris(particle, field)
    >>> t, x, u = sim.solve((0, 2 * np.pi), N=1000)
    '''

    def __init__(self, particle: Particle, field: Field):
        if not isinstance(particle, Particle):
            raise TypeError(f'particle must be a Particle instance, got {type(particle).__name__}')
        if not isinstance(field, Field):
            raise TypeError(f'field must be a Field instance, got {type(field).__name__}')
        self.particle = particle
        self.field = field
        self.q_over_m = particle.q / particle.m

    @abstractmethod
    def advance(self, t_n: float, dt: float) -> tuple:
        '''
        Advance the particle state by one time step.

        Must be implemented by all concrete subclasses. The particle
        state is read from and written to self.particle by the solve()
        loop.

        Parameters
        ----------
        t_n : float
            Current time at the start of the step.
        dt : float
            Time step size.

        Returns
        -------
        x_new : np.ndarray
            Updated particle position.
        u_new : np.ndarray
            Updated particle velocity.
        '''

    def solve(self, t_span, N):
        '''
        Integrate the equations of motion over a given time interval.

        Note that self.particle is updated in-place during integration.
        After solve() returns, self.particle holds the final state.
        Calling solve() again will continue from that final state rather
        than from the original initial conditions.

        Parameters
        ----------
        t_span : tuple of float
            Integration interval (t_start, t_end). Must satisfy
            t_start < t_end.
        N : int
            Number of time steps. Must be a positive integer.

        Returns
        -------
        t : np.ndarray
            Time array at integer steps, shape (N + 1,).
        x_out : np.ndarray
            Particle position array, shape (N + 1, n_dims), where
            n_dims is 3 for lab-frame pushers or 4 for comoving-frame
            pushers.
        u_out : np.ndarray
            Particle velocity array, shape (N + 1, n_dims). For
            comoving-frame pushers with a staggered leapfrog scheme,
            u_out has shape (N, n_dims) instead.

        Raises
        ------
        TypeError
            If N is not an integer.
        ValueError
            If N is not positive, or if t_span does not contain exactly
            two elements with t_start < t_end.
        '''
        if not isinstance(N, (int, np.integer)):
            raise TypeError(f'N must be an integer, got {type(N).__name__}')
        if N <= 0:
            raise ValueError(f'N must be a positive integer, got {N}')
        try:
            t_start, t_end = t_span
        except (TypeError, ValueError):
            raise ValueError(f't_span must be a two-element sequence, got {t_span!r}')
        if t_start >= t_end:
            raise ValueError(f't_span must satisfy t_start < t_end, got ({t_start}, {t_end})')

        dt = (t_end - t_start) / N
        t = np.linspace(t_start, t_end, N + 1)

        n_dims = self.particle.x.size
        x_out = np.zeros((N + 1, n_dims))
        u_out = np.zeros((N + 1, n_dims))
        x_out[0] = self.particle.x
        u_out[0] = self.particle.u

        for n in range(N):
            x_out[n + 1], u_out[n + 1] = self.advance(t[n], dt)
            self.particle.x = x_out[n + 1]
            self.particle.u = u_out[n + 1]

        return t, x_out, u_out


class PusherOrderFour(Pusher):
    '''
    Frame-agnostic Yoshida triple-jump fourth-order composition.

    Lifts a symmetric second-order pusher to fourth-order accuracy by applying
    its base step three times with coefficients (w1, w0, w1), where the central
    coefficient w0 is negative (a backward sub-step), chosen so that the leading
    third-order error term of the symmetric base step cancels.

    This class provides the composed advance() only. The single-step method
    _step() is supplied by a concrete pusher mixed in by the subclass, listed
    after this class so that its _step() resolves via the method resolution
    order while this advance() takes precedence, e.g.
    ``class BorisOrderFour(PusherOrderFour, Boris)``.

    The two frames differ only in the _step() signature, selected by the
    _lab_time class attribute:

      * lab-frame (_lab_time = True, the default), _step(x, u, t_n, dt): lab
        time is explicit and is threaded across sub-steps by the cumulative
        fractional step, so time-dependent fields are sampled correctly;
      * comoving-frame (_lab_time = False), _step(x, u, dt): lab time rides in
        the zeroth component of the 4-position and needs no threading, so a
        comoving subclass must set ``_lab_time = False``.

    Notes
    -----
    The sub-step state is threaded directly through the three composition
    stages rather than through self.particle, which the solve() loop writes
    back only once per full step. Routing it through self.particle instead
    would make all three stages start from the same point rather than compose.

    References
    ----------
    Yoshida, H., 1990. Construction of higher order symplectic
    integrators. Physics Letters A, 150(5-7), pp.262-268.
    '''

    w1 = 1 / (2 - 2**(1/3))
    w0 = -2**(1/3) / (2 - 2**(1/3))
    coeffs = [w1, w0, w1]

    _lab_time = True

    def advance(self, t_n, dt):
        x, u = self.particle.x, self.particle.u
        t = t_n
        for c in self.coeffs:
            if self._lab_time:
                x, u = self._step(x, u, t, c * dt)
            else:
                x, u = self._step(x, u, c * dt)
            t += c * dt
        return x, u
