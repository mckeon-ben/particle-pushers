'''
Base class for relativistic charged particle pushers.

Provides the common interface and integration loop shared by all
pusher implementations. All quantities are in natural units where c = 1.
'''

import numpy as np
from .field import Field
from .particle import Particle


class Pusher:
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

    Examples
    --------
    Concrete pushers are instantiated directly rather than through
    the base class:

    >>> field = StaticField(B_func=lambda x: [0., 0., 1.])
    >>> particle = Particle(x=[1., 0., 0.],
    ...                     u=[0., 1., 0.],
    ...                     q=1., m=1.)
    >>> sim = Boris(particle, field)
    >>> t, x, u = sim.solve(t_span=(0, 2 * np.pi), N=1000)
    '''

    def __init__(self, particle: Particle, field: Field):
        self.particle = particle
        self.field = field
        self.q_over_m = particle.q / particle.m

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

        Raises
        ------
        NotImplementedError
            If called on the base class directly.
        '''
        raise NotImplementedError

    def solve(self, t_span, N):
        '''
        Integrate the equations of motion over a given time interval.

        Parameters
        ----------
        t_span : tuple of float
            Integration interval (t_start, t_end).
        N : int
            Number of time steps.

        Returns
        -------
        t : np.ndarray
            Time array at integer steps, shape (N + 1,).
        x_out : np.ndarray
            Particle position array, shape (N + 1, n_dims), where
            n_dims is 3 for lab-frame pushers or 4 for comoving-frame
            pushers.
        u_out : np.ndarray
            Particle velocity array, shape (N + 1, n_dims).
        '''
        t_start, t_end = t_span
        dt = (t_end - t_start) / N
        t = np.linspace(t_start, t_end, N + 1)

        n_dims = self.particle.x.size
        x_out = np.zeros([N + 1, n_dims])
        u_out = np.zeros([N + 1, n_dims])
        x_out[0] = self.particle.x
        u_out[0] = self.particle.u

        for n in range(N):
            x_out[n + 1], u_out[n + 1] = self.advance(t[n], dt)
            self.particle.x = x_out[n + 1]
            self.particle.u = u_out[n + 1]

        return t, x_out, u_out
