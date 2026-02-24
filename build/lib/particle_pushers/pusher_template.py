import numpy as np
from particle_pushers.utils import Stagger

class Pusher:
	def __init__(self, 
				 x0,
				 u0,
				 E_field,
				 B_field,
				 q,
				 m,
				 units='natural',
				 stagger=False,
				 phi=None,
				 A_field=None,
				 A_prime=None
				 ):
		
		# Initialising variables.
		self.x0 = np.asarray(x0)
		self.u0 = np.asarray(u0)
		self.E_field = lambda x_input: np.asarray(E_field(x_input), dtype=float)
		self.B_field = lambda x_input: np.asarray(B_field(x_input), dtype=float)
		self.q = q
		self.m = m
		self.units = units
		self.stagger = stagger

		# Additional condition to initialise electric potential.
		if phi is not None:
			self.phi = lambda x_input: np.asarray(phi(x_input), dtype=float)
		else:
			self.phi = phi

		# Additional condition to initialise magnetic potential and its Jacobian.
		if A_field is not None and A_prime is not None:
			self.A_field = lambda x_input: np.asarray(A_field(x_input), dtype=float)
			self.A_prime = lambda x_input: np.asarray(A_prime(x_input), dtype=float)
		else:
			self.A_field = A_field
			self.A_prime = A_prime

		# Determine the unit system.
		if self.units == 'natural':
			self.c = 1
		elif self.units == 'SI':
			self.c = 299792458.0
		else:
			raise ValueError('Unknown units used. Please select either \'natural\' or \'SI\'.')

		# Check initial condition vectors are 3-vectors.
		if self.x0.size != 3 or self.u0.size != 3:
			raise ValueError('Initial position and velocity arrays must be 3-vectors (i.e. 1×3 arrays)!')

	# Solving the system with initial conditions x0 and u0.
	def solve(self, t_span, N):
		# Populating time step array.
		t_start, t_end = t_span
		self.dt = (t_end - t_start) / N
		self.t = np.linspace(t_start, t_end, N + 1)

		# Number of equations to solve (i.e. length of input vectors). 
		num_eqs = self.x0.size

		# Preallocating arrays to store the position and velocity.
		self.x = np.zeros((N + 1, num_eqs))
		self.u = np.zeros((N + 1, num_eqs))

		# Applying initial conditions.
		self.x[0] = self.x0
		self.u[0] = self.u0

		# Staggering the velocity by half a time step, if desired.
		if self.stagger == True:
			# Applying stagger operator and updating position.
			self.u_half = Stagger(self.x0, self.u0, self.E_field, self.B_field, self.q, self.m, self.c, self.dt)

			# Iteration evolving (staggered) system from t_start to t_end.
			for n in range(N):
				self.x_n = self.x[n]
				self.x[n + 1], self.u[n + 1] = self.advance()

		# Default process for non-staggered methods.
		else:
			# Iteration evolving system from t_start to t_end.
			for n in range(N):
				self.x_n = self.x[n]
				self.u_n = self.u[n]
				self.x[n + 1], self.u[n + 1] = self.advance()
		return self.t, self.x, self.u

	def advance(self):
		raise NotImplementedError('Method \'advance\' not implemented in base class!')
