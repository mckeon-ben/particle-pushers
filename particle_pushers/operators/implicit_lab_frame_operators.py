import numpy as np
from scipy.optimize import fsolve
from particle_pushers.utils import GammaFactor,\
								   DiscreteGradientElectricField

def LapentaPositionOperator(x_in, u_in_1, u_in_2, c, dt):
	'''
	Lapenta-Markidis position time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in_1 : array_like
		Test particle initial velocity vector.
	u_in_2 : array_like
		Test particle velocity vector obtained
		via Newton's method.
	c : float
		Speed of light in vacuum.
	dt : float
		Lab frame time step.

	Returns
	-------
	x_out : numpy.ndarray
		Position vector at time step dt.
	'''
	x_out = x_in + (u_in_1 + u_in_2) / (GammaFactor(u_in_1, c) + GammaFactor(u_in_2, c)) * dt
	return x_out

def PetriPositionOperator(x_in, u_in_1, u_in_2, c, dt):
	'''
	Pétri position time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in_1 : array_like
		Test particle initial velocity vector.
	u_in_2 : array_like
		Test particle velocity vector obtained
		via Newton's method.
	c : float
		Speed of light in vacuum.
	dt : float
		Lab frame time step.

	Returns
	-------
	x_out : numpy.ndarray
		Position vector at time step dt.
	'''
	x_out = x_in + (u_in_1 / GammaFactor(u_in_1, c) + u_in_2 / GammaFactor(u_in_2, c)) * (dt / 2)
	return x_out

def DiscreteGradientPositionOperator(x_in, u_in_1, u_in_2, c, dt):
	'''
	Discrete gradient position time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in_1 : array_like
		Test particle initial velocity vector.
	u_in_2 : array_like
		Test particle velocity vector obtained
		via Newton's method.
	c : float
		Speed of light in vacuum.
	dt : float
		Lab frame time step.

	Returns
	-------
	x_out : numpy.ndarray
		Position vector at time step dt.
	'''
	# Average relativistic velocity.
	u_bar = (u_in_1 + u_in_2) / 2
	
	x_out = x_in + u_bar / GammaFactor(u_bar, c) * dt
	return x_out

def LapentaVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Lapenta-Markidis velocity time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	E_field : callable
		User-defined position-dependent electric field.
	B_field : callable
		User-defined position-dependent magnetic field.
	q : float
		Test particle charge.
	m : float
		Test particle mass.
	c : float
		Speed of light in vacuum.
	dt : float
		Lab frame time step.

	Returns
	-------
	u_out : numpy.ndarray
		Velocity vector at time step dt.
	'''
	# Calculating initial gamma factor.
	gamma_u_in = GammaFactor(u_in, c)

	def LapentaResidual(u_k):
		# Calculating residual gamma factor.
		gamma_k = GammaFactor(u_k, c)

		# Calculating position and velocity at the next half-step.
		v_half = (u_k + u_in) / (gamma_k + gamma_u_in)
		x_half = x_in + v_half * (dt / 2)

		# Returning the residual to be solved.
		return u_k - u_in - (E_field(x_half) + np.cross(v_half, B_field(x_half))) * (q / m) * dt

	# Solving for the velocity via Newton's method.
	u_out = fsolve(func=LapentaResidual, x0=u_in)
	return u_out

def PetriVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Pétri velocity time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	E_field : callable
		User-defined position-dependent electric field.
	B_field : callable
		User-defined position-dependent magnetic field.
	q : float
		Test particle charge.
	m : float
		Test particle mass.
	c : float
		Speed of light in vacuum.
	dt : float
		Lab frame time step.

	Returns
	-------
	u_out : numpy.ndarray
		Velocity vector at time step dt.
	'''
	# Calculating initial gamma factor.
	gamma_u_in = GammaFactor(u_in, c)

	def PetriResidual(u_k):
		# Calculating residual gamma factor.
		gamma_k = GammaFactor(u_k, c)

		# Calculating position and velocity at the next half-step.
		v_half = (u_k / gamma_k + u_in / gamma_u_in) / 2
		x_half = x_in + v_half * (dt / 2)

		# Returning the residual to be solved.
		return u_k - u_in - (E_field(x_half) + np.cross(v_half, B_field(x_half))) * (q / m) * dt

	# Solving for the velocity via Newton's method.
	u_out = fsolve(func=PetriResidual, x0=u_in)
	return u_out

def DiscreteGradientVelocityOperator(x_in, u_in, E_field, phi, B_field, q, m, c, dt):
	'''
	Discrete gradient velocity time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	E_field : callable
		User-defined position-dependent electric field.
	phi : callable
		User-defined scalar potential corresponding
		to the electric field defined by E_field.
	B_field : callable
		User-defined position-dependent magnetic field.
	q : float
		Test particle charge.
	m : float
		Test particle mass.
	c : float
		Speed of light in vacuum.
	dt : float
		Lab frame time step.

	Returns
	-------
	u_out : numpy.ndarray
		Velocity vector at time step dt.
	'''
	def DiscreteGradientResidual(u_k):
		# Calculating average relativistic velocity.
		u_half = (u_in + u_k) / 2
		
		# Calculating average gamma factor.
		gamma_half = GammaFactor(u_half, c)

		# Calculating position and velocity at the next half-step.
		v_half = u_half / gamma_half
		x_half = x_in + v_half * (dt / 2)

		# Advancing the position through a full time step.
		x_full = x_in + v_half * dt

		# Calculating the electric field via the discrete gradient.
		E_bar_vector = DiscreteGradientElectricField(x_full, x_in, E_field, phi)

		# Returning the residual to be solved.
		return u_k - u_in - (E_bar_vector + np.cross(v_half, B_field(x_half))) * (q / m) * dt

	# Solving for the velocity via Newton's method.
	u_out = fsolve(func=DiscreteGradientResidual, x0=u_in)
	return u_out
