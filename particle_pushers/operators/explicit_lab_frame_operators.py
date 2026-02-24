import numpy as np
from particle_pushers.utils import GammaFactor

def ExplicitLabFramePositionOperator(x_in, u_in, c, dt):
	'''
	Position time evolution operator for all explicit methods
	in the lab frame.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	c : float
		Speed of light in vacuum.
	dt : float
		Lab frame time step.

	Returns
	-------
	x_out : numpy.ndarray
		Position vector at time step dt.
	'''
	x_out = x_in + u_in / GammaFactor(u_in, c) * dt
	return x_out

def BorisVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Boris velocity time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	E_field : callable
		User-defined electric field.
	B_field : callable
		User-defined magnetic field.
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
	# First velocity update before rotation in B-field.
	u_minus = u_in + E_field(x_in) * (q / m) * (dt / 2)

	# Lorentz factor after first velocity update.
	gamma_minus = GammaFactor(u_minus, c)

	# Rotating the velocity vector in B-field.
	t = (B_field(x_in) / gamma_minus) * (q / m) * (dt / 2)
	s = (2 * t) / (1 + np.linalg.norm(t)**2)
	u_plus = u_minus + np.cross((u_minus + np.cross(u_minus, t)), s)

	# Second velocity update after rotation in B-field.
	u_out = u_plus + E_field(x_in) * (q / m) * (dt / 2)
	return u_out

def VayVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Vay velocity time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	E_field : callable
		User-defined electric field.
	B_field : callable
		User-defined magnetic field.
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

	# First velocity update before rotation in B-field.
	u_half = u_in + (E_field(x_in) + np.cross((u_in / gamma_u_in), B_field(x_in))) * (q / m) * (dt / 2)

	# Rotating the velocity vector in B-field.
	tau = B_field(x_in) * (q / m) * (dt / 2)
	u_prime = u_half + E_field(x_in) * (q / m) * (dt / 2)
	u_star = np.dot(u_prime, tau / c)
	gamma_prime = GammaFactor(u_prime, c)
	sigma = gamma_prime**2 - np.linalg.norm(tau)**2

	# Lorentz gamma factor after rotation in B-field.
	gamma_new = np.sqrt((sigma + np.sqrt(sigma**2 + 4 * (np.linalg.norm(tau)**2 + u_star**2))) / 2)

	# Scaling B-field for second velocity update.
	t = tau / gamma_new
	s = 1 / (1 + np.linalg.norm(t)**2)

	# Second velocity update after rotation in B-field.
	u_out = s * (u_prime + np.dot(u_prime, t) * t + np.cross(u_prime, t))
	return u_out

def HigueraVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Higuera-Cary velocity time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	E_field : callable
		User-defined electric field.
	B_field : callable
		User-defined magnetic field.
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
	# First velocity update before rotation in B-field.
	u_minus = u_in + E_field(x_in) * (q / m) * (dt / 2)

	# Rotating the velocity vector in B-field.
	gamma_minus = GammaFactor(u_minus, c)
	tau = B_field(x_in) * (q / m) * (dt / 2)
	u_star = np.dot(u_minus, tau / c)
	sigma = gamma_minus**2 - np.linalg.norm(tau)**2

	# Lorentz gamma factor after rotation in B-field.
	gamma_plus = np.sqrt((sigma + np.sqrt(sigma**2 + 4 * (np.linalg.norm(tau)**2 + u_star**2))) / 2)

	# Scaling B-field for second velocity update.
	t = tau / gamma_plus
	s = 1 / (1 + np.linalg.norm(t)**2)

	# Intermediate velocity after rotation in B-field.
	u_plus = s * (u_minus + np.dot(u_minus, t) * t + np.cross(u_minus, t))

	# Second velocity update after rotation in B-field.
	u_out = u_plus + E_field(x_in) * (q / m) * (dt / 2) + np.cross(u_plus, t)
	return u_out
