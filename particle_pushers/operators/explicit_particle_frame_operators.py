import numpy as np
from particle_pushers.utils import GammaFactor,\
								   VectorToSpinor,\
								   SpinorToVector,\
								   ApproxExponential,\
								   ElectromagneticTensor,\
								   CayleyTransform

def ExplicitParticleFramePositionOperator(x_in, u_in, c, dt):
	'''
	Position time evolution operator for all explicit methods
	in the particle's comoving frame.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle initial velocity vector.
	c : float
		Speed of light in vacuum.
	dt : float
		Proper time step.

	Returns
	-------
	x_out : numpy.ndarray
		Position vector at time step dt.
	'''
	x_out = x_in + u_in * dt
	return x_out

def GordonQuadraticVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Gordon-Hafizi quadratic velocity time evolution operator.

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
		Proper time step.

	Returns
	-------
	u_out : numpy.ndarray
		Velocity vector at time step dt.
	'''
	# Calculating initial gamma factor.
	gamma_u_in = GammaFactor(u_in, c)

	# Constructing the electromagnetic field 3-vector.
	F_three_vec = (1 / 2) * (q / m) * ((E_field(x_in) / c) + 1j * B_field(x_in))

	# Converting 3-vectors to 4-vectors.
	u_in_four_vec = np.hstack([c * gamma_u_in, u_in])
	F_four_vec = np.hstack([0, F_three_vec])

	# Constructing spinors from 4-vectors.
	U_in = VectorToSpinor(u_in_four_vec)
	F_spinor = VectorToSpinor(F_four_vec)

	# Approximate time evolution operator.
	time_operator = ApproxExponential(F_spinor * dt)

	# Applying time evolution operator to previous velocity spinor.
	lambda_cross_U = time_operator @ U_in
	lambda_dagger = np.conj(time_operator.T)
	U_out = lambda_cross_U @ lambda_dagger

	# Converting the spinor back into a 4-vector.
	u_out_four_vec = SpinorToVector(U_out)

	# Dropping first index to return a 3-vector.
	u_out = u_out_four_vec[1:]
	return u_out

def GordonExactVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Gordon-Hafizi exact velocity time evolution operator.

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
		Proper time step.

	Returns
	-------
	u_out : numpy.ndarray
		Velocity vector at time step dt.
	'''
	# Calculating initial gamma factor.
	gamma_u_in = GammaFactor(u_in, c)

	# Constructing the electromagnetic field 3-vector.
	F_three_vec = (1 / 2) * (q / m) * ((E_field(x_in) / c) + 1j * B_field(x_in))

	# Calculating the norm of the electromagnetic field 3-vector.
	norm_F = np.sqrt(np.dot(F_three_vec, F_three_vec))

	# Converting 3-vectors to 4-vectors.
	u_in_four_vec = np.hstack([c * gamma_u_in, u_in])
	F_four_vec = np.hstack([0, F_three_vec])

	# Constructing spinors from 4-vectors.
	U_in = VectorToSpinor(u_in_four_vec)
	F_spinor = VectorToSpinor(F_four_vec)

	# Check to determine whether or not we are in a field-free region.
	if norm_F == 0:
		# Time evolution operator in a field-free region (i.e. the identity matrix).
		time_operator = np.eye(2)
	else:
		# Time evolution operator in all other circumstances.
		time_operator = np.cosh(norm_F * dt) * np.eye(2) + np.sinh(norm_F * dt) * (F_spinor / norm_F)

	# Applying time evolution operator to previous velocity spinor.
	lambda_cross_U = time_operator @ U_in
	lambda_dagger = np.conj(time_operator.T)
	U_out = lambda_cross_U @ lambda_dagger

	# Converting the spinor back into a 4-vector.
	u_out_four_vec = SpinorToVector(U_out)

	# Dropping first index to return a 3-vector.
	u_out = u_out_four_vec[1:]
	return u_out

def HairerExplicitVelocityOperator(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Hairer-Lubich-Shi explicit velocity time evolution operator.

	Parameters
	----------
	x_in : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle velocity vector calculated 
		using the stagger operator.
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
		Proper time step.

	Returns
	-------
	u_out : numpy.ndarray
		Velocity vector at time step dt.
	'''
	# Calculating initial gamma factor.
	gamma_u_in = GammaFactor(u_in, c)

	# Constructing input electric and magnetic field vectors.
	E_vector = E_field(x_in)
	B_vector = B_field(x_in)

	# Constructing the electromagnetic field tensor.
	F_in = ElectromagneticTensor(E_vector, B_vector, c)

	# Constructing 4-vector from a 3-vector.
	u_in_four_vec = np.hstack([c * gamma_u_in, u_in])
	# Output velocity 4-vector.
	u_out_four_vec = CayleyTransform(F_in * (q / m) * dt / 2) @ u_in_four_vec.T
	# Converting 4-vector back to a 3-vector.
	u_out = u_out_four_vec.T[1:]
	return u_out
