import numpy as np
from scipy.optimize import fsolve
from particle_pushers.utils import GammaFactor,\
								   DiscreteGradientElectricField,\
								   ElectromagneticTensor,\
								   CovariantDerivative,\
								   CayleyTransform

def HairerDiscreteGradientVelocityOperator(x_in_2, x_in_1, u_in, E_field, phi, B_field, q, m, c, dt):
	'''
	Hairer-Lubich-Shi discrete gradient velocity time evolution operator.

	Parameters
	----------
	x_in_2 : array_like
		Test particle position vector calculated
		during current time step.
	x_in_1 : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle velocity vector calculated
		using stagger function.
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
		Proper time step.

	Returns
	-------
	u_out : numpy.ndarray
		Velocity vector at time step dt.
	'''
	# Calculating initial gamma factor.
	gamma_u_in = GammaFactor(u_in, c)

	def HairerDiscreteGradientResidual(u_k):
		# Calculating residual gamma factor.
		gamma_k = GammaFactor(u_k, c)

		# Constructing 4-vectors from 3-vectors.
		u_in_four_vec = np.hstack([c * gamma_u_in, u_in])
		u_k_four_vec = np.hstack([c * gamma_k, u_k])

		# Defining the position vector at both the previous and next half steps.
		x_prev_half = (x_in_2 + x_in_1) / 2
		x_next_half = x_in_2 + u_k * (dt / 2)

		# Calculating the modified electric field vector via the discrete gradient.
		E_bar_vector = DiscreteGradientElectricField(x_next_half, x_prev_half, E_field, phi)

		# Magnetic field vector calculated as normal.
		B_vector = B_field(x_in_2)

		# Constructing the electromagnetic field tensor.
		F_bar = ElectromagneticTensor(E_bar_vector, B_vector, c)

		# Calculating the residual to be solved.
		residual = u_k_four_vec.T - CayleyTransform(F_bar * (q / m) * dt / 2) @ u_in_four_vec.T

		# Converting residual 4-vector back to a 3-vector.
		return residual[1:].T

	# Solving for the via Newton's method.
	u_out = fsolve(func=HairerDiscreteGradientResidual, x0=u_in)
	return u_out

def HairerVariationalVelocityOperator(x_in_2, x_in_1, u_in, E_field, phi, B_field, A_field, A_prime, q, m, c, dt):
	'''
	Hairer-Lubich-Shi variational velocity time evolution operator.

	Parameters
	----------
	x_in_2 : array_like
		Test particle position vector calculated
		during current time step.
	x_in_1 : array_like
		Test particle initial position vector.
	u_in : array_like
		Test particle velocity vector calculated
		using stagger function.
	E_field : callable
		User-defined position-dependent electric field.
	phi : callable
		User-defined scalar potential corresponding
		to the electric field defined by E_field.
	B_field : callable
		User-defined position-dependent magnetic field.
	A_field : callable
		User-defined vector potential corresponding
		to the magnetic field defined by B_field.
	A_prime : callable
		Jacobian matrix corresponding to the vector
		potential A_field
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

	def HairerVariationalResidual(u_k):
		# Calculating residual gamma factor.
		gamma_k = GammaFactor(u_k, c)
		
		# Output position defined implicitly.
		x_out_imp = x_in_2 + u_k * dt

		# Constructing velocity 4-vectors from 3-vectors.
		u_in_four_vec = np.hstack([c * gamma_u_in, u_in])
		u_k_four_vec = np.hstack([c * gamma_k, u_k])

		# Constructing potential 4-vectors from 3-vectors.
		A_in_four_vec = np.hstack([phi(x_in_1), A_field(x_in_1)])
		A_k_four_vec = np.hstack([phi(x_out_imp), A_field(x_out_imp)])
		
		# Average potential.
		A_bar = (A_k_four_vec - A_in_four_vec) / 2

		# Constructing input electric and magnetic field vectors.
		E_vector = E_field(x_in_2)
		B_vector = B_field(x_in_2)
		
		# Constructing magnetic vector potential Jacobian matrix.
		A_Jacobian = A_prime(x_in_2)

		# Constructing the electromagnetic field tensor.
		F_in = ElectromagneticTensor(E_vector, B_vector, c)

		# Constructing the covariant derivative of A.
		A_cov_deriv = CovariantDerivative(E_vector, A_Jacobian, c)
		
		# Modified electromagnetic tensor.
		F_mod = F_in + A_cov_deriv

		# Calculating the residual to be solved.
		residual = u_k_four_vec.T - CayleyTransform(F_mod * (q / m) * dt / 2) @ u_in_four_vec.T \
		+ np.linalg.inv(np.eye(4) - F_mod * (q / m) * dt / 2) @ A_bar.T

		# Converting residual 4-vector back to a 3-vector.
		return residual[1:].T

	# Solving for the via Newton's method.
	u_out = fsolve(func=HairerVariationalResidual, x0=u_in)
	return u_out
