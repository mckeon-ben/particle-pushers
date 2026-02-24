import numpy as np

def GammaFactor(u_in, c):
	'''
	Returns the Lorentz gamma factor associated with
	velocity u_in.

	Parameters
	----------
	u_in : array_like
		velocity vector.
	c : float
		Speed of light in vacuum.

	Returns
	-------
	gamma : float
		Lorentz gamma factor for velocity u_in.
	'''
	gamma = np.sqrt(1 + (np.linalg.norm(u_in) / c)**2)
	return gamma

def VectorToSpinor(a_in):
	'''
	Function to convert a 4-vector to a spinor
	(i.e. the inverse operation of SpinorToVector).

	Parameters
	----------
	a_in : array_like
		4-vector to be converted to spinor.
		Note: Function will fail if a_in is not a 4-vector!

	Returns
	-------
	A_out : ndarray
		The spinor corresponding to 4-vector a_in.
	'''
	# Check to ensure input argument has correct dimensions.
	if len(a_in) != 4:
		raise ValueError('Input argument must be a 4-vector (i.e. a 1×4 array)!')

	# Array containing Pauli matrices.
	pauli_array = np.array([np.eye(2),
							np.array([[0, 1], [1, 0]]),
							np.array([[0, -1j], [1j, 0]]),
							np.array([[1, 0], [0, -1]])
							])

	# Performing the dot product of the Pauli matrix array with the
	# input 4-vector.
	pauli_swap = np.swapaxes(pauli_array, axis1=0, axis2=-1)
	pauli_dot_a_in = np.swapaxes(pauli_swap * a_in, axis1=-1, axis2=0)

	# Output spinor.
	A_out = np.sum(pauli_dot_a_in, axis=0)
	return A_out

def SpinorToVector(A_in):
	'''
	Function to convert a spinor to a 4-vector
	(i.e. the inverse operation of VectorToSpinor).

	Parameters
	----------
	A_in : array_like
		Spinor to be converted back into 4-vector.
		Note: Function will fail if A_in is not a spinor!

	Returns
	-------
	a_out : ndarray
		The 4-vector corresponding to spinor A_in.
	'''
	# Check to ensure input argument has correct dimensions.
	if A_in.shape != (2, 2):
		raise ValueError('Input argument must be a spinor (i.e. a 2×2 array)!')

	# Array containing Pauli matrices.
	pauli_array = np.array([np.eye(2),
							np.array([[0, 1], [1, 0]]),
							np.array([[0, -1j], [1j, 0]]),
							np.array([[1, 0], [0, -1]])
							])

	# Transforming the spinor back into a 4-vector.
	a_out = np.trace(pauli_array @ A_in, axis1=1, axis2=2).real / 2
	return a_out

def ApproxExponential(F_in):
	'''
	Approximate time evolution operator used by the
	Gordon-Hafizi quadratic method.

	Parameters
	----------
	F_in : array_like
		Electromagnetic field spinor responsible for
		velocity time evolution.
		Note: Function will fail if F_in is not a spinor!

	Returns
	-------
	time_evol : ndarray
		Approximate time evolution operator.
	'''
	# Check to ensure input argument has correct dimensions.
	if F_in.shape != (2, 2):
		raise ValueError('Input argument must be a spinor (i.e. a 2×2 array)!')

	# Calculating the square of the time evolution operator.
	numerator_sq = np.linalg.matrix_power(np.eye(2) + F_in / 2, 2)
	denominator_sq = np.linalg.matrix_power(np.eye(2) - F_in / 2, -2)
	time_evol_sq = numerator_sq @ denominator_sq

	# Calculating the trace, determinant and other auxiliary
	# quantities relevant to the previous matrix.
	trace = np.trace(time_evol_sq)
	det = np.linalg.det(time_evol_sq)
	s = np.sqrt(det)
	t = np.sqrt(trace + 2 * s)

	# Returning the approximate time evolution operator.
	approx_time_evol = (1 / t) * (s * np.eye(2) + time_evol_sq)
	return approx_time_evol

def DiscreteGradientElectricField(x_in_2, x_in_1, E_field, phi):
	'''
	Function to construct the modified electric
	field for use with the Hairer-Lubich-Shi
	discrete gradient method.

	Parameters
	----------
	x_in_2 : array_like
		Test particle position vector calculated
		using stagger function.
	x_in_1 : array_like
		Test particle initial position vector.
	E_field : callable
		User-defined electric field.
	phi : callable
		User-defined scalar potential corresponding
		to the electric field defined by E_field.

	Returns
	-------
	E_bar : numpy.ndarray
		Electromagnetic field vector calculated via
		the discrete gradient method.
	'''
	# Calculating the average and difference of input vectors.
	x_bar = (x_in_2 + x_in_1) / 2
	delta_x = x_in_2 - x_in_1

	# Modified E-field constructed via the discrete gradient method.
	E_bar = ((phi(x_in_1) - phi(x_in_2) - np.dot(E_field(x_bar), delta_x)) / np.linalg.norm(delta_x)**2) * delta_x \
	+ E_field(x_bar)
	return E_bar

def ElectromagneticTensor(E_field, B_field, c):
	'''
	Electromagnetic field tensor constructed from
	electric and magnetic field vectors.

	Parameters
	----------
	E_field : array_like
		Electric field vector.
	B_field : array_like
		Magnetic field vector.
	c : float
		Speed of light in vacuum.

	Returns
	-------
	F_out : numpy.ndarray
		Electromagnetic field tensor corresponding
		to vectors E_field and B_field.
	'''
	# Initialising the electromagnetic tensor.
	F_out = np.zeros([4, 4])

	# Allocating the electric field elements.
	F_out[0, 1:] = E_field / c
	F_out[1:, 0] = F_out[0, 1:].T

	# Allocating the magnetic field elements.
	for i in range(1, 4):
		F_out[1:, i] = np.cross(np.eye(3)[:, i - 1], B_field)
	return F_out

def CovariantDerivative(E_field, A_prime, c):
	'''
	Covariant derivative constructed from
	electric field vector and the Jacobian
	of the magnetic vector potential.

	Parameters
	----------
	E_field : array_like
		Electric field vector.
	A_prime : array_like
		Jacobian of magnetic vector potential.
	c : float
		Speed of light in vacuum.

	Returns
	-------
	A_dash_out : numpy.ndarray
		Electromagnetic field tensor corresponding
		to electric field vector and magnetic
		vector potential Jacobian:
	'''
	# Initialising the covariant derivative.
	A_dash_out = np.zeros([4, 4])

	# Allocating the electric field elements.
	A_dash_out[0, 1:] = -E_field / c
	
	# Allocating the magnetic vector potential
	# Jacobian elements.
	A_dash_out[1:, 1:] = A_prime
	return A_dash_out

def Stagger(x_in, u_in, E_field, B_field, q, m, c, dt):
	'''
	Function to stagger velocity by half a time 
	step before beginning time evolution.

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
		Proper time step.

	Returns
	-------
	u_half : numpy.ndarray
		Velocity vector at time step dt / 2.
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
	u_half_four_vec = u_in_four_vec @ (np.eye(4) + F_in.T * (q / m) * dt / 2)

	# Converting 4-vector back to a 3-vector.
	u_half = u_half_four_vec[1:]
	return u_half

def CayleyTransform(A_in):
	'''
	Function to compute the Cayley transform of
	electromagnetic field tensor for use with
	all Hairer-Lubich-Shi methods.

	Parameters
	----------
	A_in : array_like
		Electromagnetic tensor to be transformed.
		Note: Function will fail if A_in is not a 4×4 matrix!

	Returns
	-------
	A_out : ndarray
		Transformed electromagnetic field tensor.
	'''
	# Check to ensure input argument has correct dimensions.
	if A_in.shape != (4, 4):
		raise ValueError('Input argument must be a 4×4 array!')

	A_out = np.linalg.inv(np.eye(4) - A_in) @ (np.eye(4) + A_in)
	return A_out
