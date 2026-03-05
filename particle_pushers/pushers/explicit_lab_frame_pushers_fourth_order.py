import numpy as np
from particle_pushers.pusher_template import Pusher
from particle_pushers.operators.explicit_lab_frame_operators import ExplicitLabFramePositionOperator,\
																	BorisVelocityOperator,\
																	VayVelocityOperator,\
																	HigueraVelocityOperator

class BorisFourthOrder(Pusher):
	def __init__(self,
				 x0,
				 u0,
				 E_field,
				 B_field,
				 q,
				 m
				 ):

		# Inherit arguments from the base class.
		super().__init__(x0,
						 u0,
						 E_field,
						 B_field,
						 q,
						 m,
						 )
		
		# Coefficients for 4th-order method.
		self.c1 = 1 / (2 * (2 - np.cbrt(2)))
		self.d1 = 1 / (2 - np.cbrt(2))
		self.c2 = (1 - np.cbrt(2)) / (2 * (2 - np.cbrt(2)))
		self.d2 = -np.cbrt(2) / (2 - np.cbrt(2))
		self.c3 = (1 - np.cbrt(2)) / (2 * (2 - np.cbrt(2)))
		self.d3 = 1 / (2 - np.cbrt(2))
		self.c4 = 1 / (2 * (2 - np.cbrt(2)))

	def advance(self):
		# Creating local variables.
		x_n = self.x_n
		u_n = self.u_n
		E_field = self.E_field
		B_field = self.B_field
		q = self.q
		m = self.m
		c = self.c
		dt = self.dt

		# Creating local coefficients.
		c1 = self.c1
		d1 = self.d1
		c2 = self.c2
		d2 = self.d2
		c3 = self.c3
		d3 = self.d3
		c4 = self.c4

		# 4th-order update scheme.
		x_sub_1 = ExplicitLabFramePositionOperator(x_n, u_n, c, c1 * dt)
		u_sub_1 = BorisVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, d1 * dt)
		x_sub_2 = ExplicitLabFramePositionOperator(x_sub_1, u_sub_1, c, c2 * dt)
		u_sub_2 = BorisVelocityOperator(x_sub_2, u_sub_1, E_field, B_field, q, m, c, d2 * dt)
		x_sub_3 = ExplicitLabFramePositionOperator(x_sub_2, u_sub_2, c, c3 * dt)

		# Output values for velocity and position.
		u_out = BorisVelocityOperator(x_sub_3, u_sub_2, E_field, B_field, q, m, c, d3 * dt)
		x_out = ExplicitLabFramePositionOperator(x_sub_3, u_out, c, c4 * dt)
		return x_out, u_out

class VayFourthOrder(Pusher):
	def __init__(self,
				 x0,
				 u0,
				 E_field,
				 B_field,
				 q,
				 m
				 ):

		# Inherit arguments from the base class.
		super().__init__(x0,
						 u0,
						 E_field,
						 B_field,
						 q,
						 m
						 )
		
		# Coefficients for 4th-order method.
		self.c1 = 1 / (2 * (2 - np.cbrt(2)))
		self.d1 = 1 / (2 - np.cbrt(2))
		self.c2 = (1 - np.cbrt(2)) / (2 * (2 - np.cbrt(2)))
		self.d2 = -np.cbrt(2) / (2 - np.cbrt(2))
		self.c3 = (1 - np.cbrt(2)) / (2 * (2 - np.cbrt(2)))
		self.d3 = 1 / (2 - np.cbrt(2))
		self.c4 = 1 / (2 * (2 - np.cbrt(2)))

	def advance(self):
		# Creating local variables.
		x_n = self.x_n
		u_n = self.u_n
		E_field = self.E_field
		B_field = self.B_field
		q = self.q
		m = self.m
		c = self.c
		dt = self.dt

		# Creating local coefficients.
		c1 = self.c1
		d1 = self.d1
		c2 = self.c2
		d2 = self.d2
		c3 = self.c3
		d3 = self.d3
		c4 = self.c4

		# 4th-order update scheme.
		x_sub_1 = ExplicitLabFramePositionOperator(x_n, u_n, c, c1 * dt)
		u_sub_1 = VayVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, d1 * dt)
		x_sub_2 = ExplicitLabFramePositionOperator(x_sub_1, u_sub_1, c, c2 * dt)
		u_sub_2 = VayVelocityOperator(x_sub_2, u_sub_1, E_field, B_field, q, m, c, d2 * dt)
		x_sub_3 = ExplicitLabFramePositionOperator(x_sub_2, u_sub_2, c, c3 * dt)

		# Output values for velocity and position.
		u_out = VayVelocityOperator(x_sub_3, u_sub_2, E_field, B_field, q, m, c, d3 * dt)
		x_out = ExplicitLabFramePositionOperator(x_sub_3, u_out, c, c4 * dt)
		return x_out, u_out

class HigueraFourthOrder(Pusher):
	def __init__(self,
				 x0,
				 u0,
				 E_field,
				 B_field,
				 q,
				 m
				 ):

		# Inherit arguments from the base class.
		super().__init__(x0,
						 u0,
						 E_field,
						 B_field,
						 q,
						 m
						 )
		
		# Coefficients for 4th-order method.
		self.c1 = 1 / (2 * (2 - np.cbrt(2)))
		self.d1 = 1 / (2 - np.cbrt(2))
		self.c2 = (1 - np.cbrt(2)) / (2 * (2 - np.cbrt(2)))
		self.d2 = -np.cbrt(2) / (2 - np.cbrt(2))
		self.c3 = (1 - np.cbrt(2)) / (2 * (2 - np.cbrt(2)))
		self.d3 = 1 / (2 - np.cbrt(2))
		self.c4 = 1 / (2 * (2 - np.cbrt(2)))

	def advance(self):
		# Creating local variables.
		x_n = self.x_n
		u_n = self.u_n
		E_field = self.E_field
		B_field = self.B_field
		q = self.q
		m = self.m
		c = self.c
		dt = self.dt

		# Creating local coefficients.
		c1 = self.c1
		d1 = self.d1
		c2 = self.c2
		d2 = self.d2
		c3 = self.c3
		d3 = self.d3
		c4 = self.c4

		# 4th-order update scheme.
		x_sub_1 = ExplicitLabFramePositionOperator(x_n, u_n, c, c1 * dt)
		u_sub_1 = HigueraVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, d1 * dt)
		x_sub_2 = ExplicitLabFramePositionOperator(x_sub_1, u_sub_1, c, c2 * dt)
		u_sub_2 = HigueraVelocityOperator(x_sub_2, u_sub_1, E_field, B_field, q, m, c, d2 * dt)
		x_sub_3 = ExplicitLabFramePositionOperator(x_sub_2, u_sub_2, c, c3 * dt)

		# Output values for velocity and position.
		u_out = HigueraVelocityOperator(x_sub_3, u_sub_2, E_field, B_field, q, m, c, d3 * dt)
		x_out = ExplicitLabFramePositionOperator(x_sub_3, u_out, c, c4 * dt)
		return x_out, u_out
