import numpy as np
from particle_pushers.pusher_template import Pusher
from particle_pushers.operators.explicit_lab_frame_operators import ExplicitLabFramePositionOperator,\
																	BorisVelocityOperator,\
																	VayVelocityOperator,\
																	HigueraVelocityOperator

class Boris(Pusher):
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

		# 2nd-order update scheme.
		x_sub_1 = ExplicitLabFramePositionOperator(x_n, u_n, c, dt / 2)

		# Output values for velocity and position.
		u_out = BorisVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, dt)
		x_out = ExplicitLabFramePositionOperator(x_sub_1, u_out, c, dt / 2)
		return x_out, u_out

class Vay(Pusher):
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

		# 2nd-order update scheme.
		x_sub_1 = ExplicitLabFramePositionOperator(x_n, u_n, c, dt / 2)

		# Output values for velocity and position.
		u_out = VayVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, dt)
		x_out = ExplicitLabFramePositionOperator(x_sub_1, u_out, c, dt / 2)
		return x_out, u_out

class Higuera(Pusher):
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

		# 2nd-order update scheme.
		x_sub_1 = ExplicitLabFramePositionOperator(x_n, u_n, c, dt / 2)

		# Output values for velocity and position.
		u_out = HigueraVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, dt)
		x_out = ExplicitLabFramePositionOperator(x_sub_1, u_out, c, dt / 2)
		return x_out, u_out
