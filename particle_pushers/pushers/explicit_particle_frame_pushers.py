import numpy as np
from particle_pushers.pusher_template import Pusher
from particle_pushers.operators.explicit_particle_frame_operators import ExplicitParticleFramePositionOperator,\
																		 GordonQuadraticVelocityOperator,\
																		 GordonExactVelocityOperator,\
																		 HairerExplicitVelocityOperator

class GordonQuadratic(Pusher):
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
		x_sub_1 = ExplicitParticleFramePositionOperator(x_n, u_n, c, dt / 2)

		# Output values for velocity and position.
		u_out = GordonQuadraticVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, dt)
		x_out = ExplicitParticleFramePositionOperator(x_sub_1, u_out, c, dt / 2)
		return x_out, u_out

class GordonExact(Pusher):
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
		x_sub_1 = ExplicitParticleFramePositionOperator(x_n, u_n, c, dt / 2)

		# Output values for velocity and position.
		u_out = GordonExactVelocityOperator(x_sub_1, u_n, E_field, B_field, q, m, c, dt)
		x_out = ExplicitParticleFramePositionOperator(x_sub_1, u_out, c, dt / 2)
		return x_out, u_out

class HairerExplicit(Pusher):
	def advance(self):
		# Creating local variables.
		x_n = self.x_n
		u_half = self.u_half
		E_field = self.E_field
		B_field = self.B_field
		q = self.q
		m = self.m
		c = self.c
		dt = self.dt

		# Output values for velocity and position.
		x_out = ExplicitParticleFramePositionOperator(x_n, u_half, c, dt)
		u_three_half = HairerExplicitVelocityOperator(x_out, u_half, E_field, B_field, q, m, c, dt)

		# Calculating the velocity at integer time steps.
		u_out = (u_three_half + u_half) / 2
		# Updating the velocity for the next iteration.
		self.u_half = u_three_half
		return x_out, u_out
