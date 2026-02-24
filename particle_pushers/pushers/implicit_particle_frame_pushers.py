import numpy as np
from particle_pushers.pusher_template import Pusher
from particle_pushers.operators.explicit_particle_frame_operators import ExplicitParticleFramePositionOperator
from particle_pushers.operators.implicit_particle_frame_operators import HairerDiscreteGradientVelocityOperator,\
																		 HairerVariationalVelocityOperator

class HairerDiscreteGradient(Pusher):
	def advance(self):
		# Creating local variables.
		x_n = self.x_n
		u_half = self.u_half
		E_field = self.E_field
		phi = self.phi
		B_field = self.B_field
		q = self.q
		m = self.m
		c = self.c
		dt = self.dt

		# Output values for velocity and position.
		x_out = ExplicitParticleFramePositionOperator(x_n, u_half, c, dt)
		u_three_half = HairerDiscreteGradientVelocityOperator(x_out, x_n, u_half, E_field, phi, B_field, q, m, c, dt)
		
		# Calculating the velocity at integer time steps.
		u_out = (u_three_half + u_half) / 2
		# Updating the velocity for the next iteration.
		self.u_half = u_three_half
		return x_out, u_out

class HairerVariational(Pusher):
	def advance(self):
		# Creating local variables.
		x_n = self.x_n
		u_half = self.u_half
		E_field = self.E_field
		phi = self.phi
		B_field = self.B_field
		A_field = self.A_field
		A_prime = self.A_prime
		q = self.q
		m = self.m
		c = self.c
		dt = self.dt

		# Output values for velocity and position.
		x_out = ExplicitParticleFramePositionOperator(x_n, u_half, c, dt)
		u_three_half = HairerVariationalVelocityOperator(x_out, x_n, u_half, E_field, phi, B_field, A_field, A_prime, q, 
														 m, c, dt)

		# Calculating the velocity at integer time steps.
		u_out = (u_three_half + u_half) / 2
		# Updating the velocity for the next iteration.
		self.u_half = u_three_half
		return x_out, u_out
