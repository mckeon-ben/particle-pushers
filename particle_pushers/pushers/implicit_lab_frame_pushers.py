import numpy as np
from particle_pushers.pusher_template import Pusher
from particle_pushers.operators.implicit_lab_frame_operators import LapentaPositionOperator,\
																	PetriPositionOperator,\
																	DiscreteGradientPositionOperator,\
														 			LapentaVelocityOperator,\
														 			PetriVelocityOperator,\
														 			DiscreteGradientVelocityOperator

class Lapenta(Pusher):
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

		# Output values for velocity and position.
		u_out = LapentaVelocityOperator(x_n, u_n, E_field, B_field, q, m, c, dt)
		x_out = LapentaPositionOperator(x_n, u_n, u_out, c, dt)
		return x_out, u_out

class Petri(Pusher):
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

		# Output values for velocity and position.
		u_out = PetriVelocityOperator(x_n, u_n, E_field, B_field, q, m, c, dt)
		x_out = PetriPositionOperator(x_n, u_n, u_out, c, dt)
		return x_out, u_out

class DiscreteGradient(Pusher):
	def advance(self):
		# Creating local variables.
		x_n = self.x_n
		u_n = self.u_n
		E_field = self.E_field
		phi = self.phi
		B_field = self.B_field
		q = self.q
		m = self.m
		c = self.c
		dt = self.dt

		# Output values for velocity and position.
		u_out = DiscreteGradientVelocityOperator(x_n, u_n, E_field, phi, B_field, q, m, c, dt)
		x_out = DiscreteGradientPositionOperator(x_n, u_n, u_out, c, dt)
		return x_out, u_out
