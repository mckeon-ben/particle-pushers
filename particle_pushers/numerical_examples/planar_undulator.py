'''
Planar undulator.

Natural units, c = 1, with k = 2 pi so the undulator period is unity.
'''

import json
import os
import time

import numpy as np
from particle_pushers.particle import Particle
from particle_pushers.field import StaticField
from particle_pushers.lab_frame import (
    Boris, Vay, Higuera,
    BorisOrderFour, VayOrderFour, HigueraOrderFour,
)
from particle_pushers.comoving_frame import (
    GordonQuadraticLab, GordonExactLab,
    GordonQuadraticLabOrderFour, GordonExactLabOrderFour
)


# Written into the data folder.
DATA_DIR = 'data'
DATA_FILE = os.path.join(DATA_DIR, 'planar_undulator.json')

# Names the figure.
EXPERIMENT = 'Planar undulator'

# Method sets, grouped by the state representation they carry, so the
# spatial parts can be extracted consistently.
LAB_FRAME_METHODS_O2 = {
    'Boris': Boris,
    'Vay': Vay,
    'Higuera-Cary': Higuera,
}
GORDON_METHODS_O2 = {
    'Gordon-Hafizi (quadratic)': GordonQuadraticLab,
    'Gordon-Hafizi (exact)': GordonExactLab,
}

# Fourth-order (Yoshida triple-jump) counterparts. The display names
# match the second-order ones so the two columns of the figure line up;
# the class name is stored in the data file to disambiguate.
LAB_FRAME_METHODS_O4 = {
    'Boris': BorisOrderFour,
    'Vay': VayOrderFour,
    'Higuera-Cary': HigueraOrderFour,
}
GORDON_METHODS_O4 = {
    'Gordon-Hafizi (quadratic)': GordonQuadraticLabOrderFour,
    'Gordon-Hafizi (exact)': GordonExactLabOrderFour,
}

# One entry per figure column: label, method sets, nominal order, step counts.
FAMILIES = [
    ('2nd order', LAB_FRAME_METHODS_O2, GORDON_METHODS_O2, 2,
     [2048, 4096, 8192, 16384, 32768]),
    ('4th order', LAB_FRAME_METHODS_O4, GORDON_METHODS_O4, 4,
     [1024, 2048, 4096, 8192, 16384]),
]


# Undulator strength and period. K = B0 / k is the undulator
# parameter; below one the wiggle is a small perturbation on the
# forward motion, which is the usual operating regime.
B0 = 1.0
K_WAVE = 2.0 * np.pi
Q, M = 1.0, 1.0

# Launch state: proton injected along the axis.
X_START = np.zeros(3)
U_START = np.array([0.0, 0.0, 5.0])

# Final lab time.
T = 32.0


def undulator_field(b0=B0, k=K_WAVE):
    '''Planar undulator, an exact vacuum field.'''

    def B_func(x):
        return np.array([0.0,
                         b0 * np.cosh(k * x[1]) * np.cos(k * x[2]),
                         -b0 * np.sinh(k * x[1]) * np.sin(k * x[2])])

    return StaticField(B_func=B_func)


def _spatial(x, u, is_gordon):
    '''Spatial 3-vectors (x, u), whichever convention the method uses.

    Gordon methods carry 4-vectors with lab time in x[0] and gamma in
    u[0]; lab-frame methods carry bare 3-vectors. Dropping the zeroth
    slot puts both on the same footing.
    '''
    if is_gordon:
        return x[1:], u[1:]
    return x, u


def final_state(method_cls, x0, u0, q, m, field, T, N, is_gordon):
    '''Integrate one pusher over [0, T] with N steps; return final (x, u).'''
    particle = Particle(x=x0.copy(), u=u0.copy(), q=q, m=m)
    pusher = method_cls(particle, field)
    _, x_out, u_out = pusher.solve((0.0, T), N)
    return _spatial(x_out[-1], u_out[-1], is_gordon)


def run_method(method_cls, x0, u0, q, m, field, T, n_list, is_gordon):
    '''Final states for one method at every step count, shape (len, 3).'''
    finals = [final_state(method_cls, x0, u0, q, m, field, T, N, is_gordon)
              for N in n_list]
    return (np.array([f[0] for f in finals]),
            np.array([f[1] for f in finals]))


def run_family(field, x0_3, u0_3, q, m, T, n_list, lab_methods,
               gordon_methods, label='', order=2):
    '''Run one family and return its serialisable record.

    Lab-frame methods take the 3-vector state; Gordon methods take the
    4-vector lift (lab time in x[0], gamma in u[0]).
    '''
    gamma0 = np.sqrt(1.0 + u0_3 @ u0_3)
    x0_4 = np.hstack([0.0, x0_3])
    u0_4 = np.hstack([gamma0, u0_3])

    methods = {}
    todo = ([(n, c, False) for n, c in lab_methods.items()]
            + [(n, c, True) for n, c in gordon_methods.items()])
    for name, cls, is_gordon in todo:
        x0, u0 = (x0_4, u0_4) if is_gordon else (x0_3, u0_3)
        t0 = time.perf_counter()
        xs, us = run_method(cls, x0, u0, q, m, field, T, n_list, is_gordon)
        methods[name] = {
            'class': cls.__name__,
            'frame': 'comoving' if is_gordon else 'lab',
            'x': xs.tolist(), 'u': us.tolist(),
        }
        print(f'  {label:>9s}  {name:<28s} {cls.__name__:<28s} '
              f'{time.perf_counter() - t0:6.1f}s', flush=True)

    return {'label': label, 'order': order,
            'dt': [T / N for N in n_list],
            'method_order': list(lab_methods) + list(gordon_methods),
            'methods': methods}


def main(filename=DATA_FILE, q=Q, m=M):
    field = undulator_field()
    x0_3, u0_3 = X_START.copy(), U_START.copy()

    print('\n' + EXPERIMENT)
    print('-' * 80)
    print(f'{"Family":>11s}  {"Method":<28s} {"Class":<28s} {"Time":>7s}')
    print('-' * 80)
    families = [
        run_family(field, x0_3, u0_3, q, m, T, n_list, lab_m, gor_m,
                   label=label, order=order)
        for label, lab_m, gor_m, order, n_list in FAMILIES
    ]

    record = {
        'schema': 1,
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'experiment': EXPERIMENT,
        'parameters': {
            'B0': B0, 'k': K_WAVE, 'K': B0 / K_WAVE,
            'x_start': X_START.tolist(),
            'u_start': U_START.tolist(),
            'q': q, 'm': m, 'T': T,
            'N_list': {lab: n for lab, *_, n in FAMILIES},
        },
        'initial_state': {'x': x0_3.tolist(), 'u': u0_3.tolist()},
        'families': families,
    }
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filename, 'w') as fh:
        json.dump(record, fh, indent=1)
    print(f'\nWrote {filename}')
    return record


if __name__ == '__main__':
    main()
