# Importing standard modules.
import os
import time
import numpy as np

# Importing pushers.
from particle_pushers.pushers.explicit_lab_frame_pushers_fourth_order import BorisFourthOrder,\
																			 VayFourthOrder,\
																			 HigueraFourthOrder

# Methods to be compared.
integrators = (BorisFourthOrder,
			   HigueraFourthOrder,
			   VayFourthOrder
			   )

# User-defined E-field.
def Linear_E(x):
    length = len(x)
    E_out = np.zeros(length)
    E_out[0] = -x[0]
    return E_out

# User-defined B-field.
def Sheared_B(x):
    length = len(x)
    B_out = np.zeros(length)
    B_out[0] = 2 * x[1]
    return B_out

# Particle charge and mass.
q_test = 1
m_test = 1

# Speed of light in atomic units.
c = 1

# Initial conditions.
x_in = (1.0, 1.0, 0)
u_in = (np.sqrt(37) / 2, 1.5, 0)

# Time span to integrate over.
times = (0, 1000)

# Calculating time steps.
scales = np.linspace(5e-3, 1e-1, num=20)
step_array = [4 * int(10**np.log10(times[-1] / scale)) for scale in scales]

# Names for each of the output arrays.
output_names = ['x', 'u']

# Root folder to store all relevant output files.
experiment_root_folder = 'experiments/sheared_b_field/'

# Creating new folders to store data and CPU time for this experiment.
data_folder = experiment_root_folder + 'data/'
cpu_time_folder = experiment_root_folder + 'cpu_times/'
if not os.path.exists(data_folder) or not os.path.exists(cpu_time_folder):
    os.makedirs(data_folder)
    os.makedirs(cpu_time_folder)

for integrator in integrators:
    # Extracting the name of each integrator.
    integrator_name = str(integrator.__name__)

    # Creating a directory to store data for each integrator.
    integrator_folder = data_folder + integrator_name + '/'
    if not os.path.exists(integrator_folder):
        os.makedirs(integrator_folder)

    # CPU time filename.
    cpu_time_filename = cpu_time_folder + integrator_name + '_cpu_time.csv'

    # Writing a header line for CPU time file.
    with open(cpu_time_filename, 'w') as file:
        file.write('dt,CPU time (s)\n')

    for num_steps in step_array:        
        # Starting the timer.
        tic = time.process_time()

        # Initialising the integrator.
        method = integrator(x_in,
                            u_in,
                            Linear_E,
                            Sheared_B,
                            q_test,
                            m_test,
                            )

        # Solving the system for the given time span.
        t, x, u = method.solve(t_span=times, N=num_steps)

        # Ending the timer.
        toc = time.process_time()

        # Rounding CPU time.
        cpu_time = np.round(toc - tic, 5)

        # Calculating and rounding the time step.
        diff = (times[1] - times[0]) / num_steps
        dt = np.round(diff, 5)

        # Writing CPU time for each time step to a file.
        with open(cpu_time_filename, 'a') as file:
            file.write('{0},{1}\n'.format(dt, cpu_time))

        # Array to store all simulation outputs.
        outputs = [x, u]

        # Creating the relevant filename and exporting the data.
        for i, output in enumerate(outputs):
            filename = integrator_folder + integrator_name \
                       + '_{0}_dt_{1}.csv'.format(output_names[i], dt)
            np.savetxt(filename, output, delimiter=',')
