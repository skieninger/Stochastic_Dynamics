import numpy as np
from inspect import getsource


class MD_Simulation_1D_model_system:

    def __init__(self, total_steps, potential, derivative_potential, time_step, collision_rate, mass, temperature, kboltz=0.008314, initial_position=0, initial_velocity=0):
        self.potential = potential
        self.derivative_potential = derivative_potential
        self.time_step = time_step
        self.collision_rate = collision_rate
        self.mass = mass
        self.temperature = temperature
        self.total_steps = total_steps
        self.kboltz = kboltz
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.total_time = self.time_step * self.total_steps

    def write_log_file(self, filepath):
        f= open(filepath + "traj_LOG.txt","w+")
        f.write("\n******* USED PARAMETERS ******** \n\n")
        f.write("time step = %s \n" %self.time_step)
        f.write("collision rate = %s \n" %self.collision_rate)
        f.write("mass = %s \n" %self.mass)
        f.write("temperature = %s \n" %self.temperature)
        f.write("total number of steps = %s \n" %self.total_steps)
        f.write("Boltzmann constant = %s \n" %self.kboltz)
        f.write("initial position = %s \n" %self.initial_position)
        f.write("initial velocity = %s \n" %self.initial_velocity)
        f.write("%s" %getsource(self.potential))
        f.write("%s" %getsource(self.derivative_potential))
        f.close()

    def draw_random_number_sequence(self, mean=0, variance=1, length=None):
        if length == None:
            eta = np.random.normal(mean, variance, (self.total_steps))
        else:
            eta = np.random.normal(mean, variance, (length))
        return eta

    def compute_overdamped_Langevin_dynamics(self, random_numbers=np.array([])):
        position = np.zeros((self.total_steps + 1))
        position[0] = self.initial_position
        if random_numbers.size == 0:
            eta = self.draw_random_number_sequence()
        else:
            eta = random_numbers
        # propagate in time
        for k in range(0, self.total_steps):
            position[k + 1] = position[k] - self.derivative_potential(position[k]) * (
                        self.time_step / (self.collision_rate * self.mass)) + np.sqrt(
                (2 * self.kboltz * self.temperature * self.time_step) / (self.collision_rate * self.mass)) * eta[k]
        return position

    def compute_Langevin_dynamics(self, random_numbers=np.array([])):
        position = np.zeros((self.total_steps + 1))
        position[0] = self.initial_position
        velocity = np.zeros((self.total_steps + 1))
        velocity[0] = self.initial_velocity
        # constans for integrator
        tau = 1. / self.collision_rate
        c1 = np.exp(-self.time_step / tau)
        c2 = (1 - c1) * tau
        c3 = np.sqrt((2 * self.kboltz * self.temperature) / tau) * np.sqrt(0.5 * (1 - c1 * c1) * tau)
        if random_numbers.size == 0:
            eta = self.draw_random_number_sequence()
        else:
            eta = random_numbers
        # propagate in time
        for k in range(0, self.total_steps):
            position[k + 1] = position[k] + c1 * velocity[k] * self.time_step - c2 * self.time_step * (
                        self.derivative_potential(position[k]) / self.mass) + (
                                          (c3 * self.time_step) / np.sqrt(self.mass)) * eta[k]
            velocity[k + 1] = (position[k + 1] - position[k]) / self.time_step
        return position, velocity