import pytest
from Code import MD_class
import numpy as np

traj1 = MD_class.MD_Simulation_1D_model_system(total_steps=10,
                                               potential=lambda x: (x ** 2 - 1) ** 2,
                                               derivative_potential=lambda x: 4 * x * (x ** 2 - 1),
                                               time_step=0.01,
                                               collision_rate=0.5,
                                               mass=1,
                                               temperature=300)
traj2 = MD_class.MD_Simulation_1D_model_system(total_steps=10,
                                               potential=lambda x: (x ** 2 - 1) ** 2,
                                               derivative_potential=lambda x: 4 * x * (x ** 2 - 1),
                                               time_step=0.01,
                                               collision_rate=0.5,
                                               mass=1,
                                               temperature=300,
                                               kboltz=0.008315,
                                               initial_position=-2,
                                               initial_velocity=3)

def test_input():
    testvalues = np.arange(-4,4,1)
    expected_result_potential = np.array([225,  64,   9,   0,   1,   0,   9,  64])
    expected_result_der_potential = np.array([-240,  -96,  -24,    0,    0,    0,   24,   96])
    assert traj1.total_steps == 10 , "Set total_steps passed incorrectly"
    assert traj1.time_step == 0.01 , "Set time_steps passed incorrectly"
    assert traj1.collision_rate == 0.5 , "Set collision_rate passed incorrectly"
    assert traj1.mass == 1 , "Set mass passed incorrectly"
    assert traj1.temperature == 300 , "Set temperature passed incorrectly"
    assert np.array_equal(traj1.potential(testvalues), expected_result_potential) , "Potential energy funcion passed incorrectly"
    assert np.array_equal(traj1.derivative_potential(testvalues), expected_result_der_potential), "Derivative of Potential energy funcion passed incorrectly"
    assert traj1.kboltz == 0.008314 , "Can't call correct Boltzmann constant"
    assert traj1.initial_position == 0 , "Can't call initial position"
    assert traj1.initial_velocity == 0 , "Can't call initial velocity"
    assert traj2.kboltz == 0.008315 , "Set kboltz passed incorrectly"
    assert traj2.initial_position == -2, "Set initial position passed incorrectly"
    assert traj2.initial_velocity == 3, "Set initial velocity passed incorrectly"

def test_random_number_sequence():
    eta1 = traj1.draw_random_number_sequence()
    eta2 = traj1.draw_random_number_sequence(2, 3, 1000000)
    eta3 = traj1.draw_random_number_sequence(length = 1000000)
    assert len(eta1) == 10
    assert len(eta2) == 1000000
    assert len(traj1.draw_random_number_sequence(mean=2,variance=3)) == 10
    assert round(abs(np.mean(eta3)),1) == 0
    assert round(abs(np.var(eta3)),1) == 1
    assert round(abs(np.mean(eta2)),1) == 2
    assert round(abs(np.var(eta2)),1) == 3**2

def test_compute_overdamped_Langevin_dynamics():
    result = np.array([ 0., -0.403117, -0.469953, -0.249351,  0.038743, -0.274638, -0.007078,  0.266478,  0.766314,  0.287189,  0.357309])
    etatest = np.array([-1.27624853, -0.12609331,  0.79115622,  0.97131991, -1.00194723, 0.91139671,  0.86785655,  1.51975771, -1.59699915,  0.15525502])
    assert len(traj1.compute_overdamped_Langevin_dynamics()) == 10+1
    assert traj1.compute_overdamped_Langevin_dynamics()[0] == 0
    assert traj2.compute_overdamped_Langevin_dynamics()[0] == -2
    assert np.array_equal(np.around(traj1.compute_overdamped_Langevin_dynamics(etatest),6), result)

def test_compute_Langevin_dynamics():
    etatest = np.array([-1.27624853, -0.12609331, 0.79115622, 0.97131991, -1.00194723, 0.91139671, 0.86785655, 1.51975771, -1.59699915, 0.15525502])
    result_positions = np.array([ 0., -0.002011, -0.004211, -0.005155, -0.004566, -0.005561, -0.005117, -0.00331 ,  0.00088 ,  0.002534,  0.004426])
    result_velocities = np.array([ 0., -0.201056, -0.219997, -0.094432,  0.058851, -0.099467, 0.044385,  0.180678,  0.419062,  0.165422,  0.189156])
    x, v = traj1.compute_Langevin_dynamics(etatest)
    assert np.array_equal(np.around(x,6), result_positions)
    assert np.array_equal(np.around(v, 6), result_velocities)
    assert x[0] == 0
    assert v[0] == 0
    assert traj2.compute_Langevin_dynamics()[0][0] == -2
    assert traj2.compute_Langevin_dynamics()[1][0] == 3
