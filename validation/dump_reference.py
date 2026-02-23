#!/usr/bin/env python3
"""
Generate JSON reference data for C++ validation tests.

Runs the Python simulation components with fixed seeds and dumps
intermediate results that the Catch2 tests can load and compare against.
"""

import json
import sys
import os
import numpy as np

# Add the experiment directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', '..', '..', 'experment', 'magnetic-bugs'))

from compass import CompassSensor, singlet_yield
from ring_attractor import RingAttractor
from landscape import Landscape
from path_integration import CPU4
from agent import Bug


def to_list(x):
    """Convert numpy arrays/scalars to JSON-serialisable lists."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def dump_compass_reference(outdir):
    """Test CompassSensor with known inputs."""
    rng = np.random.default_rng(42)

    # Test singlet_yield function
    alphas = np.linspace(0, 2 * np.pi, 32)
    yields_c015 = singlet_yield(alphas, contrast=0.15, mean_yield=0.5)
    yields_c001 = singlet_yield(alphas, contrast=0.01, mean_yield=0.5)

    # Test CompassSensor.read at several headings
    sensor = CompassSensor(n_cry=1000, n_channels=8, contrast=0.15,
                           mean_yield=0.5, sigma_sensor=0.02, rng=rng)
    headings = [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
    readings = {}
    for h in headings:
        # Reset RNG for reproducibility per reading
        readings[f"{h:.6f}"] = to_list(sensor.read(h))

    # Noiseless sensor for deterministic validation
    sensor_noiseless = CompassSensor(n_cry=1000, n_channels=8, contrast=0.15,
                                      mean_yield=0.5, sigma_sensor=0.0,
                                      rng=np.random.default_rng(0))
    noiseless_readings = {}
    for h in headings:
        noiseless_readings[f"{h:.6f}"] = to_list(sensor_noiseless.read(h))

    data = {
        'singlet_yield': {
            'alphas': to_list(alphas),
            'yields_c015': to_list(yields_c015),
            'yields_c001': to_list(yields_c001),
        },
        'sensor_readings': readings,
        'noiseless_readings': noiseless_readings,
        'sensor_params': {
            'n_cry': 1000,
            'n_channels': 8,
            'contrast': 0.15,
            'mean_yield': 0.5,
            'sigma_sensor': 0.02,
        },
    }

    with open(os.path.join(outdir, 'compass_reference.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print("  compass_reference.json written")


def dump_ring_attractor_reference(outdir):
    """Test RingAttractor evolution with known inputs."""
    rng = np.random.default_rng(123)
    ra = RingAttractor(n=8, tau=0.05, w_exc=1.5, w_inh=4.5,
                       g_mag=2.0, g_omega=0.5, noise_sigma=0.0, rng=rng)

    # Initialise bump at heading = 0
    ra.reset(0.0)
    initial_state = to_list(ra.r.copy())
    initial_heading = float(ra.heading())

    # Run 100 steps with no input — bump should remain stable
    dt = 0.01
    states_no_input = [to_list(ra.r.copy())]
    headings_no_input = [float(ra.heading())]
    for _ in range(100):
        ra.step(dt)
        states_no_input.append(to_list(ra.r.copy()))
        headings_no_input.append(float(ra.heading()))

    # Reset and run with compass input driving toward pi/2
    ra.reset(0.0)
    # Create a compass signal peaked at pi/2
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    compass_at_pi2 = 0.5 + 0.075 * (1.0 + np.cos(2.0 * (theta - np.pi / 2)))

    states_compass = [to_list(ra.r.copy())]
    headings_compass = [float(ra.heading())]
    for _ in range(200):
        ra.step(dt, compass_input=compass_at_pi2)
        states_compass.append(to_list(ra.r.copy()))
        headings_compass.append(float(ra.heading()))

    # Reset and run with angular velocity input
    ra.reset(0.0)
    omega = 1.0  # rad/s
    states_omega = [to_list(ra.r.copy())]
    headings_omega = [float(ra.heading())]
    for _ in range(100):
        ra.step(dt, angular_velocity=omega)
        states_omega.append(to_list(ra.r.copy()))
        headings_omega.append(float(ra.heading()))

    data = {
        'params': {
            'n': 8, 'tau': 0.05, 'w_exc': 1.5, 'w_inh': 4.5,
            'g_mag': 2.0, 'g_omega': 0.5, 'noise_sigma': 0.0,
        },
        'dt': dt,
        'initial_state': initial_state,
        'initial_heading': initial_heading,
        'no_input': {
            'states': states_no_input,
            'headings': headings_no_input,
        },
        'compass_drive': {
            'compass_input': to_list(compass_at_pi2),
            'states': states_compass,
            'headings': headings_compass,
        },
        'angular_velocity': {
            'omega': omega,
            'states': states_omega,
            'headings': headings_omega,
        },
    }

    with open(os.path.join(outdir, 'ring_attractor_reference.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print("  ring_attractor_reference.json written")


def dump_cpu4_reference(outdir):
    """Test CPU4 path integration with known inputs."""
    cpu4 = CPU4(n=8, leak=0.0, gain=1.0)

    dt = 0.01
    # Walk North (heading=0) for 100 steps at speed 1
    for _ in range(100):
        cpu4.update(heading=0.0, speed=1.0, dt=dt)
    north_memory = to_list(cpu4.memory.copy())
    north_dist, north_dir = cpu4.home_vector()
    north_dx, north_dy = cpu4.displacement()

    # Continue East (heading=pi/2) for 100 steps
    for _ in range(100):
        cpu4.update(heading=np.pi / 2, speed=1.0, dt=dt)
    ne_memory = to_list(cpu4.memory.copy())
    ne_dist, ne_dir = cpu4.home_vector()
    ne_dx, ne_dy = cpu4.displacement()

    # Test leaky integrator
    cpu4_leaky = CPU4(n=8, leak=0.1, gain=1.0)
    for _ in range(100):
        cpu4_leaky.update(heading=0.0, speed=1.0, dt=dt)
    leaky_memory = to_list(cpu4_leaky.memory.copy())
    leaky_dist, leaky_dir = cpu4_leaky.home_vector()

    data = {
        'params': {'n': 8, 'leak': 0.0, 'gain': 1.0},
        'dt': dt,
        'after_north': {
            'memory': north_memory,
            'home_distance': float(north_dist),
            'home_direction': float(north_dir),
            'displacement': [float(north_dx), float(north_dy)],
        },
        'after_north_east': {
            'memory': ne_memory,
            'home_distance': float(ne_dist),
            'home_direction': float(ne_dir),
            'displacement': [float(ne_dx), float(ne_dy)],
        },
        'leaky': {
            'params': {'n': 8, 'leak': 0.1, 'gain': 1.0},
            'memory': leaky_memory,
            'home_distance': float(leaky_dist),
            'home_direction': float(leaky_dir),
        },
    }

    with open(os.path.join(outdir, 'cpu4_reference.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print("  cpu4_reference.json written")


def dump_landscape_reference(outdir):
    """Test Landscape magnetic field computation."""
    # Uniform field
    land = Landscape(extent=(1000, 1000), B0=50.0, declination=0.0,
                     inclination=np.radians(65.0))
    d0, i0, inc0 = land.magnetic_direction(500.0, 500.0)

    # With dipole anomaly
    anom = [{'type': 'dipole', 'pos': (500, 500), 'strength': 5.0, 'depth': 50.0}]
    land_anom = Landscape(extent=(1000, 1000), B0=50.0, declination=0.0,
                          inclination=np.radians(65.0), anomalies=anom)

    # Sample points around the dipole
    test_pts = [(400, 400), (500, 500), (600, 400), (500, 600), (450, 500)]
    anom_results = {}
    for (px, py) in test_pts:
        d, i, inc = land_anom.magnetic_direction(float(px), float(py))
        dev = land_anom.direction_deviation(float(px), float(py))
        anom_results[f"{px},{py}"] = {
            'direction': float(d),
            'intensity': float(i),
            'inclination': float(inc),
            'deviation': float(dev),
        }

    # With fault anomaly
    fault = [{'type': 'fault', 'pos': (500, 500), 'azimuth': 0.0,
              'contrast': 3.0, 'width': 50.0}]
    land_fault = Landscape(extent=(1000, 1000), B0=50.0, declination=0.0,
                           inclination=np.radians(65.0), anomalies=fault)
    fault_results = {}
    for (px, py) in test_pts:
        d, i, inc = land_fault.magnetic_direction(float(px), float(py))
        fault_results[f"{px},{py}"] = {
            'direction': float(d),
            'intensity': float(i),
            'inclination': float(inc),
        }

    # With gradient anomaly
    grad = [{'type': 'gradient', 'magnitude': 0.01, 'direction': 0.0}]
    land_grad = Landscape(extent=(1000, 1000), B0=50.0, declination=0.0,
                          inclination=np.radians(65.0), anomalies=grad)
    grad_results = {}
    for (px, py) in test_pts:
        d, i, inc = land_grad.magnetic_direction(float(px), float(py))
        grad_results[f"{px},{py}"] = {
            'direction': float(d),
            'intensity': float(i),
            'inclination': float(inc),
        }

    data = {
        'uniform': {
            'params': {'B0': 50.0, 'declination': 0.0,
                       'inclination_deg': 65.0},
            'centre': {
                'direction': float(d0),
                'intensity': float(i0),
                'inclination': float(inc0),
            },
        },
        'dipole': {
            'anomaly': anom[0],
            'results': anom_results,
        },
        'fault': {
            'anomaly': fault[0],
            'results': fault_results,
        },
        'gradient': {
            'anomaly': grad[0],
            'results': grad_results,
        },
    }

    with open(os.path.join(outdir, 'landscape_reference.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print("  landscape_reference.json written")


def dump_bug_reference(outdir):
    """Test full Bug agent for a short trajectory."""
    # Simple landscape, no anomalies
    land = Landscape(extent=(1000, 1000), B0=50.0, declination=0.0,
                     inclination=np.radians(65.0))

    bug = Bug(x0=500, y0=100, heading0=0.0,
              goal_heading=3 * np.pi / 4,
              speed=1.0, kappa=2.0,
              sigma_theta=0.0, sigma_xy=0.0,
              compass_params={'n_cry': 1000, 'n_channels': 8,
                              'contrast': 0.15, 'sigma_sensor': 0.0},
              attractor_params={'noise_sigma': 0.0},
              seed=42)

    # Run for 200 steps (deterministic)
    dt = 0.01
    for _ in range(200):
        bug.step(dt, land)

    # Save trajectory at sampled points
    indices = [0, 50, 100, 150, 200]
    trajectory = {}
    for i in indices:
        trajectory[str(i)] = {
            'x': float(bug.history['x'][i]),
            'y': float(bug.history['y'][i]),
            'heading': float(bug.history['heading'][i]),
            'estimated_heading': float(bug.history['estimated_heading'][i]),
        }

    data = {
        'params': {
            'x0': 500.0, 'y0': 100.0, 'heading0': 0.0,
            'goal_heading': 3 * np.pi / 4,
            'speed': 1.0, 'kappa': 2.0,
            'sigma_theta': 0.0, 'sigma_xy': 0.0,
            'seed': 42,
        },
        'landscape': {
            'B0': 50.0, 'declination': 0.0, 'inclination_deg': 65.0,
        },
        'dt': dt,
        'n_steps': 200,
        'trajectory': trajectory,
    }

    with open(os.path.join(outdir, 'bug_reference.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print("  bug_reference.json written")


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'reference')
    os.makedirs(outdir, exist_ok=True)

    print("Generating reference data for C++ validation...")
    dump_compass_reference(outdir)
    dump_ring_attractor_reference(outdir)
    dump_cpu4_reference(outdir)
    dump_landscape_reference(outdir)
    dump_bug_reference(outdir)
    print("Done.")
