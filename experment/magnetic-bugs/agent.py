"""
The navigating bug.

Integrates the radical-pair compass sensor, ring attractor neural compass,
and Langevin locomotion into a single agent that navigates a 2D landscape
toward a magnetic heading goal.

Equations of motion (Euler–Maruyama):

    θ_{t+1} = θ_t + κ sin(θ_goal - θ̂_t) Δt + σ_θ √Δt η_θ
    x_{t+1} = x_t + v cos(θ_t) Δt + σ_x √Δt η_x
    y_{t+1} = y_t + v sin(θ_t) Δt + σ_y √Δt η_y

where θ̂ is the heading estimate from the ring attractor.
"""

import numpy as np
from compass import CompassSensor
from ring_attractor import RingAttractor
from landscape import Landscape


class Bug:
    """A magnetically navigating bug.

    Parameters
    ----------
    x0, y0 : float
        Initial position (body-lengths).
    heading0 : float
        Initial heading (rad, 0 = North, π/2 = East).
    goal_heading : float
        Desired magnetic heading (rad). Default 3π/4 ≈ SW.
    speed : float
        Forward speed (body-lengths / s).
    kappa : float
        Steering gain (rad/s). How strongly the bug corrects toward goal.
    sigma_theta : float
        Angular noise intensity (rad / √s).
    sigma_xy : float
        Translational noise intensity (BL / √s).
    compass_params : dict
        Keyword arguments for CompassSensor.
    attractor_params : dict
        Keyword arguments for RingAttractor.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, x0=500, y0=100, heading0=None,
                 goal_heading=3 * np.pi / 4,
                 speed=1.0, kappa=2.0,
                 sigma_theta=0.1, sigma_xy=0.05,
                 compass_params=None, attractor_params=None,
                 seed=None):
        self.rng = np.random.default_rng(seed)

        # State
        self.x = x0
        self.y = y0
        self.heading = heading0 if heading0 is not None else self.rng.uniform(0, 2 * np.pi)
        self.speed = speed

        # Goal
        self.goal_heading = goal_heading

        # Noise
        self.kappa = kappa
        self.sigma_theta = sigma_theta
        self.sigma_xy = sigma_xy

        # Compass sensor
        cp = compass_params or {}
        cp.setdefault('rng', self.rng)
        self.compass = CompassSensor(**cp)

        # Ring attractor
        ap = attractor_params or {}
        ap.setdefault('n', self.compass.n_channels)
        ap.setdefault('rng', self.rng)
        self.attractor = RingAttractor(**ap)

        # Initialise attractor bump near the actual heading
        self.attractor.reset(self.heading)

        # History (for trajectory plotting)
        self.history = {
            'x': [self.x],
            'y': [self.y],
            'heading': [self.heading],
            'estimated_heading': [self.attractor.heading()],
            'bump_amplitude': [self.attractor.bump_amplitude()],
        }

    def step(self, dt, landscape):
        """Advance the bug by one timestep.

        Parameters
        ----------
        dt : float
            Timestep (seconds).
        landscape : Landscape
            The environment providing the magnetic field.

        Returns
        -------
        bool
            True if the bug is still in bounds.
        """
        # 1. Read the local magnetic field
        mag_dir, _, _ = landscape.magnetic_direction(self.x, self.y)

        # 2. Compute the heading relative to the local field
        relative_heading = self.heading - mag_dir

        # 3. Read compass sensor
        compass_signal = self.compass.read(relative_heading)

        # 4. Compute angular velocity from previous heading change
        # (simplified: use the steering command as angular velocity)
        estimated_heading = self.attractor.heading() + mag_dir
        heading_error = self.goal_heading - estimated_heading
        angular_command = self.kappa * np.sin(heading_error)

        # 5. Update ring attractor
        self.attractor.step(dt, compass_input=compass_signal,
                           angular_velocity=angular_command)

        # 6. Steer: update heading
        noise_theta = self.sigma_theta * np.sqrt(dt) * self.rng.standard_normal()
        self.heading += angular_command * dt + noise_theta
        self.heading = self.heading % (2 * np.pi)

        # 7. Move: update position
        noise_x = self.sigma_xy * np.sqrt(dt) * self.rng.standard_normal()
        noise_y = self.sigma_xy * np.sqrt(dt) * self.rng.standard_normal()
        self.x += self.speed * np.cos(self.heading) * dt + noise_x
        self.y += self.speed * np.sin(self.heading) * dt + noise_y

        # 8. Record history
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['heading'].append(self.heading)
        self.history['estimated_heading'].append(
            self.attractor.heading() + mag_dir)
        self.history['bump_amplitude'].append(self.attractor.bump_amplitude())

        return landscape.in_bounds(self.x, self.y)

    def run(self, landscape, duration, dt=0.01):
        """Run the bug for a given duration.

        Parameters
        ----------
        landscape : Landscape
        duration : float
            Total simulation time (seconds).
        dt : float
            Timestep (seconds).

        Returns
        -------
        dict
            History arrays (x, y, heading, estimated_heading, bump_amplitude).
        """
        n_steps = int(duration / dt)
        for _ in range(n_steps):
            in_bounds = self.step(dt, landscape)
            if not in_bounds:
                break

        # Convert lists to arrays
        return {k: np.array(v) for k, v in self.history.items()}

    def distance_from_start(self):
        """Euclidean distance from starting position."""
        x0 = self.history['x'][0]
        y0 = self.history['y'][0]
        return np.sqrt((self.x - x0)**2 + (self.y - y0)**2)

    def mean_heading_error(self):
        """Mean absolute heading error relative to goal (rad)."""
        h = np.array(self.history['heading'])
        err = np.abs((h - self.goal_heading + np.pi) % (2 * np.pi) - np.pi)
        return np.mean(err)
