"""
CPU4/CPU1 path integration circuit.

Simplified from Stone et al. (2017) Curr. Biol. 27:3069-3085.

The central complex maintains a home vector through velocity integration:
  h(t) = ∫₀ᵗ v(t') ê(θ(t')) dt'

where θ is the heading estimate from the ring attractor (TB1/E-PG neurons).

CPU4 neurons encode the displacement in a distributed population code:
  m_i(t) += speed(t) × [cos(θ(t) − φ_i)]₊ × dt

where φ_i is the preferred direction of neuron i and [·]₊ is half-wave
rectification.  Home direction is decoded by population vector of m_i.

Optionally leaky: m_i(t+dt) = (1 − λdt) m_i(t) + speed × [cos(θ − φ_i)]₊ dt
With λ = 0, perfect integration (drift accumulates forever).
With λ > 0, memory decays — limits integration window but bounds drift.
"""

import numpy as np


class CPU4:
    """CPU4 path integration neurons (scalar, for single-bug simulation).

    Parameters
    ----------
    n : int
        Number of CPU4 neurons (typically 8).
    leak : float
        Leak rate (1/s).  0 = perfect integrator.
    gain : float
        Integration gain (scales speed input).
    """

    def __init__(self, n=8, leak=0.0, gain=1.0):
        self.n = n
        self.leak = leak
        self.gain = gain
        self.phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self.memory = np.zeros(n)

    def update(self, heading, speed, dt):
        """Integrate one timestep of velocity.

        Parameters
        ----------
        heading : float
            Current heading estimate (rad, from ring attractor).
        speed : float
            Current forward speed (BL/s).
        dt : float
            Timestep (s).
        """
        cos_vals = np.cos(heading - self.phi)
        drive = self.gain * speed * np.maximum(cos_vals, 0.0)
        if self.leak > 0:
            self.memory *= (1.0 - self.leak * dt)
        self.memory += drive * dt

    def home_vector(self):
        """Decode home direction and distance.

        Returns
        -------
        distance : float
            Estimated distance from start (BL).
        direction : float
            Direction TO home (rad), i.e. opposite of displacement.
        """
        dx = np.sum(self.memory * np.cos(self.phi))
        dy = np.sum(self.memory * np.sin(self.phi))
        dist = np.sqrt(dx**2 + dy**2)
        home_dir = np.arctan2(-dy, -dx)
        return dist, home_dir

    def displacement(self):
        """Estimated displacement (x, y) from start."""
        dx = np.sum(self.memory * np.cos(self.phi))
        dy = np.sum(self.memory * np.sin(self.phi))
        return dx, dy

    def reset(self):
        self.memory = np.zeros(self.n)
