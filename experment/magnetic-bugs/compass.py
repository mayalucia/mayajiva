"""
Radical-pair compass sensor.

Models the singlet yield anisotropy of a cryptochrome-based magnetic compass.
The analytical approximation uses the axial HFC result:

    Φ_S(α) = Φ̄_S + (δΦ_S / 2)(1 + cos 2α)

where α is the angle between the bug's body axis and the local magnetic field
direction projected onto the horizontal plane.

An array of N_cry oriented cryptochrome molecules produces a set of noisy
sensor readings that can be fed into a ring attractor for heading estimation.
"""

import numpy as np


def singlet_yield(alpha, contrast=0.15, mean_yield=0.5):
    """Analytical singlet yield for the axial HFC approximation.

    Parameters
    ----------
    alpha : float or array
        Angle between sensor orientation and magnetic field direction (rad).
    contrast : float
        Relative anisotropy δΦ_S / Φ̄_S.  ~0.01 for [FAD·⁻ TrpH·⁺],
        ~0.15 for [FAD·⁻ O₂·⁻], up to ~4 with scavenger amplification.
    mean_yield : float
        Mean singlet yield Φ̄_S (dimensionless, ∈ [0, 1]).

    Returns
    -------
    float or array
        Singlet yield Φ_S(α).
    """
    delta = contrast * mean_yield
    return mean_yield + 0.5 * delta * (1.0 + np.cos(2.0 * alpha))


class CompassSensor:
    """Array of oriented cryptochrome molecules forming a compass.

    Each molecule has a fixed orientation φ_k relative to the bug's body axis.
    Given the bug's heading θ relative to magnetic North, molecule k sees an
    angle α_k = θ - φ_k, and produces a noisy singlet yield reading.

    The population average over N_cry molecules reduces noise by ~1/√N_cry.

    Parameters
    ----------
    n_cry : int
        Number of cryptochrome molecules in the array.
    n_channels : int
        Number of compass channels (matching ring attractor neurons).
        Molecules are binned into channels by their preferred direction.
    contrast : float
        Compass contrast C = δΦ_S / Φ̄_S.
    mean_yield : float
        Mean singlet yield.
    sigma_sensor : float
        Per-molecule sensor noise (std dev of Gaussian).
    rng : np.random.Generator or None
        Random number generator for reproducibility.
    """

    def __init__(self, n_cry=1000, n_channels=8, contrast=0.15,
                 mean_yield=0.5, sigma_sensor=0.02,
                 quantum_compass=None, rng=None):
        self.n_cry = n_cry
        self.n_channels = n_channels
        self.contrast = contrast
        self.mean_yield = mean_yield
        self.sigma_sensor = sigma_sensor
        self.quantum_compass = quantum_compass
        self.rng = rng or np.random.default_rng()

        # Molecule orientations: uniformly distributed around the circle
        self.phi = np.linspace(0, 2 * np.pi, n_cry, endpoint=False)

        # Channel centres (matching ring attractor preferred directions)
        self.channel_centres = np.linspace(0, 2 * np.pi, n_channels,
                                           endpoint=False)

        # Assign each molecule to its nearest channel
        # Angular distance from each molecule to each channel centre
        diffs = self.phi[:, None] - self.channel_centres[None, :]
        diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        self.assignments = np.argmin(np.abs(diffs), axis=1)

        # Pre-compute molecules per channel
        self.channel_counts = np.array([
            np.sum(self.assignments == c) for c in range(n_channels)
        ])

    def read(self, heading):
        """Read the compass at a given heading.

        Parameters
        ----------
        heading : float
            Bug's heading relative to magnetic North (rad).

        Returns
        -------
        channels : ndarray, shape (n_channels,)
            Population-averaged, noisy singlet yield per channel.
            Higher values indicate the channel's preferred direction
            is more aligned with the magnetic field.
        """
        # Angle each molecule sees
        alpha = heading - self.phi

        # Singlet yield per molecule (noiseless)
        if self.quantum_compass is not None:
            yields = self.quantum_compass.singlet_yield(alpha)
        else:
            yields = singlet_yield(alpha, self.contrast, self.mean_yield)

        # Add per-molecule noise
        noise = self.rng.normal(0, self.sigma_sensor, self.n_cry)
        noisy_yields = yields + noise

        # Average within each channel
        channels = np.zeros(self.n_channels)
        for c in range(self.n_channels):
            mask = self.assignments == c
            if np.any(mask):
                channels[c] = np.mean(noisy_yields[mask])

        return channels

    def signal_to_noise(self):
        """Theoretical signal-to-noise ratio per channel.

        SNR = (contrast * mean_yield) / (sigma_sensor / sqrt(n_per_channel))
        """
        n_per = self.n_cry / self.n_channels
        return (self.contrast * self.mean_yield) / (
            self.sigma_sensor / np.sqrt(n_per))
