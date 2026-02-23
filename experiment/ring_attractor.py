"""
Ring attractor neural compass.

Models the insect central complex heading circuit as a ring of N neurons
with local cosine excitatory connectivity, activity-dependent global
inhibition (Δ7 pathway), and external input from the compass sensor array.

The bump of activity on the ring encodes the bug's estimated heading.
Angular velocity input shifts the bump (P-EN equivalent).
Compass input anchors the bump to the magnetic field direction.

Architecture (following Kakaria & de Bivort 2017):
  - E-PG neurons: N wedge neurons with preferred directions on [0, 2π)
  - Local excitation: cosine connectivity profile (positive lobe only)
  - Δ7 inhibition: global, proportional to mean ring activity
  - Activation: piece-wise linear [0, r_max]

The separation of excitation and inhibition is key: the Δ7 pathway
provides activity-dependent gain control, preventing uniform saturation
and maintaining a localized bump.

Based on:
  - Kim et al. (2017) Science 356:849 — ring attractor in Drosophila
  - Kakaria & de Bivort (2017) Front. Behav. Neurosci. 11:8
  - Stone et al. (2017) Curr. Biol. 27:3069 — CX path integration
"""

import numpy as np


class RingAttractor:
    """Rate-model ring attractor for heading representation.

    Parameters
    ----------
    n : int
        Number of neurons in the ring (typically 8 or 16).
    tau : float
        Time constant (seconds).
    w_exc : float
        Local excitatory weight amplitude (cosine profile, positive lobe).
    w_inh : float
        Global inhibition weight (Δ7 pathway). Inhibition = w_inh * mean(r).
        Must be > ~2.4 * w_exc for bump stability (suppresses uniform mode).
    g_mag : float
        Gain for magnetic compass input.
    g_omega : float
        Gain for angular velocity input.
    threshold : float
        Firing threshold.
    r_max : float
        Maximum firing rate.
    noise_sigma : float
        Intrinsic neural noise (std dev per timestep).
    rng : np.random.Generator or None
    """

    def __init__(self, n=8, tau=0.05, w_exc=1.5, w_inh=4.5,
                 g_mag=2.0, g_omega=0.5, threshold=0.0,
                 r_max=1.0, noise_sigma=0.01, rng=None):
        self.n = n
        self.tau = tau
        self.w_exc = w_exc
        self.w_inh = w_inh
        self.g_mag = g_mag
        self.g_omega = g_omega
        self.threshold = threshold
        self.r_max = r_max
        self.noise_sigma = noise_sigma
        self.rng = rng or np.random.default_rng()

        # Preferred directions
        self.theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Excitatory connectivity: local cosine profile, positive lobe only
        # Each neuron excites itself and nearest neighbours; no long-range excitation
        dtheta = self.theta[:, None] - self.theta[None, :]
        self.W_exc = w_exc * np.maximum(0.0, np.cos(dtheta))

        # State: firing rates in [0, r_max]
        self.r = np.zeros(n)

        # Initialise with a small bump to break symmetry
        self._init_bump()

    def _init_bump(self):
        """Initialise with a weak bump at a random position."""
        idx = self.rng.integers(0, self.n)
        for i in range(self.n):
            d = min(abs(i - idx), self.n - abs(i - idx))
            self.r[i] = max(0, 0.5 - 0.15 * d)

    def step(self, dt, compass_input=None, angular_velocity=0.0):
        """Advance the ring attractor by one timestep.

        Parameters
        ----------
        dt : float
            Timestep (seconds).
        compass_input : ndarray, shape (n,) or None
            Magnetic compass signal per channel (from CompassSensor.read).
        angular_velocity : float
            Bug's angular velocity (rad/s). Positive = counterclockwise.
        """
        # Local excitation (E-PG → E-PG via P-EN/P-EG)
        exc = self.W_exc @ self.r

        # Global inhibition (Δ7 pathway): proportional to mean ring activity
        inh = self.w_inh * np.mean(self.r)

        # Magnetic compass input — error-correcting shift
        #
        # The compass signal has cos 2α anisotropy, creating TWO peaks on the
        # ring (heading and heading+π).  Rather than adding this ambiguous
        # pattern as spatial excitation, we extract the heading in double-angle
        # space, compute the error relative to the current bump, and shift the
        # bump toward the nearest compatible heading.  This resolves the π
        # ambiguity inherent in the inclination compass.
        I_mag = np.zeros(self.n)
        if compass_input is not None:
            centred = compass_input - np.mean(compass_input)
            # Extract heading from compass in double-angle space
            z_compass = np.sum(centred * np.exp(2j * self.theta))
            if np.abs(z_compass) > 1e-10:
                double_heading = np.angle(z_compass)  # 2 * heading_actual
                double_bump = 2.0 * self.heading()
                # Error in double-angle space → actual error in [-π/2, π/2]
                error = np.angle(np.exp(1j * (double_heading - double_bump))) / 2.0
                # Shift bump toward the correct heading (same mechanism as ω)
                bump_grad = np.roll(self.r, 1) - np.roll(self.r, -1)
                I_mag = self.g_mag * error * bump_grad

        # Angular velocity input (P-EN equivalent): shifts bump in direction
        # of rotation.  Positive ω = counterclockwise → bump toward higher θ.
        I_omega = np.zeros(self.n)
        if angular_velocity != 0.0:
            I_omega = self.g_omega * angular_velocity * (
                np.roll(self.r, 1) - np.roll(self.r, -1))

        # Neural noise
        noise = self.rng.normal(0, self.noise_sigma, self.n)

        # Total drive
        drive = exc - inh + I_mag + I_omega - self.threshold + noise

        # Piece-wise linear activation: ReLU with saturation at r_max
        activated = np.clip(drive, 0, self.r_max)

        # Rate dynamics (Euler integration)
        dr = (-self.r + activated) / self.tau
        self.r = np.clip(self.r + dr * dt, 0, self.r_max)

    def heading(self):
        """Estimated heading from population vector decode.

        Returns
        -------
        float
            Estimated heading in [0, 2π).
        """
        z = np.sum(self.r * np.exp(1j * self.theta))
        if np.abs(z) < 1e-10:
            return 0.0
        return np.angle(z) % (2 * np.pi)

    def bump_amplitude(self):
        """Peak-to-trough amplitude of the activity bump."""
        return np.max(self.r) - np.min(self.r)

    def reset(self, heading=None):
        """Reset the attractor state.

        Parameters
        ----------
        heading : float or None
            If given, initialise bump centred on this heading (rad).
        """
        self.r = np.zeros(self.n)
        if heading is not None:
            diffs = (self.theta - heading + np.pi) % (2 * np.pi) - np.pi
            self.r = np.maximum(0, 0.5 * np.cos(diffs))
        else:
            self._init_bump()
