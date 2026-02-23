"""
Magnetic field landscape.

The field is characterised by:
  - B0: total field intensity (μT)
  - declination: angle between geographic North and magnetic North (rad)
  - inclination: dip angle below horizontal (rad, positive downward in N hemisphere)

The compass bug operates in the horizontal plane, so the relevant quantity
is the horizontal projection of the field and its direction.

Anomaly types (Direction A):
  - 'gaussian': original isotropic Gaussian blob (backward compatible)
  - 'dipole':   buried vertical magnetic dipole (volcanic intrusion)
  - 'fault':    linear fault juxtaposing differently magnetised rocks
  - 'gradient': regional linear field gradient
"""

import numpy as np


class Landscape:
    """2D landscape with a magnetic field.

    Parameters
    ----------
    extent : tuple of float
        (width, height) of the landscape in body-lengths.
    B0 : float
        Total geomagnetic field intensity (μT). Default 50.
    declination : float
        Magnetic declination (rad). Angle from geographic N to magnetic N,
        positive eastward. Default 0.
    inclination : float
        Magnetic inclination (rad). Dip angle below horizontal.
        ~65° at mid-latitudes. Default 65° ≈ 1.134 rad.
    anomalies : list of dict or None
        Each dict must contain a 'type' key ('gaussian', 'dipole', 'fault',
        'gradient'). Dicts without 'type' default to 'gaussian' for backward
        compatibility.
    """

    def __init__(self, extent=(1000, 1000), B0=50.0, declination=0.0,
                 inclination=np.radians(65.0), anomalies=None):
        self.extent = extent
        self.B0 = B0
        self.declination = declination
        self.inclination = inclination
        self.anomalies = anomalies or []

        # Horizontal component of the geomagnetic field
        self.B_horizontal = B0 * np.cos(inclination)
        # Vertical component (into the ground in N hemisphere)
        self.B_vertical = B0 * np.sin(inclination)

        # Background direction (for fast deviation queries)
        self._bg_dir = declination

    def magnetic_direction(self, x, y):
        """Local magnetic field direction in the horizontal plane.

        Parameters
        ----------
        x, y : float or array
            Position in the landscape.

        Returns
        -------
        direction : float or array
            Angle of the horizontal field component from geographic North (rad).
        intensity : float or array
            Horizontal field intensity (μT).
        inclination : float or array
            Local inclination angle (rad).
        """
        # Start with uniform field
        Bx = self.B_horizontal * np.cos(self.declination)
        By = self.B_horizontal * np.sin(self.declination)
        Bz = self.B_vertical

        # Broadcast for array inputs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        Bx = np.broadcast_to(Bx, x.shape).copy()
        By = np.broadcast_to(By, x.shape).copy()
        Bz = np.broadcast_to(Bz, x.shape).copy()

        # Add anomalies
        for anom in self.anomalies:
            dBx, dBy, dBz = self._anomaly_perturbation(x, y, anom)
            Bx += dBx
            By += dBy
            Bz += dBz

        B_h = np.sqrt(Bx**2 + By**2)
        direction = np.arctan2(By, Bx)
        local_incl = np.arctan2(Bz, B_h)

        return direction, B_h, local_incl

    def direction_deviation(self, x, y):
        """Local field direction minus background direction (rad).

        Vectorised over arrays of (x, y). This is the quantity that
        biases the compass: δφ > 0 means the field is rotated eastward.
        """
        direction, _, _ = self.magnetic_direction(x, y)
        delta = direction - self._bg_dir
        # Wrap to [-π, π]
        return (delta + np.pi) % (2 * np.pi) - np.pi

    # ── anomaly dispatch ──────────────────────────────────────────

    def _anomaly_perturbation(self, x, y, anom):
        """Compute field perturbation (dBx, dBy, dBz) from one anomaly.

        Works with scalar or array (x, y).
        """
        atype = anom.get('type', 'gaussian')
        if atype == 'gaussian':
            return self._gaussian_perturbation(x, y, anom)
        elif atype == 'dipole':
            return self._dipole_perturbation(x, y, anom)
        elif atype == 'fault':
            return self._fault_perturbation(x, y, anom)
        elif atype == 'gradient':
            return self._gradient_perturbation(x, y, anom)
        else:
            raise ValueError(f"Unknown anomaly type: {atype}")

    def _gaussian_perturbation(self, x, y, anom):
        """Original Gaussian blob — backward compatible.

        Keys: pos, strength, radius.
        """
        ax, ay = anom['pos']
        r = np.sqrt((x - ax)**2 + (y - ay)**2)
        radius = anom['radius']
        envelope = anom['strength'] * np.exp(-0.5 * (r / radius)**2)
        envelope = np.where(r < 3 * radius, envelope, 0.0)
        safe_r = np.where(r > 1e-6, r, 1.0)
        dBx = np.where(r > 1e-6, envelope * (x - ax) / safe_r, 0.0)
        dBy = np.where(r > 1e-6, envelope * (y - ay) / safe_r, 0.0)
        return dBx, dBy, np.zeros_like(dBx)

    def _dipole_perturbation(self, x, y, anom):
        """Buried vertical magnetic dipole.

        Models a magnetised body (volcanic intrusion, iron-rich inclusion)
        as a vertical magnetic dipole at depth d below the surface.

        The horizontal field peaks at ρ = d/2 from directly above and
        decays as ~1/ρ⁴ at large distances.  The vertical component is
        strongest directly above (≈ 2.3× peak horizontal) and reverses
        sign at ρ = d√2.

        Keys: pos (x,y), strength (peak horizontal anomaly, μT),
              depth (burial depth, body-lengths).
        """
        px, py = anom['pos']
        strength = anom['strength']
        depth = anom['depth']

        dx = x - px
        dy = y - py
        rho2 = dx**2 + dy**2
        R2 = rho2 + depth**2
        R5 = R2**2.5

        # Prefactor: normalised so peak horizontal anomaly = strength.
        # Peak occurs at ρ = d/2; analytic normalisation gives:
        alpha = strength * (5**2.5) * depth**3 / 48.0

        dBx = alpha * 3.0 * depth * dx / R5
        dBy = alpha * 3.0 * depth * dy / R5
        dBz = alpha * (2.0 * depth**2 - rho2) / R5

        return dBx, dBy, dBz

    def _fault_perturbation(self, x, y, anom):
        """Linear magnetic fault.

        Models two half-planes with different remanent magnetisation
        joined along a line.  The horizontal field jumps by `contrast`
        across the fault over a transition width `width` (tanh profile).

        The perturbation is perpendicular to the fault strike.

        Keys: pos (x,y on fault), azimuth (strike from N, rad),
              contrast (μT), width (half-width, body-lengths).
        """
        px, py = anom['pos']
        az = anom['azimuth']
        contrast = anom['contrast']
        width = anom['width']

        # Signed perpendicular distance (positive on right side of fault)
        d_perp = (x - px) * np.sin(az) - (y - py) * np.cos(az)
        profile = np.tanh(d_perp / width)

        # Perturbation in the fault-normal direction
        dBx = (contrast / 2.0) * profile * np.sin(az)
        dBy = -(contrast / 2.0) * profile * np.cos(az)
        return dBx, dBy, np.zeros_like(dBx)

    def _gradient_perturbation(self, x, y, anom):
        """Regional linear field gradient.

        Models the approach to a large-scale geological feature.
        The horizontal field increases linearly along direction `direction`
        at rate `magnitude` (μT per body-length) from a reference point.

        Keys: magnitude (μT/BL), direction (rad from N),
              ref (x,y reference, default: landscape centre).
        """
        mag = anom['magnitude']
        dirn = anom['direction']
        rx, ry = anom.get('ref', (self.extent[0] / 2, self.extent[1] / 2))

        # Displacement along gradient direction
        s = (x - rx) * np.cos(dirn) + (y - ry) * np.sin(dirn)

        dBx = mag * s * np.cos(dirn)
        dBy = mag * s * np.sin(dirn)
        return dBx, dBy, np.zeros_like(dBx)

    # ── random anomaly generation ─────────────────────────────────

    @staticmethod
    def random_dipoles(n, extent, strength, depth, rng=None):
        """Generate n random dipole anomalies with random-sign strengths."""
        rng = rng or np.random.default_rng()
        anomalies = []
        for _ in range(n):
            pos = (rng.uniform(0, extent[0]), rng.uniform(0, extent[1]))
            s = strength * rng.choice([-1, 1])
            anomalies.append({
                'type': 'dipole', 'pos': pos,
                'strength': s, 'depth': depth,
            })
        return anomalies

    # ── bounds ────────────────────────────────────────────────────

    def in_bounds(self, x, y):
        """Check if position is within the landscape."""
        w, h = self.extent
        return 0 <= x <= w and 0 <= y <= h
