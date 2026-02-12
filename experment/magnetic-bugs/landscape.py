"""
Magnetic field landscape.

For Phase 1: uniform geomagnetic field on a flat 2D plane.
The field is characterised by:
  - B0: total field intensity (μT)
  - declination: angle between geographic North and magnetic North (rad)
  - inclination: dip angle below horizontal (rad, positive downward in N hemisphere)

The compass bug operates in the horizontal plane, so the relevant quantity
is the horizontal projection of the field and its direction.

Optional: localised magnetic anomalies (iron-rich geology, etc.)
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
        Magnetic anomalies: [{'pos': (x,y), 'strength': δB, 'radius': r}, ...]
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

    def magnetic_direction(self, x, y):
        """Local magnetic field direction in the horizontal plane.

        Parameters
        ----------
        x, y : float
            Position in the landscape.

        Returns
        -------
        direction : float
            Angle of the horizontal field component from geographic North (rad).
            Equals the declination for a uniform field.
        intensity : float
            Horizontal field intensity (μT).
        inclination : float
            Local inclination angle (rad).
        """
        # Start with uniform field
        Bx = self.B_horizontal * np.cos(self.declination)  # North component
        By = self.B_horizontal * np.sin(self.declination)  # East component
        Bz = self.B_vertical

        # Add anomalies
        for anom in self.anomalies:
            ax, ay = anom['pos']
            r = np.sqrt((x - ax)**2 + (y - ay)**2)
            radius = anom['radius']
            if r < 3 * radius:  # cutoff at 3 radii
                # Gaussian anomaly in horizontal components
                strength = anom['strength'] * np.exp(-0.5 * (r / radius)**2)
                # Point radially away from anomaly centre
                if r > 1e-6:
                    Bx += strength * (x - ax) / r
                    By += strength * (y - ay) / r

        B_h = np.sqrt(Bx**2 + By**2)
        direction = np.arctan2(By, Bx)
        local_incl = np.arctan2(Bz, B_h)

        return direction, B_h, local_incl

    def in_bounds(self, x, y):
        """Check if position is within the landscape."""
        w, h = self.extent
        return 0 <= x <= w and 0 <= y <= h
