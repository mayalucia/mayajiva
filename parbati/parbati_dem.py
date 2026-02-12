#!/usr/bin/env python3
"""
First look at the Parbati Pyramid — 3D reconstruction from SRTM DEM.

Parvati Parbat: 6632m, 32.0905°N, 77.7347°E
SRTM 1-arc-second tile N32E077 from AWS (public, no auth).

Usage:
    vload py310
    python3 parbati/parbati_dem.py
"""

import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from urllib.request import urlopen

# ── Configuration ──────────────────────────────────────────────────────

PEAK_LAT = 32.0905   # Parvati Parbat latitude
PEAK_LON = 77.7347   # Parvati Parbat longitude
HALF_SIZE = 0.10     # Half-width in degrees (~11 km each side)

# SRTM tile covering this area (1°×1°, named by SW corner)
TILE_LAT = 32
TILE_LON = 77
TILE = f"N{TILE_LAT:02d}E{TILE_LON:03d}"

# Mapzen terrain tiles on AWS — public, no authentication
SRTM_URL = f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/N{TILE_LAT:02d}/{TILE}.hgt.gz"

# SRTM 1-arc-second: 3601 × 3601 pixels, signed 16-bit big-endian
SRTM_SIZE = 3601
VOID = -32768

# ── Data handling ──────────────────────────────────────────────────────

def download_srtm(url, cache_path):
    """Download and decompress SRTM .hgt.gz tile, caching locally."""
    if os.path.exists(cache_path):
        print(f"  cached: {cache_path}")
        return
    print(f"  downloading {url} ...")
    response = urlopen(url)
    raw = gzip.decompress(response.read())
    with open(cache_path, 'wb') as f:
        f.write(raw)
    print(f"  saved: {cache_path} ({len(raw)/(1024*1024):.1f} MB)")


def load_hgt(path):
    """Load SRTM .hgt as numpy float32 array. Voids become NaN."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='>i2')  # big-endian int16
    dem = data.reshape((SRTM_SIZE, SRTM_SIZE)).astype(np.float32)
    dem[dem == VOID] = np.nan
    return dem


def crop_dem(dem, center_lat, center_lon, half_deg):
    """Crop DEM to bounding box, return (elevation, lats, lons)."""
    north = TILE_LAT + 1  # top edge of tile
    west = TILE_LON        # left edge

    lat_min, lat_max = center_lat - half_deg, center_lat + half_deg
    lon_min, lon_max = center_lon - half_deg, center_lon + half_deg

    # Row 0 = north edge, row 3600 = south edge
    r0 = int((north - lat_max) * (SRTM_SIZE - 1))
    r1 = int((north - lat_min) * (SRTM_SIZE - 1))
    c0 = int((lon_min - west) * (SRTM_SIZE - 1))
    c1 = int((lon_max - west) * (SRTM_SIZE - 1))

    r0, r1 = max(0, r0), min(SRTM_SIZE - 1, r1)
    c0, c1 = max(0, c0), min(SRTM_SIZE - 1, c1)

    crop = dem[r0:r1+1, c0:c1+1]
    lats = np.linspace(lat_max, lat_min, crop.shape[0])
    lons = np.linspace(lon_min, lon_max, crop.shape[1])
    return crop, lats, lons

# ── Hillshade ──────────────────────────────────────────────────────────

def hillshade(elev, azimuth_deg=315, altitude_deg=45):
    """Classic analytical hillshade from elevation grid."""
    dy, dx = np.gradient(elev)
    az = np.radians(azimuth_deg)
    alt = np.radians(altitude_deg)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    return (np.cos(alt) * np.cos(slope) +
            np.sin(alt) * np.sin(slope) * np.cos(az - aspect))

# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    cache_path = os.path.join(data_dir, f"{TILE}.hgt")

    # 1. Get the data
    print("Fetching SRTM tile...")
    download_srtm(SRTM_URL, cache_path)

    # 2. Load and crop
    print("Loading elevation data...")
    dem = load_hgt(cache_path)
    print(f"  full tile: {dem.shape}, range: {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m")

    elev, lats, lons = crop_dem(dem, PEAK_LAT, PEAK_LON, HALF_SIZE)
    print(f"  cropped:   {elev.shape}, range: {np.nanmin(elev):.0f}–{np.nanmax(elev):.0f} m")

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # 3. Hillshade map (2D)
    print("Rendering hillshade...")
    hs = hillshade(elev)

    fig1, ax1 = plt.subplots(figsize=(12, 10))
    ax1.imshow(elev,
               extent=[lons[0], lons[-1], lats[-1], lats[0]],
               cmap='terrain', alpha=0.7)
    ax1.imshow(hs,
               extent=[lons[0], lons[-1], lats[-1], lats[0]],
               cmap='gray', alpha=0.4)
    ax1.plot(PEAK_LON, PEAK_LAT, 'r^', markersize=14,
             label=f'Parvati Parbat (6632 m)')
    ax1.legend(fontsize=12, loc='upper left')
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.set_title('Parbati Pyramid — SRTM Hillshade', fontsize=14)
    plt.colorbar(ax1.images[0], ax=ax1, label='Elevation (m)', shrink=0.7)

    hs_path = os.path.join(data_dir, 'parbati_hillshade.png')
    fig1.savefig(hs_path, dpi=150, bbox_inches='tight')
    print(f"  saved: {hs_path}")

    # 4. 3D surface
    print("Rendering 3D surface...")
    fig2 = plt.figure(figsize=(14, 10))
    ax2 = fig2.add_subplot(111, projection='3d')

    # Subsample for performance
    s = 2
    ax2.plot_surface(lon_grid[::s, ::s],
                     lat_grid[::s, ::s],
                     elev[::s, ::s],
                     cmap='terrain',
                     linewidth=0,
                     antialiased=True,
                     alpha=0.9)

    ax2.set_xlabel('Lon (°E)')
    ax2.set_ylabel('Lat (°N)')
    ax2.set_zlabel('Elevation (m)')
    ax2.set_title('Parbati Pyramid — 3D Surface', fontsize=14)
    # View from the south-west, looking up
    ax2.view_init(elev=35, azim=225)

    surf_path = os.path.join(data_dir, 'parbati_3d.png')
    fig2.savefig(surf_path, dpi=150, bbox_inches='tight')
    print(f"  saved: {surf_path}")

    plt.show()
    print("Done.")
