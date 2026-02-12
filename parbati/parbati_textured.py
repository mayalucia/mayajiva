#!/usr/bin/env python3
"""
Parvati Valley and Parbati Pyramid — satellite-textured DEM.

Sentinel-2 cloudless imagery (EOX WMS, zero auth) draped over SRTM.
Produces:
  - Valley hillshade blended with satellite color (2D)
  - Peak 3D surface with satellite texture (3D)

Usage:
    vload py310
    python3 parbati/parbati_textured.py
"""

import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from PIL import Image
from io import BytesIO
from urllib.request import urlopen, Request

# ── Configuration ──────────────────────────────────────────────────────

SRTM_SIZE = 3601
VOID = -32768
SRTM_BASE = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"

PEAK_LAT, PEAK_LON = 32.0905, 77.7347

# Valley extent (wide view)
VALLEY = dict(lat_min=31.84, lat_max=32.20, lon_min=77.10, lon_max=77.88)
# Peak extent (close-up for 3D)
PEAK = dict(lat_min=31.99, lat_max=32.19, lon_min=77.63, lon_max=77.83)

TILES = [(32, 77), (31, 77)]

# EOX Sentinel-2 Cloudless WMS
EOX_WMS = "https://tiles.maps.eox.at/wms"

# ── SRTM functions (shared with parbati_valley.py) ────────────────────

def srtm_url(lat, lon):
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    name = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
    return f"{SRTM_BASE}/{ns}{abs(lat):02d}/{name}.hgt.gz", name


def download_tile(lat, lon, data_dir):
    url, name = srtm_url(lat, lon)
    path = os.path.join(data_dir, f"{name}.hgt")
    if os.path.exists(path):
        return path
    print(f"  downloading {name} ...")
    raw = gzip.decompress(urlopen(url).read())
    with open(path, 'wb') as f:
        f.write(raw)
    return path


def load_hgt(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='>i2')
    dem = data.reshape((SRTM_SIZE, SRTM_SIZE)).astype(np.float32)
    dem[dem == VOID] = np.nan
    return dem


def stitch_and_crop(tiles_data, lat_min, lat_max, lon_min, lon_max):
    all_lats = sorted(set(t[0] for t in tiles_data))
    all_lons = sorted(set(t[1] for t in tiles_data))
    n_lat, n_lon = len(all_lats), len(all_lons)
    rows = n_lat * (SRTM_SIZE - 1) + 1
    cols = n_lon * (SRTM_SIZE - 1) + 1
    full = np.full((rows, cols), np.nan, dtype=np.float32)
    for sw_lat, sw_lon, dem in tiles_data:
        lat_idx = n_lat - 1 - all_lats.index(sw_lat)
        lon_idx = all_lons.index(sw_lon)
        r0 = lat_idx * (SRTM_SIZE - 1)
        c0 = lon_idx * (SRTM_SIZE - 1)
        full[r0:r0+SRTM_SIZE, c0:c0+SRTM_SIZE] = dem
    full_north = max(all_lats) + 1
    full_south = min(all_lats)
    full_west = min(all_lons)
    full_east = max(all_lons) + 1
    r0 = int((full_north - lat_max) / (full_north - full_south) * (rows - 1))
    r1 = int((full_north - lat_min) / (full_north - full_south) * (rows - 1))
    c0 = int((lon_min - full_west) / (full_east - full_west) * (cols - 1))
    c1 = int((lon_max - full_west) / (full_east - full_west) * (cols - 1))
    r0, r1 = max(0, r0), min(rows - 1, r1)
    c0, c1 = max(0, c0), min(cols - 1, c1)
    crop = full[r0:r1+1, c0:c1+1]
    lats = np.linspace(lat_max, lat_min, crop.shape[0])
    lons = np.linspace(lon_min, lon_max, crop.shape[1])
    return crop, lats, lons

# ── Satellite imagery ─────────────────────────────────────────────────

def fetch_s2_cloudless(bbox, width, height, cache_path):
    """Fetch Sentinel-2 cloudless image from EOX WMS."""
    if os.path.exists(cache_path):
        print(f"  cached: {cache_path}")
        return np.array(Image.open(cache_path))

    params = (
        f"service=WMS&version=1.1.1&request=GetMap"
        f"&layers=s2cloudless-2024"
        f"&bbox={bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"
        f"&srs=EPSG:4326"
        f"&width={width}&height={height}"
        f"&format=image/jpeg"
    )
    url = f"{EOX_WMS}?{params}"
    print(f"  fetching S2 cloudless ({width}×{height}) ...")
    req = Request(url, headers={'User-Agent': 'MayaLucIA/0.1'})
    data = urlopen(req).read()
    img = Image.open(BytesIO(data))
    img.save(cache_path)
    print(f"  saved: {cache_path}")
    return np.array(img)

# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Load DEM tiles
    print("Loading DEM...")
    tiles_data = []
    for lat, lon in TILES:
        path = download_tile(lat, lon, data_dir)
        tiles_data.append((lat, lon, load_hgt(path)))

    # ──────────────────────────────────────────────────────────────────
    # 2. VALLEY VIEW: satellite + hillshade blend (2D)
    # ──────────────────────────────────────────────────────────────────
    print("\n── Valley view ──")
    v_elev, v_lats, v_lons = stitch_and_crop(tiles_data, **VALLEY)
    print(f"  DEM: {v_elev.shape}, {np.nanmin(v_elev):.0f}–{np.nanmax(v_elev):.0f} m")

    # Aspect ratio of the bounding box
    dlat = VALLEY['lat_max'] - VALLEY['lat_min']
    dlon = VALLEY['lon_max'] - VALLEY['lon_min']
    aspect = dlon / dlat * np.cos(np.radians(32.0))  # correct for latitude
    img_w = 4096  # max WMS allows
    img_h = int(img_w / aspect)

    sat_valley = fetch_s2_cloudless(
        VALLEY, img_w, img_h,
        os.path.join(data_dir, 's2_valley.jpg'))
    print(f"  satellite: {sat_valley.shape}")

    # Compute hillshade
    ls = LightSource(azdeg=315, altdeg=40)

    # Resize satellite to match DEM grid
    sat_resized = np.array(Image.fromarray(sat_valley).resize(
        (v_elev.shape[1], v_elev.shape[0]), Image.LANCZOS))
    sat_float = sat_resized.astype(np.float64) / 255.0

    # Shade the satellite image with the DEM hillshade
    rgb_shaded = ls.shade_rgb(sat_float, v_elev, blend_mode='soft', vert_exag=2)

    fig1, ax1 = plt.subplots(figsize=(18, 10))
    extent = [VALLEY['lon_min'], VALLEY['lon_max'],
              VALLEY['lat_min'], VALLEY['lat_max']]
    ax1.imshow(rgb_shaded, extent=extent, aspect='auto')
    ax1.plot(PEAK_LON, PEAK_LAT, 'r^', markersize=12, markeredgecolor='white',
             markeredgewidth=1)
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.set_title('Parvati Valley — Sentinel-2 Cloudless + SRTM Hillshade',
                   fontsize=14)
    out1 = os.path.join(data_dir, 'valley_textured.png')
    fig1.savefig(out1, dpi=180, bbox_inches='tight')
    print(f"  saved: {out1}")

    # ──────────────────────────────────────────────────────────────────
    # 3. PEAK 3D: satellite-textured surface
    # ──────────────────────────────────────────────────────────────────
    print("\n── Peak 3D view ──")
    p_elev, p_lats, p_lons = stitch_and_crop(tiles_data, **PEAK)
    print(f"  DEM: {p_elev.shape}, {np.nanmin(p_elev):.0f}–{np.nanmax(p_elev):.0f} m")

    p_aspect = (PEAK['lon_max'] - PEAK['lon_min']) / (PEAK['lat_max'] - PEAK['lat_min']) \
               * np.cos(np.radians(32.0))
    p_w = 2048
    p_h = int(p_w / p_aspect)

    sat_peak = fetch_s2_cloudless(
        PEAK, p_w, p_h,
        os.path.join(data_dir, 's2_peak.jpg'))
    print(f"  satellite: {sat_peak.shape}")

    # Subsample DEM for 3D rendering performance
    step = 3
    elev_s = p_elev[::step, ::step]
    nr, nc = elev_s.shape

    # Resize satellite to match subsampled DEM
    sat_for_3d = np.array(Image.fromarray(sat_peak).resize(
        (nc, nr), Image.LANCZOS))
    sat_norm = sat_for_3d.astype(np.float64) / 255.0

    # facecolors needs (nr-1, nc-1, 4) — average the 4 corner pixel colors per face
    fc = np.zeros((nr - 1, nc - 1, 4))
    fc[:, :, 0] = (sat_norm[:-1, :-1, 0] + sat_norm[1:, :-1, 0] +
                    sat_norm[:-1, 1:, 0] + sat_norm[1:, 1:, 0]) / 4
    fc[:, :, 1] = (sat_norm[:-1, :-1, 1] + sat_norm[1:, :-1, 1] +
                    sat_norm[:-1, 1:, 1] + sat_norm[1:, 1:, 1]) / 4
    fc[:, :, 2] = (sat_norm[:-1, :-1, 2] + sat_norm[1:, :-1, 2] +
                    sat_norm[:-1, 1:, 2] + sat_norm[1:, 1:, 2]) / 4
    fc[:, :, 3] = 1.0  # alpha

    # Simple shading: darken faces based on slope toward light
    dy, dx = np.gradient(elev_s)
    # Light from NW, 45° altitude
    az, alt = np.radians(315), np.radians(45)
    shade = (np.cos(alt) * np.cos(np.arctan(np.sqrt(dx**2 + dy**2))) +
             np.sin(alt) * np.sin(np.arctan(np.sqrt(dx**2 + dy**2))) *
             np.cos(az - np.arctan2(-dy, dx)))
    shade = np.clip(shade, 0.3, 1.0)
    # Average shade per face
    shade_f = (shade[:-1, :-1] + shade[1:, :-1] + shade[:-1, 1:] + shade[1:, 1:]) / 4
    fc[:, :, :3] *= shade_f[:, :, np.newaxis]

    lon_grid, lat_grid = np.meshgrid(
        np.linspace(PEAK['lon_min'], PEAK['lon_max'], nc),
        np.linspace(PEAK['lat_max'], PEAK['lat_min'], nr))

    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(lon_grid, lat_grid, elev_s,
                     facecolors=fc,
                     linewidth=0,
                     antialiased=True,
                     shade=False)
    ax2.set_xlabel('Lon (°E)')
    ax2.set_ylabel('Lat (°N)')
    ax2.set_zlabel('Elevation (m)')
    ax2.set_title('Parbati Pyramid — Satellite Texture on SRTM', fontsize=14)
    ax2.view_init(elev=30, azim=220)

    out2 = os.path.join(data_dir, 'peak_textured_3d.png')
    fig2.savefig(out2, dpi=180, bbox_inches='tight')
    print(f"  saved: {out2}")

    # ──────────────────────────────────────────────────────────────────
    # 4. Bonus: second 3D angle — from the east (Pin Parvati side)
    # ──────────────────────────────────────────────────────────────────
    ax2.view_init(elev=25, azim=135)
    out3 = os.path.join(data_dir, 'peak_textured_3d_east.png')
    fig2.savefig(out3, dpi=180, bbox_inches='tight')
    print(f"  saved: {out3}")

    plt.show()
    print("\nDone.")
