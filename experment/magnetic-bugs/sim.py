"""
Simulation runner and visualisation for the magnetic bug.

Usage:
    python sim.py                  # Single trajectory demo
    python sim.py --sweep          # Parameter sweep (contrast vs noise)
    python sim.py --ensemble N     # Ensemble of N trajectories
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from agent import Bug
from landscape import Landscape


# ── Single trajectory ────────────────────────────────────────────────

def run_single(seed=42, duration=500, dt=0.01, contrast=0.15,
               sigma_theta=0.1, goal=3*np.pi/4, landscape=None):
    """Run a single bug and return its history."""
    if landscape is None:
        landscape = Landscape()

    bug = Bug(
        x0=500, y0=100, goal_heading=goal, speed=1.0,
        kappa=2.0, sigma_theta=sigma_theta, sigma_xy=0.05,
        compass_params={'contrast': contrast, 'n_cry': 1000,
                        'sigma_sensor': 0.02},
        seed=seed
    )
    history = bug.run(landscape, duration=duration, dt=dt)
    return history, bug


def plot_trajectory(history, title="Magnetic Bug Trajectory", goal=3*np.pi/4):
    """Plot the bug's trajectory coloured by heading error."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Trajectory (coloured by time) ──
    ax = axes[0]
    x, y = history['x'], history['y']
    t = np.arange(len(x))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', linewidth=0.8)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.set_xlim(x.min() - 20, x.max() + 20)
    ax.set_ylim(y.min() - 20, y.max() + 20)
    ax.set_aspect('equal')
    ax.set_xlabel('x (body-lengths)')
    ax.set_ylabel('y (body-lengths)')
    ax.set_title('Trajectory')
    ax.plot(x[0], y[0], 'go', ms=8, label='start')
    ax.plot(x[-1], y[-1], 'r*', ms=10, label='end')

    # Draw goal direction arrow from start
    arrow_len = 50
    ax.annotate('', xy=(x[0] + arrow_len * np.cos(goal),
                        y[0] + arrow_len * np.sin(goal)),
                xytext=(x[0], y[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.legend(fontsize=8)
    plt.colorbar(lc, ax=ax, label='time step')

    # ── Heading over time ──
    ax = axes[1]
    dt_plot = np.arange(len(history['heading']))
    ax.plot(dt_plot, np.degrees(history['heading']), 'b-', lw=0.5,
            alpha=0.6, label='actual')
    ax.plot(dt_plot, np.degrees(history['estimated_heading']), 'r-', lw=0.5,
            alpha=0.6, label='estimated')
    ax.axhline(np.degrees(goal), color='k', ls='--', lw=1, label='goal')
    ax.set_xlabel('time step')
    ax.set_ylabel('heading (°)')
    ax.set_title('Heading vs Time')
    ax.legend(fontsize=8)

    # ── Ring attractor bump amplitude ──
    ax = axes[2]
    ax.plot(dt_plot, history['bump_amplitude'], 'g-', lw=0.5)
    ax.set_xlabel('time step')
    ax.set_ylabel('bump amplitude')
    ax.set_title('Ring Attractor Stability')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


# ── Ensemble ─────────────────────────────────────────────────────────

def run_ensemble(n_runs=50, duration=500, dt=0.01, contrast=0.15,
                 sigma_theta=0.1, goal=3*np.pi/4):
    """Run an ensemble of bugs and compute statistics."""
    landscape = Landscape()
    distances = []
    mean_errors = []

    for i in range(n_runs):
        bug = Bug(
            x0=500, y0=100, goal_heading=goal, speed=1.0,
            kappa=2.0, sigma_theta=sigma_theta, sigma_xy=0.05,
            compass_params={'contrast': contrast, 'n_cry': 1000,
                            'sigma_sensor': 0.02},
            seed=i
        )
        bug.run(landscape, duration=duration, dt=dt)
        distances.append(bug.distance_from_start())
        mean_errors.append(bug.mean_heading_error())

    return {
        'distances': np.array(distances),
        'mean_errors': np.array(mean_errors),
        'mean_distance': np.mean(distances),
        'mean_error_deg': np.degrees(np.mean(mean_errors)),
    }


def plot_ensemble(n_runs=20, duration=300, dt=0.01, contrast=0.15,
                  sigma_theta=0.1, goal=3*np.pi/4):
    """Plot an ensemble of trajectories."""
    landscape = Landscape()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i in range(n_runs):
        bug = Bug(
            x0=500, y0=100, goal_heading=goal, speed=1.0,
            kappa=2.0, sigma_theta=sigma_theta, sigma_xy=0.05,
            compass_params={'contrast': contrast, 'n_cry': 1000,
                            'sigma_sensor': 0.02},
            seed=i
        )
        history = bug.run(landscape, duration=duration, dt=dt)
        ax.plot(history['x'], history['y'], lw=0.5, alpha=0.5)

    ax.plot(500, 100, 'go', ms=10, zorder=5)
    arrow_len = 80
    ax.annotate('', xy=(500 + arrow_len * np.cos(goal),
                        100 + arrow_len * np.sin(goal)),
                xytext=(500, 100),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))

    ax.set_aspect('equal')
    ax.set_xlabel('x (body-lengths)')
    ax.set_ylabel('y (body-lengths)')
    ax.set_title(f'Ensemble ({n_runs} bugs), contrast={contrast}, '
                 f'σ_θ={sigma_theta}')
    plt.tight_layout()
    return fig


# ── Parameter sweep: navigation phase diagram ────────────────────────

def parameter_sweep(contrasts=None, sigma_thetas=None, n_runs=20,
                    duration=300, dt=0.01):
    """Sweep compass contrast vs angular noise.

    Returns a 2D array of mean heading errors (degrees).
    """
    if contrasts is None:
        contrasts = np.logspace(-2.5, 0, 12)  # 0.003 to 1.0
    if sigma_thetas is None:
        sigma_thetas = np.logspace(-2, 0.5, 12)  # 0.01 to ~3

    errors = np.zeros((len(contrasts), len(sigma_thetas)))

    total = len(contrasts) * len(sigma_thetas)
    done = 0

    for i, C in enumerate(contrasts):
        for j, sig in enumerate(sigma_thetas):
            result = run_ensemble(n_runs=n_runs, duration=duration, dt=dt,
                                  contrast=C, sigma_theta=sig)
            errors[i, j] = result['mean_error_deg']
            done += 1
            print(f'  [{done}/{total}] C={C:.4f}, σ={sig:.3f} → '
                  f'err={errors[i,j]:.1f}°')

    return contrasts, sigma_thetas, errors


def plot_phase_diagram(contrasts, sigma_thetas, errors):
    """Plot the navigation phase diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    im = ax.pcolormesh(sigma_thetas, contrasts, errors,
                       shading='auto', cmap='RdYlGn_r')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Angular noise σ_θ (rad/√s)')
    ax.set_ylabel('Compass contrast C')
    ax.set_title('Navigation Phase Diagram\n'
                 '(mean heading error, degrees)')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Mean heading error (°)')

    # Overlay contour at 10° and 30°
    ax.contour(sigma_thetas, contrasts, errors,
               levels=[10, 30, 60], colors='white', linewidths=1.5)

    # Mark the biologically relevant regime
    ax.axhline(0.15, color='cyan', ls='--', lw=1.5,
               label='[FAD·⁻ O₂·⁻] contrast')
    ax.axhline(0.01, color='orange', ls='--', lw=1.5,
               label='[FAD·⁻ TrpH·⁺] contrast')
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Magnetic bug navigation simulation')
    parser.add_argument('--sweep', action='store_true',
                        help='Run parameter sweep (slow)')
    parser.add_argument('--ensemble', type=int, default=0,
                        help='Run N-trajectory ensemble')
    parser.add_argument('--duration', type=float, default=500,
                        help='Simulation duration (seconds)')
    parser.add_argument('--contrast', type=float, default=0.15,
                        help='Compass contrast C')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Angular noise σ_θ')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figures to this prefix (e.g., "fig_")')
    args = parser.parse_args()

    if args.sweep:
        print('Running parameter sweep...')
        C, S, E = parameter_sweep(n_runs=10, duration=200, dt=0.02)
        fig = plot_phase_diagram(C, S, E)
        if args.save:
            fig.savefig(f'{args.save}phase_diagram.png', dpi=150)
        plt.show()

    elif args.ensemble > 0:
        print(f'Running ensemble of {args.ensemble} bugs...')
        fig = plot_ensemble(n_runs=args.ensemble, duration=args.duration,
                            contrast=args.contrast, sigma_theta=args.sigma)
        if args.save:
            fig.savefig(f'{args.save}ensemble.png', dpi=150)
        plt.show()

    else:
        print('Running single trajectory...')
        history, bug = run_single(duration=args.duration,
                                  contrast=args.contrast,
                                  sigma_theta=args.sigma)
        print(f'  Distance from start: {bug.distance_from_start():.1f} BL')
        print(f'  Mean heading error:  {bug.mean_heading_error()*180/np.pi:.1f}°')
        fig = plot_trajectory(history)
        if args.save:
            fig.savefig(f'{args.save}trajectory.png', dpi=150)
        plt.show()


if __name__ == '__main__':
    main()
