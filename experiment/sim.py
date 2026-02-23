"""
Simulation runner and visualisation for the magnetic bug.

Usage:
    python sim.py                        # Single trajectory demo
    python sim.py --sweep                # Parameter sweep (contrast vs noise)
    python sim.py --ensemble N           # Ensemble of N trajectories
    python sim.py --quantum toy_fad_o2   # Use quantum compass model
    python sim.py --validate             # Quantum yield curves for all models
    python sim.py --compare-models       # Navigation accuracy per model
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from agent import Bug
from landscape import Landscape


# ── Single trajectory ────────────────────────────────────────────────

def run_single(seed=42, duration=500, dt=0.01, contrast=0.15,
               sigma_theta=0.1, goal=3*np.pi/4, landscape=None,
               compass_params=None):
    """Run a single bug and return its history."""
    if landscape is None:
        landscape = Landscape()
    if compass_params is None:
        compass_params = {'contrast': contrast, 'n_cry': 1000,
                          'sigma_sensor': 0.02}

    bug = Bug(
        x0=500, y0=100, goal_heading=goal, speed=1.0,
        kappa=2.0, sigma_theta=sigma_theta, sigma_xy=0.05,
        compass_params=compass_params,
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


# ── Quantum compass models ───────────────────────────────────────────

_QUANTUM_MODELS = {}


def _ensure_spin_dynamics():
    """Lazy import to avoid loading spin_dynamics for analytical runs."""
    if not _QUANTUM_MODELS:
        from spin_dynamics import (toy_fad_o2, toy_fad_trp,
                                   intermediate_fad_o2, intermediate_fad_trp,
                                   RadicalPairCompass)
        _QUANTUM_MODELS.update({
            'toy_fad_o2': toy_fad_o2,
            'toy_fad_trp': toy_fad_trp,
            'intermediate_fad_o2': intermediate_fad_o2,
            'intermediate_fad_trp': intermediate_fad_trp,
        })
        _QUANTUM_MODELS['_RadicalPairCompass'] = RadicalPairCompass
    return _QUANTUM_MODELS


def make_quantum_compass(model_name, **kwargs):
    """Create a RadicalPairCompass from a model name."""
    models = _ensure_spin_dynamics()
    factory = models[model_name]
    RPC = models['_RadicalPairCompass']
    return RPC(model=factory(), **kwargs)


# ── Validation: quantum yield curves ─────────────────────────────────

def plot_validate(save_prefix=None):
    """Plot Phi_S(theta) for all quantum models, overlay analytical fit."""
    models = _ensure_spin_dynamics()
    RPC = models['_RadicalPairCompass']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    names = ['toy_fad_o2', 'toy_fad_trp', 'intermediate_fad_o2',
             'intermediate_fad_trp']

    for ax, name in zip(axes.flat, names):
        factory = models[name]
        model = factory()
        compass = RPC(model=model, n_theta=360)
        thetas, yields = compass.yield_curve()
        th_deg = np.degrees(thetas)

        ax.plot(th_deg, yields, 'b-', lw=2, label='quantum')

        # Analytical fit: Phi(a) = mean + (C*mean/2)(1 + cos 2a)
        mean = compass.mean_yield
        C = compass.contrast
        y_an = mean + 0.5 * C * mean * (1 + np.cos(2 * thetas))
        ax.plot(th_deg, y_an, 'r--', lw=1.5, label=f'analytical (C={C:.3f})')

        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$\Phi_S$')
        ax.set_title(model['name'])
        ax.legend(fontsize=8)
        ax.text(0.02, 0.02, f'C = {C:.3f}\nmean = {mean:.4f}',
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

    fig.suptitle('Quantum Singlet Yield Anisotropy', fontsize=14)
    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}validate.png', dpi=150)
        print(f'Saved {save_prefix}validate.png')
    return fig


# ── Model comparison: navigation accuracy ────────────────────────────

def run_model_comparison(n_runs=20, duration=300, dt=0.02, sigma_theta=0.1,
                         save_prefix=None):
    """Run navigation ensemble for each quantum model + analytical baseline."""
    models = _ensure_spin_dynamics()
    names = ['toy_fad_o2', 'toy_fad_trp', 'intermediate_fad_o2',
             'intermediate_fad_trp']

    results = {}

    # Analytical baseline
    print('  [analytical] running...')
    r = run_ensemble(n_runs=n_runs, duration=duration, dt=dt,
                     contrast=0.15, sigma_theta=sigma_theta)
    results['analytical\n(C=0.15)'] = r['mean_error_deg']
    print(f'  [analytical] err={r["mean_error_deg"]:.1f}')

    for name in names:
        qc = make_quantum_compass(name)
        print(f'  [{name}] running (C={qc.contrast:.3f})...')

        landscape = Landscape()
        errors = []
        for i in range(n_runs):
            bug = Bug(
                x0=500, y0=100, goal_heading=3*np.pi/4, speed=1.0,
                kappa=2.0, sigma_theta=sigma_theta, sigma_xy=0.05,
                compass_params={'quantum_compass': qc, 'n_cry': 1000,
                                'sigma_sensor': 0.02},
                seed=i
            )
            bug.run(landscape, duration=duration, dt=dt)
            errors.append(bug.mean_heading_error())
        err_deg = np.degrees(np.mean(errors))
        label = name.replace('_', ' ')
        label += f'\n(C={qc.contrast:.3f})'
        results[label] = err_deg
        print(f'  [{name}] err={err_deg:.1f}')

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    vals = [results[k] for k in labels]
    colors = ['grey'] + ['#2196F3', '#4CAF50', '#1565C0', '#2E7D32']
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean heading error (degrees)')
    ax.set_title('Navigation Accuracy: Analytical vs Quantum Compass Models')

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}compare_models.png', dpi=150)
        print(f'Saved {save_prefix}compare_models.png')
    return fig


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Magnetic bug navigation simulation')
    parser.add_argument('--sweep', action='store_true',
                        help='Run parameter sweep (slow)')
    parser.add_argument('--ensemble', type=int, default=0,
                        help='Run N-trajectory ensemble')
    parser.add_argument('--validate', action='store_true',
                        help='Plot quantum yield curves for all models')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare navigation accuracy across models')
    parser.add_argument('--quantum', type=str, default=None,
                        choices=['toy_fad_o2', 'toy_fad_trp',
                                 'intermediate_fad_o2', 'intermediate_fad_trp'],
                        help='Use quantum compass model')
    parser.add_argument('--duration', type=float, default=500,
                        help='Simulation duration (seconds)')
    parser.add_argument('--contrast', type=float, default=0.15,
                        help='Compass contrast C')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Angular noise sigma_theta')
    parser.add_argument('--save', type=str, default=None,
                        help='Save figures to this prefix (e.g., "fig_")')
    args = parser.parse_args()

    # Build compass_params with optional quantum compass
    compass_params = {'contrast': args.contrast, 'n_cry': 1000,
                      'sigma_sensor': 0.02}
    if args.quantum:
        qc = make_quantum_compass(args.quantum)
        compass_params['quantum_compass'] = qc
        print(f'Using quantum compass: {args.quantum} '
              f'(C={qc.contrast:.3f}, mean={qc.mean_yield:.4f})')

    if args.validate:
        print('Generating quantum yield validation plots...')
        fig = plot_validate(save_prefix=args.save)
        plt.show()

    elif args.compare_models:
        print('Running model comparison...')
        fig = run_model_comparison(n_runs=20, duration=300, dt=0.02,
                                   sigma_theta=args.sigma,
                                   save_prefix=args.save)
        plt.show()

    elif args.sweep:
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
                                  sigma_theta=args.sigma,
                                  compass_params=compass_params)
        print(f'  Distance from start: {bug.distance_from_start():.1f} BL')
        print(f'  Mean heading error:  {bug.mean_heading_error()*180/np.pi:.1f}')
        fig = plot_trajectory(history)
        if args.save:
            fig.savefig(f'{args.save}trajectory.png', dpi=150)
        plt.show()


if __name__ == '__main__':
    main()
