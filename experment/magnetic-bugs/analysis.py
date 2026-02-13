"""
Deeper analysis of the magnetic bug navigation model.

Probes the parameter regime where compass model differences matter:
near the navigation phase boundary where Peclet number ~ O(1).

Usage:
    python analysis.py --peclet           # Peclet number study
    python analysis.py --differentiate    # Model comparison at the boundary
    python analysis.py --harmonics        # Spherical harmonic decomposition
    python analysis.py --critical-noise   # Critical noise per model
    python analysis.py --all              # Everything
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from landscape import Landscape
from compass import singlet_yield, CompassSensor
from ring_attractor import RingAttractor
from sim import make_quantum_compass, _ensure_spin_dynamics


# ── Vectorised fast simulation ─────────────────────────────────────
# The per-step Python loop in agent.py is the bottleneck for sweeps.
# This fast path skips the ring attractor and simulates the OU heading
# process directly, using the compass contrast as a parameter.
#
# The heading evolves as:
#   θ_{t+1} = θ_t + κ sin(θ_goal - θ_t + η_compass) dt + σ_θ √dt ξ
#
# where η_compass is the compass heading error.  For an N_cry sensor
# array with contrast C and per-molecule noise σ_sensor, the compass
# heading error std is approximately:
#   σ_compass ≈ 1 / (C * √N_cry * SNR_per_molecule)
#
# This is an OU process on the circle.  We simulate it with Euler-
# Maruyama for N_bugs in parallel (fully vectorised).

def fast_ensemble(n_bugs, duration, dt, kappa, sigma_theta,
                  contrast, n_cry, sigma_sensor, goal=3*np.pi/4,
                  speed=1.0, sigma_xy=0.05, seed=0, mean_yield=None):
    """Vectorised simulation of n_bugs navigating bugs.

    Returns mean heading error (degrees) and array of final distances.

    Parameters
    ----------
    mean_yield : float or None
        Mean singlet yield.  If None, defaults to 0.5 (analytical model).
        For quantum models, pass the actual mean yield so that the
        absolute anisotropy δ = C × mean_yield is correct.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    sqrt_dt = np.sqrt(dt)

    # Initial conditions
    theta = rng.uniform(0, 2*np.pi, n_bugs)
    x = np.full(n_bugs, 500.0)
    y = np.full(n_bugs, 100.0)

    # Compass noise: approximate the compass heading error
    # The sensor array gives N_cry readings with noise σ_sensor.
    # The compass heading estimate has std ≈ σ_sensor / (δ * √(2 * N_cry/n_ch))
    # where δ = Φ_max − Φ_min = C × mean_yield is the absolute anisotropy.
    if mean_yield is None:
        mean_yield = 0.5
    delta = contrast * mean_yield
    n_per_ch = n_cry / 8.0
    # Fisher information for cos 2α signal with Gaussian noise:
    # I(α) ∝ (dΦ/dα)² / σ² = (2δ sin 2α)² / (σ²/N_per_ch)
    # Averaged over α: <sin² 2α> = 1/2
    # → σ_heading ≈ σ_sensor / (δ * √(2 * N_per_ch))
    sigma_compass = sigma_sensor / (delta * np.sqrt(2 * n_per_ch)) if delta > 1e-10 else 10.0

    # Accumulate heading error for mean
    heading_errors_sum = np.zeros(n_bugs)

    for _ in range(n_steps):
        # Compass-corrupted heading estimate
        compass_noise = rng.normal(0, sigma_compass, n_bugs)
        heading_est = theta + compass_noise

        # Steering
        heading_error = goal - heading_est
        d_theta = kappa * np.sin(heading_error) * dt
        d_theta += sigma_theta * sqrt_dt * rng.standard_normal(n_bugs)
        theta = (theta + d_theta) % (2 * np.pi)

        # Position
        x += speed * np.cos(theta) * dt + sigma_xy * sqrt_dt * rng.standard_normal(n_bugs)
        y += speed * np.sin(theta) * dt + sigma_xy * sqrt_dt * rng.standard_normal(n_bugs)

        # Track heading error
        err = np.abs(((theta - goal) + np.pi) % (2*np.pi) - np.pi)
        heading_errors_sum += err

    mean_err_per_bug = heading_errors_sum / n_steps
    distances = np.sqrt((x - 500)**2 + (y - 100)**2)

    return np.degrees(np.mean(mean_err_per_bug)), distances


# ── 1. Peclet number study ─────────────────────────────────────────

def peclet_study(save_prefix=None):
    """Compare analytical Peclet number against simulated navigation."""
    kappa = 2.0
    L = 300.0
    sigma_range = np.logspace(-1.5, 0.7, 25)

    Pe_analytical = kappa * L / sigma_range**2

    n_bugs = 100
    duration = 200
    dt = 0.02

    mean_errors = []
    success_rates = []

    for sig in sigma_range:
        err, _ = fast_ensemble(
            n_bugs=n_bugs, duration=duration, dt=dt,
            kappa=kappa, sigma_theta=sig,
            contrast=0.15, n_cry=1000, sigma_sensor=0.02)
        # Re-run for success rate
        _, dists = fast_ensemble(
            n_bugs=n_bugs, duration=duration, dt=dt,
            kappa=kappa, sigma_theta=sig,
            contrast=0.15, n_cry=1000, sigma_sensor=0.02, seed=1)
        mean_errors.append(err)
        # Success: error < 15°
        # Use error from the run
        errs_individual, _ = fast_ensemble(
            n_bugs=n_bugs, duration=duration, dt=dt,
            kappa=kappa, sigma_theta=sig,
            contrast=0.15, n_cry=1000, sigma_sensor=0.02, seed=2)
        success_rates.append(1.0 if err < 15 else max(0, 1.0 - (err - 15)/75))
        print(f'  σ_θ={sig:.3f}  Pe={kappa*L/sig**2:.0f}  err={err:.1f}°')

    mean_errors = np.array(mean_errors)
    success_rates = np.array(success_rates)

    # Better success rate: run the ensemble and check individual bugs
    success_rates_real = []
    for sig in sigma_range:
        rng = np.random.default_rng(42)
        n_steps = int(duration / dt)
        sqrt_dt = np.sqrt(dt)
        theta = rng.uniform(0, 2*np.pi, n_bugs)
        goal = 3*np.pi/4
        err_sum = np.zeros(n_bugs)
        sigma_compass = 0.02 / (0.15 * 0.5 * np.sqrt(2 * 1000/8))
        for _ in range(n_steps):
            cn = rng.normal(0, sigma_compass, n_bugs)
            he = goal - (theta + cn)
            theta = (theta + kappa * np.sin(he) * dt +
                     sig * sqrt_dt * rng.standard_normal(n_bugs)) % (2*np.pi)
            err_sum += np.abs(((theta - goal) + np.pi) % (2*np.pi) - np.pi)
        per_bug = np.degrees(err_sum / n_steps)
        success_rates_real.append(np.mean(per_bug < 15))
    success_rates_real = np.array(success_rates_real)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.loglog(sigma_range, Pe_analytical, 'b-', lw=2,
              label=r'$Pe = \kappa L / \sigma_\theta^2$')
    ax.axhline(1, color='red', ls='--', lw=1.5, label='Pe = 1')
    ax.axhline(10, color='orange', ls=':', lw=1.5, label='Pe = 10')
    ax.set_xlabel(r'$\sigma_\theta$ (rad/$\sqrt{s}$)')
    ax.set_ylabel('Peclet number')
    ax.set_title('Peclet Number')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogx(Pe_analytical, mean_errors, 'ko-', ms=4)
    ax.axhline(10, color='green', ls='--', lw=1, label=r'$10°$')
    ax.axhline(30, color='orange', ls='--', lw=1, label=r'$30°$')
    ax.axhline(90, color='red', ls=':', lw=1, alpha=0.5, label=r'$90°$ (random)')
    ax.axvline(1, color='red', ls='--', lw=1, alpha=0.3)
    ax.set_xlabel('Peclet number')
    ax.set_ylabel('Mean heading error (°)')
    ax.set_title('Navigation Accuracy vs Pe')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 95)

    ax = axes[2]
    ax.semilogx(Pe_analytical, success_rates_real, 'go-', ms=4)
    ax.axvline(1, color='red', ls='--', lw=1, alpha=0.3, label='Pe = 1')
    ax.axvline(10, color='orange', ls=':', lw=1, alpha=0.3, label='Pe = 10')
    ax.set_xlabel('Peclet number')
    ax.set_ylabel('Success rate (error < 15°)')
    ax.set_title('Navigation Success vs Pe')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    fig.suptitle('Peclet Number Governs Navigation', fontsize=14)
    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}peclet.png', dpi=150)
        print(f'Saved {save_prefix}peclet.png')
    return fig


# ── 2. Model differentiation at the phase boundary ────────────────

def model_differentiation(save_prefix=None):
    """Run all compass models at noise levels near the phase boundary."""
    sigma_range = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    n_cry = 50
    n_bugs = 200
    duration = 200
    dt = 0.02

    models = _ensure_spin_dynamics()
    model_names = ['toy_fad_o2', 'toy_fad_trp',
                   'intermediate_fad_o2', 'intermediate_fad_trp']

    # Get contrasts from quantum models
    RPC = models['_RadicalPairCompass']
    quantum_contrasts = {}
    for name in model_names:
        factory = models[name]
        qc = RPC(model=factory())
        quantum_contrasts[name] = qc.contrast

    results = {}

    # Analytical baseline C=0.15
    label = 'analytical (C=0.15)'
    errs = []
    for sig in sigma_range:
        err, _ = fast_ensemble(n_bugs=n_bugs, duration=duration, dt=dt,
                               kappa=2.0, sigma_theta=sig,
                               contrast=0.15, n_cry=n_cry, sigma_sensor=0.02)
        errs.append(err)
        print(f'  {label}  σ={sig:.2f}  err={err:.1f}°')
    results[label] = np.array(errs)

    # Quantum models — use their computed contrasts
    for name in model_names:
        C = quantum_contrasts[name]
        label = f'{name.replace("_"," ")} (C={C:.3f})'
        errs = []
        for sig in sigma_range:
            err, _ = fast_ensemble(n_bugs=n_bugs, duration=duration, dt=dt,
                                   kappa=2.0, sigma_theta=sig,
                                   contrast=C, n_cry=n_cry, sigma_sensor=0.02)
            errs.append(err)
            print(f'  {label}  σ={sig:.2f}  err={err:.1f}°')
        results[label] = np.array(errs)

    # Also run at biological contrasts (literature values with relaxation)
    for C_lit, label in [(0.01, '[FAD TrpH] lit. (C=0.01)'),
                          (0.15, '[FAD O₂] lit. (C=0.15)')]:
        errs = []
        for sig in sigma_range:
            err, _ = fast_ensemble(n_bugs=n_bugs, duration=duration, dt=dt,
                                   kappa=2.0, sigma_theta=sig,
                                   contrast=C_lit, n_cry=n_cry, sigma_sensor=0.02)
            errs.append(err)
            print(f'  {label}  σ={sig:.2f}  err={err:.1f}°')
        results[label] = np.array(errs)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ['grey', '#2196F3', '#4CAF50', '#1565C0', '#2E7D32',
              '#FF9800', '#00BCD4']
    markers = ['s', 'o', '^', 'D', 'v', 'P', 'X']

    for (label, errs), color, marker in zip(results.items(), colors, markers):
        ax.plot(sigma_range, errs, color=color, marker=marker, ms=5,
                lw=2, label=label)

    ax.axhline(10, color='green', ls=':', lw=1, alpha=0.4)
    ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
    ax.axhline(90, color='red', ls=':', lw=1, alpha=0.3)
    ax.set_xlabel(r'Angular noise $\sigma_\theta$ (rad/$\sqrt{s}$)', fontsize=12)
    ax.set_ylabel('Mean heading error (°)', fontsize=12)
    ax.set_title(f'Model Differentiation: Computed vs Literature Contrasts\n'
                 f'($N_{{cry}} = {n_cry}$, $\\kappa = 2.0$)',
                 fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 95)
    ax.set_xscale('log')

    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}differentiate.png', dpi=150)
        print(f'Saved {save_prefix}differentiate.png')
    return fig


# ── 3. Spherical harmonic decomposition ───────────────────────────

def harmonic_decomposition(save_prefix=None):
    """Decompose Φ_S(θ) into Legendre polynomial components."""
    from numpy.polynomial.legendre import legval, legfit

    models = _ensure_spin_dynamics()
    RPC = models['_RadicalPairCompass']
    model_names = ['toy_fad_o2', 'toy_fad_trp',
                   'intermediate_fad_o2', 'intermediate_fad_trp']

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    all_coeffs = {}

    for ax, name in zip(axes.flat, model_names):
        factory = models[name]
        model = factory()
        compass = RPC(model=model, n_theta=500)
        thetas, yields = compass.yield_curve()

        x = np.cos(thetas)
        max_L = 8
        coeffs = legfit(x, yields, max_L)
        y_fit = legval(x, coeffs)

        coeffs_02 = np.zeros(max_L + 1)
        coeffs_02[0] = coeffs[0]
        coeffs_02[2] = coeffs[2]
        y_02 = legval(x, coeffs_02)

        total_var = np.var(yields)
        resid_02 = np.var(yields - y_02)
        resid_full = np.var(yields - y_fit)

        ax.plot(np.degrees(thetas), yields, 'b-', lw=2, label='quantum')
        ax.plot(np.degrees(thetas), y_02, 'r--', lw=1.5, label='L=0,2 only')
        ax.plot(np.degrees(thetas), y_fit, 'g:', lw=1.5, label=f'L=0..{max_L}')
        ax.set_xlabel(r'$\theta$ (°)')
        ax.set_ylabel(r'$\Phi_S$')
        ax.set_title(model['name'])
        ax.legend(fontsize=8)

        r2_02 = 100*(1 - resid_02/total_var) if total_var > 0 else 100
        r2_full = 100*(1 - resid_full/total_var) if total_var > 0 else 100
        info = (f'C = {compass.contrast:.3f}\n'
                f'L=0,2 captures {r2_02:.1f}%\n'
                f'L=0..{max_L} captures {r2_full:.1f}%')
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                va='bottom', bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))
        all_coeffs[name] = coeffs

    fig.suptitle('Legendre Decomposition of Singlet Yield Anisotropy', fontsize=14)
    plt.tight_layout()

    # Bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    max_L = 8
    x_pos = np.arange(max_L + 1)
    width = 0.18
    colors = ['#2196F3', '#4CAF50', '#1565C0', '#2E7D32']

    for i, (name, coeffs) in enumerate(all_coeffs.items()):
        rel = np.abs(coeffs) / np.abs(coeffs[0]) if np.abs(coeffs[0]) > 1e-10 else np.abs(coeffs)
        # Replace zeros with tiny value for log plot
        rel = np.maximum(rel, 1e-7)
        ax2.bar(x_pos + i * width, rel, width, label=name.replace('_', ' '),
                color=colors[i], alpha=0.8)

    ax2.set_xticks(x_pos + 1.5 * width)
    ax2.set_xticklabels([f'L={l}' for l in range(max_L + 1)])
    ax2.set_ylabel(r'$|c_L| / c_0$')
    ax2.set_title('Relative Legendre Coefficients')
    ax2.legend(fontsize=9)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 2)

    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}harmonics_fit.png', dpi=150)
        fig2.savefig(f'{save_prefix}harmonics_bar.png', dpi=150)
        print(f'Saved {save_prefix}harmonics_fit.png and {save_prefix}harmonics_bar.png')
    return fig, fig2


# ── 4. Critical noise per model ───────────────────────────────────

def critical_noise(save_prefix=None):
    """Find σ_θ* where mean heading error crosses 30° for each model."""
    models = _ensure_spin_dynamics()
    RPC = models['_RadicalPairCompass']
    model_names = ['toy_fad_o2', 'toy_fad_trp',
                   'intermediate_fad_o2', 'intermediate_fad_trp']

    n_bugs = 200
    duration = 200
    dt = 0.02
    n_cry = 50
    threshold = 30.0

    def run_error(contrast, sig):
        err, _ = fast_ensemble(n_bugs=n_bugs, duration=duration, dt=dt,
                               kappa=2.0, sigma_theta=sig,
                               contrast=contrast, n_cry=n_cry,
                               sigma_sensor=0.02, seed=42)
        return err

    def find_critical(contrast, label):
        lo, hi = 0.1, 5.0
        err_lo = run_error(contrast, lo)
        err_hi = run_error(contrast, hi)
        print(f'  {label}: err({lo})={err_lo:.1f}°, err({hi})={err_hi:.1f}°')
        if err_lo > threshold:
            return lo
        if err_hi < threshold:
            return hi
        for _ in range(10):
            mid = np.sqrt(lo * hi)
            err_mid = run_error(contrast, mid)
            print(f'    bisect σ={mid:.3f}  err={err_mid:.1f}°')
            if err_mid < threshold:
                lo = mid
            else:
                hi = mid
        return np.sqrt(lo * hi)

    results = {}

    # Literature contrasts
    for C, label in [(0.01, '[FAD TrpH] lit. (C=0.01)'),
                      (0.15, '[FAD O₂] lit. (C=0.15)')]:
        sig_star = find_critical(C, label)
        results[label] = sig_star
        print(f'  → σ* = {sig_star:.3f}')

    # Computed quantum contrasts
    for name in model_names:
        factory = models[name]
        qc = RPC(model=factory())
        label = f'{name.replace("_"," ")} (C={qc.contrast:.3f})'
        sig_star = find_critical(qc.contrast, label)
        results[label] = sig_star
        print(f'  → σ* = {sig_star:.3f}')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    vals = [results[k] for k in labels]
    colors = ['#FF9800', '#00BCD4', '#2196F3', '#4CAF50', '#1565C0', '#2E7D32']

    bars = ax.barh(range(len(labels)), vals, color=colors[:len(labels)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r'Critical noise $\sigma_\theta^*$ (rad/$\sqrt{s}$) at 30° threshold',
                  fontsize=11)
    ax.set_title(f'Noise Tolerance by Compass Model ($N_{{cry}} = {n_cry}$)')

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'{save_prefix}critical_noise.png', dpi=150)
        print(f'Saved {save_prefix}critical_noise.png')
    return fig


# ── 5. N_cry sweep: sensor population trade-off ───────────────────

def ncry_sweep(save_prefix=None):
    """Sweep N_cry for C=0.01 and C=0.15 at several noise levels.

    Answers: how many cryptochrome molecules does the weak compass
    need to match the strong one?

    Also computes the analytical compass noise σ_compass(N_cry, C)
    and the effective total noise σ_eff = √(σ_θ² + κ² σ_compass²)
    to verify the threshold prediction.
    """
    n_cry_range = np.array([5, 10, 20, 50, 100, 200, 500, 1000,
                            2000, 5000, 10000])
    sigma_thetas = [0.3, 0.5, 1.0]
    contrasts = {'[FAD O₂] C=0.15': 0.15,
                 '[FAD TrpH] C=0.01': 0.01}
    n_bugs = 200
    duration = 200
    dt = 0.02
    sigma_sensor = 0.02

    results = {}  # {(label, sigma): array of errors}

    for sig in sigma_thetas:
        for label, C in contrasts.items():
            key = (label, sig)
            errs = []
            for n_cry in n_cry_range:
                err, _ = fast_ensemble(
                    n_bugs=n_bugs, duration=duration, dt=dt,
                    kappa=2.0, sigma_theta=sig,
                    contrast=C, n_cry=n_cry, sigma_sensor=sigma_sensor)
                errs.append(err)
                print(f'  {label}  σ={sig}  N_cry={n_cry:5d}  err={err:.1f}°')
            results[key] = np.array(errs)

    # ── Analytical prediction ──
    # σ_compass(N, C) = σ_sensor / (C * mean * √(2N/8))
    # The heading error from compass alone is approximately σ_compass
    # The total effective error rate combines motor + compass:
    #   σ_eff² ≈ σ_θ² + (κ * σ_compass)² * dt  [not quite right dimensionally]
    # More carefully: compass adds ~σ_compass rad error per reading,
    # steering gain κ converts this to κ*σ_compass rad/s torque noise,
    # which acts like additional σ_θ.
    # Effective: σ_eff = √(σ_θ² + (κ * σ_compass)²) approximately.
    # But the σ_compass is per-step, not per-√s... let's just show the
    # compass noise vs motor noise comparison.

    mean_yield = 0.5
    n_ch = 8

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)

    for ax, sig in zip(axes, sigma_thetas):
        for (label, C), color, ls in zip(
                contrasts.items(),
                ['#00BCD4', '#FF9800'],
                ['-', '--']):
            key = (label, sig)
            ax.semilogx(n_cry_range, results[key], color=color,
                        marker='o', ms=5, lw=2, ls=ls, label=label)

        ax.set_xlabel(r'$N_{cry}$ (cryptochrome molecules)', fontsize=11)
        ax.set_title(f'$\\sigma_\\theta = {sig}$ rad/$\\sqrt{{s}}$', fontsize=12)
        ax.axhline(10, color='green', ls=':', lw=1, alpha=0.4)
        ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
        ax.legend(fontsize=9)

        # Mark the N_cry where C=0.01 first matches C=0.15's asymptote
        err_high = results[('[FAD O₂] C=0.15', sig)]
        err_low = results[('[FAD TrpH] C=0.01', sig)]
        asymptote = err_high[-1]  # C=0.15 at N=10000 (saturated)
        # Find crossover
        for i, (el, nc) in enumerate(zip(err_low, n_cry_range)):
            if el <= asymptote * 1.1:  # within 10% of asymptote
                ax.axvline(nc, color='red', ls=':', lw=1, alpha=0.5)
                ax.text(nc * 1.1, asymptote + 2,
                        f'N*≈{nc}', fontsize=9, color='red')
                break

    axes[0].set_ylabel('Mean heading error (°)', fontsize=11)
    axes[0].set_ylim(0, 70)

    fig.suptitle('Sensor Population Compensates for Low Contrast', fontsize=14)
    plt.tight_layout()

    # ── Second figure: compass noise vs motor noise ──
    fig2, ax2 = plt.subplots(figsize=(9, 6))

    n_cry_fine = np.logspace(0.5, 4.5, 100)
    for label, C in contrasts.items():
        delta = C * mean_yield
        sigma_c = sigma_sensor / (delta * np.sqrt(2 * n_cry_fine / n_ch))
        ax2.loglog(n_cry_fine, np.degrees(sigma_c), lw=2, label=label)

    for sig in sigma_thetas:
        ax2.axhline(np.degrees(sig), ls='--', lw=1, alpha=0.6,
                     label=f'$\\sigma_\\theta = {sig}$')

    ax2.set_xlabel(r'$N_{cry}$', fontsize=12)
    ax2.set_ylabel(r'Compass heading noise $\sigma_{compass}$ (°)', fontsize=12)
    ax2.set_title('Compass Noise vs Motor Noise', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.1, 500)
    ax2.set_xlim(3, 30000)

    # Shade the "compass-limited" region
    ax2.fill_between(n_cry_fine,
                     np.degrees(sigma_thetas[0]) * np.ones_like(n_cry_fine),
                     500, alpha=0.05, color='red')
    ax2.text(5, 200, 'compass-limited\n(σ_compass > σ_θ)',
             fontsize=10, color='red', alpha=0.6)
    ax2.text(5000, 0.3, 'motor-limited\n(σ_compass < σ_θ)',
             fontsize=10, color='green', alpha=0.6)

    plt.tight_layout()

    if save_prefix:
        fig.savefig(f'{save_prefix}ncry_sweep.png', dpi=150)
        fig2.savefig(f'{save_prefix}ncry_noise.png', dpi=150)
        print(f'Saved {save_prefix}ncry_sweep.png and {save_prefix}ncry_noise.png')
    return fig, fig2


# ── 6. Full simulation validation at C=0.01 ──────────────────────

def validate_fast_vs_full(save_prefix=None):
    """Compare the fast Gaussian approximation against the full
    ring-attractor simulation at C=0.01 (marginal compass).

    The ring attractor's nonlinear winner-take-all dynamics may filter
    compass noise better than the Gaussian model predicts.
    """
    from agent import Bug

    sigma_range = np.array([0.1, 0.3, 0.5, 0.8, 1.0])
    n_cry = 50
    n_runs = 30
    duration = 200
    dt = 0.02
    landscape = Landscape()

    contrasts_to_test = [0.01, 0.15]
    results = {}

    for C in contrasts_to_test:
        # Fast simulation
        fast_errs = []
        for sig in sigma_range:
            err, _ = fast_ensemble(
                n_bugs=200, duration=duration, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=C, n_cry=n_cry, sigma_sensor=0.02)
            fast_errs.append(err)
        results[(C, 'fast')] = np.array(fast_errs)

        # Full ring attractor simulation
        full_errs = []
        for sig in sigma_range:
            run_errors = []
            for seed in range(n_runs):
                bug = Bug(
                    x0=500, y0=100, goal_heading=3*np.pi/4, speed=1.0,
                    kappa=2.0, sigma_theta=sig, sigma_xy=0.05,
                    compass_params={'contrast': C, 'n_cry': n_cry,
                                    'sigma_sensor': 0.02},
                    seed=seed
                )
                bug.run(landscape, duration=duration, dt=dt)
                run_errors.append(bug.mean_heading_error())
            err = np.degrees(np.mean(run_errors))
            full_errs.append(err)
            print(f'  C={C}  σ={sig}  full={err:.1f}°  fast={results[(C,"fast")][len(full_errs)-1]:.1f}°')
        results[(C, 'full')] = np.array(full_errs)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, C in zip(axes, contrasts_to_test):
        fast = results[(C, 'fast')]
        full = results[(C, 'full')]
        ax.plot(sigma_range, fast, 'b--o', ms=6, lw=2, label='fast (Gaussian)')
        ax.plot(sigma_range, full, 'r-s', ms=6, lw=2, label='full (ring attractor)')

        # Relative difference
        diff = full - fast
        for i, (s, d) in enumerate(zip(sigma_range, diff)):
            if abs(d) > 0.5:
                ax.annotate(f'{d:+.1f}°', (s, full[i]),
                            textcoords='offset points', xytext=(8, 5),
                            fontsize=8, color='purple')

        ax.set_xlabel(r'$\sigma_\theta$ (rad/$\sqrt{s}$)', fontsize=11)
        ax.set_title(f'C = {C} ($N_{{cry}}$ = {n_cry})', fontsize=12)
        ax.legend(fontsize=10)
        ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
        ax.set_ylim(0, 70)

    axes[0].set_ylabel('Mean heading error (°)', fontsize=11)

    fig.suptitle('Fast Simulation vs Full Ring Attractor', fontsize=14)
    plt.tight_layout()

    if save_prefix:
        fig.savefig(f'{save_prefix}validate_fast.png', dpi=150)
        print(f'Saved {save_prefix}validate_fast.png')
    return fig


# ── 7. Relaxation + navigation: does the threshold survive? ──────

def relaxation_navigation(save_prefix=None):
    """Navigation performance with relaxation-suppressed contrasts.

    Uses pre-computed contrast values from the spin relaxation analysis
    to check whether the C ~ 0.1 navigation threshold survives with
    realistic decoherence.
    """
    # Pre-computed contrasts from the relaxation analysis.
    # Format: (label, C_no_relax, C_T2_3us, C_T2_1us, C_asym)
    # "asym" = FAD T2=3µs, partner T2=1µs
    relaxed_data = {
        'toy FAD-O₂':   {'none': 0.391, 'T2=10µs': None, 'T2=3µs': 0.275, 'T2=1µs': 0.165, 'asym': 0.193},
        'toy FAD-TrpH':  {'none': 0.210, 'T2=10µs': None, 'T2=3µs': 0.155, 'T2=1µs': 0.095, 'asym': 0.120},
        'inter FAD-O₂': {'none': 0.448, 'T2=10µs': None, 'T2=3µs': 0.304, 'T2=1µs': 0.176, 'asym': 0.217},
        'inter FAD-TrpH': {'none': 0.245, 'T2=10µs': 0.211, 'T2=3µs': 0.158, 'T2=1µs': 0.084, 'asym': 0.115},
    }

    # Relaxation scenarios to compare
    scenarios = ['none', 'T2=3µs', 'T2=1µs']

    sigma_range = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])
    n_cry = 50
    n_bugs = 200
    duration = 200
    dt = 0.02

    # ── Figure 1: Navigation curves per relaxation scenario ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    model_colors = {
        'toy FAD-O₂': '#2196F3',
        'toy FAD-TrpH': '#4CAF50',
        'inter FAD-O₂': '#1565C0',
        'inter FAD-TrpH': '#2E7D32',
    }

    for ax, scenario in zip(axes, scenarios):
        # Literature baselines
        for C_lit, label, color, ls in [
            (0.15, '[FAD O₂] lit.', '#00BCD4', '--'),
            (0.01, '[FAD TrpH] lit.', '#FF9800', '--'),
        ]:
            errs = []
            for sig in sigma_range:
                err, _ = fast_ensemble(n_bugs=n_bugs, duration=duration, dt=dt,
                                       kappa=2.0, sigma_theta=sig,
                                       contrast=C_lit, n_cry=n_cry, sigma_sensor=0.02)
                errs.append(err)
            ax.plot(sigma_range, errs, color=color, ls=ls, marker='x', ms=5,
                    lw=1.5, label=label)

        # Quantum models with relaxation
        for model_name, contrasts in relaxed_data.items():
            C = contrasts.get(scenario)
            if C is None:
                continue
            errs = []
            for sig in sigma_range:
                err, _ = fast_ensemble(n_bugs=n_bugs, duration=duration, dt=dt,
                                       kappa=2.0, sigma_theta=sig,
                                       contrast=C, n_cry=n_cry, sigma_sensor=0.02)
                errs.append(err)
            label = f'{model_name} (C={C:.3f})'
            ax.plot(sigma_range, errs, color=model_colors[model_name],
                    marker='o', ms=4, lw=2, label=label)

        ax.axhline(10, color='green', ls=':', lw=1, alpha=0.4)
        ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
        ax.axhline(90, color='red', ls=':', lw=1, alpha=0.3)
        ax.set_xlabel(r'$\sigma_\theta$ (rad/$\sqrt{s}$)', fontsize=11)
        title = 'No relaxation' if scenario == 'none' else scenario + ' both'
        ax.set_title(title, fontsize=12)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='upper left')

    axes[0].set_ylabel('Mean heading error (°)', fontsize=11)
    axes[0].set_ylim(0, 95)

    fig.suptitle(f'Relaxation + Navigation ($N_{{cry}} = {n_cry}$): '
                 'Does the Threshold Survive?', fontsize=14)
    plt.tight_layout()

    # ── Figure 2: Contrast bar chart across scenarios ──
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    threshold_C = 0.10
    x_pos = np.arange(len(relaxed_data))
    width = 0.2
    scen_colors = {'none': '#4CAF50', 'T2=3µs': '#FF9800', 'T2=1µs': '#F44336', 'asym': '#9C27B0'}

    for i, scenario in enumerate(scenarios + ['asym']):
        vals = []
        for model_name in relaxed_data:
            c = relaxed_data[model_name].get(scenario)
            vals.append(c if c is not None else 0)
        bars = ax2.bar(x_pos + i * width, vals, width,
                       label=scenario if scenario != 'none' else 'no relaxation',
                       color=scen_colors.get(scenario, 'grey'), alpha=0.8)

    ax2.axhline(threshold_C, color='red', ls='--', lw=2, alpha=0.7,
                label=f'navigation threshold C ≈ {threshold_C}')
    ax2.set_xticks(x_pos + 1.5 * width)
    ax2.set_xticklabels(list(relaxed_data.keys()), fontsize=10)
    ax2.set_ylabel('Contrast C', fontsize=12)
    ax2.set_title('Contrast Suppression by Spin Relaxation', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 0.5)

    plt.tight_layout()

    if save_prefix:
        fig.savefig(f'{save_prefix}relax_navigation.png', dpi=150)
        fig2.savefig(f'{save_prefix}relax_contrasts.png', dpi=150)
        print(f'Saved {save_prefix}relax_navigation.png and {save_prefix}relax_contrasts.png')
    return fig, fig2


# ── 8. Unequal recombination rates ────────────────────────────────

def unequal_rates(save_prefix=None):
    """Sweep k_T/k_S and compute contrast, absolute anisotropy, navigation.

    The singlet yield Φ_S depends strongly on the ratio k_T/k_S:
    - k_T/k_S → 0: everything recombines through singlet, Φ_S → 1,
      anisotropy vanishes.
    - k_T/k_S → ∞: everything exits via triplet, Φ_S → 0, same.
    - k_T/k_S ≈ 1: maximum absolute anisotropy δ = Φ_max − Φ_min.

    The key metric for navigation is δ (not relative contrast C),
    because the compass SNR scales with δ.
    """
    models = _ensure_spin_dynamics()
    RPC = models['_RadicalPairCompass']
    model_names = ['toy_fad_o2', 'toy_fad_trp',
                   'intermediate_fad_o2', 'intermediate_fad_trp']
    # Skip dim=64 for the dense sweep — too slow
    fast_models = ['toy_fad_o2', 'toy_fad_trp', 'intermediate_fad_o2']

    k_S = 1e6
    ratios = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
                        0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0])

    data = {}  # {model_name: dict of arrays}

    for name in fast_models:
        factory = models[name]
        contrasts = []
        deltas = []
        means = []
        for r in ratios:
            rpc = RPC(model=factory(), k_S=k_S, k_T=r * k_S, n_theta=90)
            contrasts.append(rpc.contrast)
            deltas.append(rpc.max_yield - rpc.min_yield)
            means.append(rpc.mean_yield)
            print(f'  {name}  k_T/k_S={r:.3f}  C={rpc.contrast:.4f}  '
                  f'δ={rpc.max_yield - rpc.min_yield:.4f}  mean={rpc.mean_yield:.4f}')
        data[name] = {
            'C': np.array(contrasts),
            'delta': np.array(deltas),
            'mean': np.array(means),
        }

    # ── Figure 1: C and δ vs k_T/k_S ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'toy_fad_o2': '#2196F3', 'toy_fad_trp': '#4CAF50',
              'intermediate_fad_o2': '#1565C0'}

    for name in fast_models:
        d = data[name]
        label = name.replace('_', ' ')
        ax1.semilogx(ratios, d['C'], color=colors[name], marker='o', ms=4,
                     lw=2, label=label)
        ax2.semilogx(ratios, d['delta'], color=colors[name], marker='s', ms=4,
                     lw=2, label=label)

    ax1.axvline(1.0, color='grey', ls=':', lw=1, alpha=0.5)
    ax1.set_xlabel(r'$k_T / k_S$', fontsize=12)
    ax1.set_ylabel('Relative contrast C', fontsize=12)
    ax1.set_title('Relative Contrast', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.axhline(0.1, color='red', ls='--', lw=1.5, alpha=0.5,
                label='nav threshold')

    ax2.axvline(1.0, color='grey', ls=':', lw=1, alpha=0.5)
    ax2.set_xlabel(r'$k_T / k_S$', fontsize=12)
    ax2.set_ylabel(r'Absolute anisotropy $\delta = \Phi_{max} - \Phi_{min}$',
                   fontsize=12)
    ax2.set_title('Absolute Anisotropy (drives compass SNR)', fontsize=13)
    ax2.legend(fontsize=9)

    fig.suptitle('Effect of Unequal Recombination Rates', fontsize=14)
    plt.tight_layout()

    # ── Figure 2: Navigation error vs k_T/k_S at fixed noise ──
    fig2, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    sigma_thetas = [0.3, 0.5, 1.0]
    n_cry = 50
    n_bugs = 200
    duration = 200
    dt = 0.02

    for ax, sig in zip(axes, sigma_thetas):
        for name in fast_models:
            d = data[name]
            errs = []
            for i, r in enumerate(ratios):
                err, _ = fast_ensemble(
                    n_bugs=n_bugs, duration=duration, dt=dt,
                    kappa=2.0, sigma_theta=sig,
                    contrast=d['C'][i], n_cry=n_cry, sigma_sensor=0.02,
                    mean_yield=d['mean'][i])
                errs.append(err)
            ax.semilogx(ratios, errs, color=colors[name], marker='o', ms=4,
                        lw=2, label=name.replace('_', ' '))

        # Add literature baselines
        for C_lit, label, color, ls in [
            (0.15, '[FAD O₂] lit.', '#00BCD4', '--'),
            (0.01, '[FAD TrpH] lit.', '#FF9800', '--'),
        ]:
            err_lit, _ = fast_ensemble(
                n_bugs=n_bugs, duration=duration, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=C_lit, n_cry=n_cry, sigma_sensor=0.02)
            ax.axhline(err_lit, color=color, ls=ls, lw=1.5, alpha=0.5,
                       label=label)

        ax.axvline(1.0, color='grey', ls=':', lw=1, alpha=0.3)
        ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
        ax.set_xlabel(r'$k_T / k_S$', fontsize=11)
        ax.set_title(f'$\\sigma_\\theta = {sig}$', fontsize=12)
        ax.legend(fontsize=7, loc='upper right')

    axes[0].set_ylabel('Mean heading error (°)', fontsize=11)
    axes[0].set_ylim(0, 70)

    fig2.suptitle(f'Navigation vs Recombination Rate Asymmetry '
                  f'($N_{{cry}} = {n_cry}$)', fontsize=14)
    plt.tight_layout()

    if save_prefix:
        fig.savefig(f'{save_prefix}uneq_rates.png', dpi=150)
        fig2.savefig(f'{save_prefix}uneq_nav.png', dpi=150)
        print(f'Saved {save_prefix}uneq_rates.png and {save_prefix}uneq_nav.png')
    return fig, fig2


# ── 9. Orientational disorder ────────────────────────────────────

def orientational_disorder(save_prefix=None):
    """Effective contrast after averaging over cryptochrome misalignment.

    Since L=0,2 captures 99.9% of the anisotropy (harmonic analysis),
    orientational averaging simply multiplies the contrast by the P₂
    order parameter ⟨P₂(cos Δθ)⟩ of the angular distribution, where
    Δθ is the tilt of a molecule's z-axis from the mean orientation.

    For a von Mises-Fisher distribution on the sphere with concentration κ:
        ⟨P₂⟩ = 1 − 3/κ + 3 coth(κ)/κ  (exact)

    σ_orient ≈ 1/√κ gives the angular spread in radians.
    """
    models = _ensure_spin_dynamics()
    RPC = models['_RadicalPairCompass']

    # P₂ order parameter for von Mises-Fisher distribution
    def vmf_P2(kappa):
        """⟨P₂(cos θ)⟩ for von Mises-Fisher with concentration κ.

        ⟨P₂⟩ = 1 − 3L(κ)/κ  where L(κ) = coth(κ) − 1/κ (Langevin fn).
        Equivalently: 1 − 3coth(κ)/κ + 3/κ².
        Limits: κ→∞ gives 1 (perfect alignment), κ→0 gives 0 (isotropic).
        """
        if kappa < 0.01:
            return 0.0
        coth_k = 1.0 / np.tanh(kappa)
        L_k = coth_k - 1.0 / kappa   # Langevin function
        return 1.0 - 3.0 * L_k / kappa

    # Angular spread in degrees
    sigma_orient_deg = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 90])
    sigma_orient_rad = np.radians(sigma_orient_deg)

    # Convert σ to κ: for vMF, σ ≈ 1/√κ → κ ≈ 1/σ²
    # (this is an approximation; exact: E[cos θ] = coth(κ) - 1/κ ≈ 1 - 1/(2κ))
    with np.errstate(divide='ignore'):
        kappas = np.where(sigma_orient_rad > 0.01,
                          1.0 / sigma_orient_rad**2,
                          1e6)
    P2_values = np.array([vmf_P2(k) for k in kappas])

    print(f'{"σ_orient (°)":>14s}  {"κ":>10s}  {"⟨P₂⟩":>8s}')
    for s, k, p in zip(sigma_orient_deg, kappas, P2_values):
        print(f'{s:14d}  {k:10.1f}  {p:8.4f}')

    # Effective contrast = C₀ × ⟨P₂⟩ for each model
    model_configs = [
        ('toy FAD-O₂',  'toy_fad_o2',         '#2196F3'),
        ('toy FAD-TrpH', 'toy_fad_trp',        '#4CAF50'),
        ('inter FAD-O₂', 'intermediate_fad_o2', '#1565C0'),
    ]

    # Get unrelaxed contrasts
    C0 = {}
    mean0 = {}
    for label, name, _ in model_configs:
        factory = models[name]
        rpc = RPC(model=factory(), n_theta=90)
        C0[label] = rpc.contrast
        mean0[label] = rpc.mean_yield

    # ── Figure 1: Effective contrast vs disorder ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for label, name, color in model_configs:
        C_eff = C0[label] * P2_values
        ax1.plot(sigma_orient_deg, C_eff, color=color, marker='o', ms=5,
                 lw=2, label=f'{label} (C₀={C0[label]:.3f})')

    ax1.axhline(0.1, color='red', ls='--', lw=2, alpha=0.5,
                label='nav threshold')
    ax1.set_xlabel('Orientational disorder σ (°)', fontsize=12)
    ax1.set_ylabel('Effective contrast C_eff', fontsize=12)
    ax1.set_title('Contrast vs Cryptochrome Alignment Disorder', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 0.5)

    # ── Right panel: combined relaxation + disorder ──
    relax_scenarios = {
        'no relax': (0.0, 0.0),
        'T₂=3µs': (3.3e5, 3.3e5),
        'T₂=1µs': (1e6, 1e6),
    }
    relax_colors = {'no relax': '#4CAF50', 'T₂=3µs': '#FF9800', 'T₂=1µs': '#F44336'}

    # Use inter FAD-O₂ as representative
    factory = models['intermediate_fad_o2']
    for relax_label, (k_A, k_B) in relax_scenarios.items():
        rpc = RPC(model=factory(), k=1e6, k_relax_A=k_A, k_relax_B=k_B,
                  n_theta=90)
        C_eff = rpc.contrast * P2_values
        ax2.plot(sigma_orient_deg, C_eff,
                 color=relax_colors[relax_label], marker='o', ms=5, lw=2,
                 label=f'inter FAD-O₂ ({relax_label})')

    # Also FAD-TrpH with no relax
    rpc_trp = RPC(model=models['toy_fad_trp'](), n_theta=90)
    C_eff_trp = rpc_trp.contrast * P2_values
    ax2.plot(sigma_orient_deg, C_eff_trp, color='#4CAF50', ls='--',
             marker='s', ms=4, lw=1.5, label='toy FAD-TrpH (no relax)')

    ax2.axhline(0.1, color='red', ls='--', lw=2, alpha=0.5,
                label='nav threshold')
    ax2.set_xlabel('Orientational disorder σ (°)', fontsize=12)
    ax2.set_ylabel('Effective contrast C_eff', fontsize=12)
    ax2.set_title('Combined: Relaxation + Disorder', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 0.5)

    fig.suptitle('Orientational Disorder Suppresses Compass Contrast', fontsize=14)
    plt.tight_layout()

    # ── Figure 2: Navigation with combined suppression ──
    fig2, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    sigma_orients = [0, 15, 30]
    n_cry = 50
    n_bugs = 200
    duration = 200
    dt = 0.02
    sigma_range = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])

    for ax, s_orient in zip(axes, sigma_orients):
        kappa_o = 1.0 / np.radians(max(s_orient, 1))**2 if s_orient > 0 else 1e6
        P2 = vmf_P2(kappa_o)

        for label, name, color in model_configs:
            C_eff = C0[label] * P2
            mean = mean0[label]
            errs = []
            for sig in sigma_range:
                err, _ = fast_ensemble(
                    n_bugs=n_bugs, duration=duration, dt=dt,
                    kappa=2.0, sigma_theta=sig,
                    contrast=C_eff, n_cry=n_cry, sigma_sensor=0.02,
                    mean_yield=mean)
                errs.append(err)
            ax.plot(sigma_range, errs, color=color, marker='o', ms=4,
                    lw=2, label=f'{label} (C_eff={C_eff:.3f})')

        # Literature baselines
        for C_lit, lit_label, lit_color in [
            (0.15, '[FAD O₂] lit.', '#00BCD4'),
            (0.01, '[FAD TrpH] lit.', '#FF9800'),
        ]:
            errs = []
            for sig in sigma_range:
                err, _ = fast_ensemble(
                    n_bugs=n_bugs, duration=duration, dt=dt,
                    kappa=2.0, sigma_theta=sig,
                    contrast=C_lit, n_cry=n_cry, sigma_sensor=0.02)
                errs.append(err)
            ax.plot(sigma_range, errs, color=lit_color, ls='--', marker='x',
                    ms=5, lw=1.5, label=lit_label)

        ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
        ax.set_xlabel(r'$\sigma_\theta$ (rad/$\sqrt{s}$)', fontsize=11)
        ax.set_title(f'σ_orient = {s_orient}°  (⟨P₂⟩ = {P2:.2f})', fontsize=12)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='upper left')

    axes[0].set_ylabel('Mean heading error (°)', fontsize=11)
    axes[0].set_ylim(0, 95)

    fig2.suptitle(f'Navigation with Orientational Disorder ($N_{{cry}} = {n_cry}$)',
                  fontsize=14)
    plt.tight_layout()

    if save_prefix:
        fig.savefig(f'{save_prefix}orient_disorder.png', dpi=150)
        fig2.savefig(f'{save_prefix}orient_nav.png', dpi=150)
        print(f'Saved {save_prefix}orient_disorder.png and {save_prefix}orient_nav.png')
    return fig, fig2


# ── 10. Anomaly ensemble simulation ──────────────────────────────

def _build_deviation_grid(landscape, n_grid=100):
    """Pre-compute δφ on a regular grid for fast lookup.

    Returns (x_edges, y_edges, dphi_grid) where dphi_grid[j, i] is the
    field direction deviation at (x_edges[i], y_edges[j]).
    """
    w, h = landscape.extent
    xg = np.linspace(0, w, n_grid)
    yg = np.linspace(0, h, n_grid)
    Xg, Yg = np.meshgrid(xg, yg)
    dphi = landscape.direction_deviation(Xg, Yg)
    return xg, yg, dphi


def _interp_deviation(x, y, xg, yg, dphi_grid):
    """Bilinear interpolation of δφ from pre-computed grid.

    x, y are arrays of bug positions.  Out-of-bounds positions are clamped.
    """
    w = xg[-1]
    h = yg[-1]
    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    n_x = len(xg) - 1
    n_y = len(yg) - 1

    # Grid indices (clamped)
    xi = np.clip((x - xg[0]) / dx, 0, n_x - 1e-10)
    yi = np.clip((y - yg[0]) / dy, 0, n_y - 1e-10)
    ix = xi.astype(int)
    iy = yi.astype(int)
    fx = xi - ix
    fy = yi - iy

    # Bilinear
    return ((1-fx)*(1-fy) * dphi_grid[iy, ix] +
            fx*(1-fy)     * dphi_grid[iy, ix+1] +
            (1-fx)*fy     * dphi_grid[iy+1, ix] +
            fx*fy         * dphi_grid[iy+1, ix+1])


def anomaly_ensemble(n_bugs, duration, dt, kappa, sigma_theta,
                     contrast, n_cry, sigma_sensor, landscape,
                     goal=3*np.pi/4, speed=1.0, sigma_xy=0.05, seed=0,
                     mean_yield=None):
    """Vectorised simulation with position-dependent field direction.

    Like fast_ensemble but the local magnetic field direction varies with
    position according to the landscape's anomaly field.  The field
    deviation δφ(x,y) is pre-computed on a grid and bilinearly interpolated
    for speed.

    Returns (mean_heading_error_deg, distances, mean_path_deviation_deg).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(duration / dt)
    sqrt_dt = np.sqrt(dt)

    theta = rng.uniform(0, 2*np.pi, n_bugs)
    x = np.full(n_bugs, 500.0)
    y = np.full(n_bugs, 100.0)

    if mean_yield is None:
        mean_yield = 0.5
    delta = contrast * mean_yield
    n_per_ch = n_cry / 8.0
    sigma_compass = (sigma_sensor / (delta * np.sqrt(2 * n_per_ch))
                     if delta > 1e-10 else 10.0)

    # Pre-compute deviation grid (fast lookup instead of per-step anomaly eval)
    xg, yg, dphi_grid = _build_deviation_grid(landscape, n_grid=150)

    heading_errors_sum = np.zeros(n_bugs)
    deviation_sum = np.zeros(n_bugs)

    for _ in range(n_steps):
        # Fast grid lookup of field deviation
        delta_phi = _interp_deviation(x, y, xg, yg, dphi_grid)

        # Compass noise
        compass_noise = rng.normal(0, sigma_compass, n_bugs)

        # Steering: the anomaly biases the heading estimate
        heading_est = theta + compass_noise
        heading_error = goal - heading_est + delta_phi
        d_theta = kappa * np.sin(heading_error) * dt
        d_theta += sigma_theta * sqrt_dt * rng.standard_normal(n_bugs)
        theta = (theta + d_theta) % (2 * np.pi)

        # Position
        x += speed * np.cos(theta) * dt + sigma_xy * sqrt_dt * rng.standard_normal(n_bugs)
        y += speed * np.sin(theta) * dt + sigma_xy * sqrt_dt * rng.standard_normal(n_bugs)

        # Track errors
        err = np.abs(((theta - goal) + np.pi) % (2*np.pi) - np.pi)
        heading_errors_sum += err
        deviation_sum += np.abs(delta_phi)

    mean_err = np.degrees(np.mean(heading_errors_sum / n_steps))
    distances = np.sqrt((x - 500)**2 + (y - 100)**2)
    mean_dev = np.degrees(np.mean(deviation_sum / n_steps))

    return mean_err, distances, mean_dev


# ── 11. Direction A: Magnetic anomaly analysis ───────────────────

def anomaly_navigation(save_prefix=None):
    """Navigation through magnetic anomaly fields.

    Produces four figures:
      1. Field deviation maps for each anomaly type
      2. Single-bug trajectory through a dipole anomaly
      3. Navigation error vs dipole strength at varying densities
      4. Critical anomaly magnitude (phase diagram)
    """
    # ── Figure 1: Anomaly field maps ──
    print('  [1/4] Field deviation maps...')
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))

    extent = (1000, 1000)
    xg = np.linspace(0, extent[0], 200)
    yg = np.linspace(0, extent[1], 200)
    Xg, Yg = np.meshgrid(xg, yg)

    # Background field: B_h ≈ 21.1 μT at 65° inclination
    anomaly_configs = [
        ('Dipole (5 μT, depth=100 BL)',
         [{'type': 'dipole', 'pos': (500, 500), 'strength': 5.0, 'depth': 100}]),
        ('Fault (2 μT, NE strike)',
         [{'type': 'fault', 'pos': (500, 500), 'azimuth': np.pi/4,
           'contrast': 2.0, 'width': 30}]),
        ('Gradient (0.005 μT/BL, northward)',
         [{'type': 'gradient', 'magnitude': 0.005, 'direction': 0.0}]),
        ('Combined: 3 dipoles + fault',
         [{'type': 'dipole', 'pos': (300, 400), 'strength': 3.0, 'depth': 80},
          {'type': 'dipole', 'pos': (700, 600), 'strength': -4.0, 'depth': 120},
          {'type': 'dipole', 'pos': (500, 800), 'strength': 2.0, 'depth': 60},
          {'type': 'fault', 'pos': (500, 500), 'azimuth': np.pi/3,
           'contrast': 1.5, 'width': 25}]),
    ]

    for ax, (title, anoms) in zip(axes1.flat, anomaly_configs):
        ls = Landscape(extent=extent, anomalies=anoms)
        dphi = ls.direction_deviation(Xg, Yg)
        dphi_deg = np.degrees(dphi)

        vmax = max(np.abs(dphi_deg).max(), 0.5)
        im = ax.pcolormesh(Xg, Yg, dphi_deg, cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, shading='auto')
        plt.colorbar(im, ax=ax, label='δφ (°)')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x (BL)')
        ax.set_ylabel('y (BL)')
        ax.set_aspect('equal')

    fig1.suptitle('Magnetic Field Direction Anomalies', fontsize=14)
    plt.tight_layout()

    # ── Figure 2: Trajectory through a dipole ──
    print('  [2/4] Trajectory through dipole...')
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    strengths_traj = [0.0, 2.0, 8.0]
    n_bugs_traj = 50
    duration = 300
    dt = 0.02
    contrast = 0.15
    n_cry = 50

    for ax, s_dip in zip(axes2, strengths_traj):
        if s_dip > 0:
            anoms = [{'type': 'dipole', 'pos': (500, 400),
                      'strength': s_dip, 'depth': 80}]
        else:
            anoms = []
        ls = Landscape(extent=extent, anomalies=anoms)

        # Plot field deviation background
        dphi = ls.direction_deviation(Xg, Yg)
        dphi_deg = np.degrees(dphi)
        vmax = max(np.abs(dphi_deg).max(), 0.5)
        ax.pcolormesh(Xg, Yg, dphi_deg, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, shading='auto', alpha=0.3)

        # Run bugs
        rng = np.random.default_rng(42)
        n_steps = int(duration / dt)
        sqrt_dt = np.sqrt(dt)
        goal = 3*np.pi/4

        delta_val = contrast * 0.5
        n_per_ch = n_cry / 8.0
        sigma_compass = 0.02 / (delta_val * np.sqrt(2 * n_per_ch))

        theta = rng.uniform(0, 2*np.pi, n_bugs_traj)
        x_arr = np.full(n_bugs_traj, 500.0)
        y_arr = np.full(n_bugs_traj, 100.0)
        paths_x = [x_arr.copy()]
        paths_y = [y_arr.copy()]

        for _ in range(n_steps):
            dp = ls.direction_deviation(x_arr, y_arr)
            cn = rng.normal(0, sigma_compass, n_bugs_traj)
            he = goal - (theta + cn) + dp
            theta = (theta + 2.0 * np.sin(he) * dt +
                     0.3 * sqrt_dt * rng.standard_normal(n_bugs_traj)) % (2*np.pi)
            x_arr += 1.0 * np.cos(theta) * dt + 0.05 * sqrt_dt * rng.standard_normal(n_bugs_traj)
            y_arr += 1.0 * np.sin(theta) * dt + 0.05 * sqrt_dt * rng.standard_normal(n_bugs_traj)
            if _ % 50 == 0:
                paths_x.append(x_arr.copy())
                paths_y.append(y_arr.copy())

        paths_x = np.array(paths_x)
        paths_y = np.array(paths_y)

        for i in range(n_bugs_traj):
            ax.plot(paths_x[:, i], paths_y[:, i], 'k-', alpha=0.15, lw=0.5)
        # Highlight a few
        for i in range(min(5, n_bugs_traj)):
            ax.plot(paths_x[:, i], paths_y[:, i], lw=1.5, alpha=0.7)

        # Goal direction arrow
        ax.annotate('', xy=(500 + 150*np.cos(goal), 100 + 150*np.sin(goal)),
                    xytext=(500, 100),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.set_aspect('equal')
        ax.set_title(f'Dipole: {s_dip:.0f} μT' if s_dip > 0 else 'No anomaly',
                     fontsize=12)
        ax.set_xlabel('x (BL)')

    axes2[0].set_ylabel('y (BL)')
    fig2.suptitle('Bug Trajectories Through Dipole Anomalies '
                  f'(C={contrast}, N_cry={n_cry})', fontsize=14)
    plt.tight_layout()

    # ── Figure 3: Navigation error vs anomaly strength × density ──
    print('  [3/4] Error vs strength × density sweep...')
    dipole_strengths = np.array([0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    n_dipoles_list = [1, 5, 10, 20]
    depth = 80
    n_bugs = 200
    duration = 300
    dt = 0.02
    sigma_theta = 0.3

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    colors3 = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    # Left: error vs strength at different densities
    ax = axes3[0]
    for n_dip, color in zip(n_dipoles_list, colors3):
        errs = []
        for s in dipole_strengths:
            if s == 0 or n_dip == 0:
                # Uniform field baseline
                err, _ = fast_ensemble(
                    n_bugs=n_bugs, duration=duration, dt=dt,
                    kappa=2.0, sigma_theta=sigma_theta,
                    contrast=0.15, n_cry=50, sigma_sensor=0.02)
                errs.append(err)
            else:
                # Average over 5 random realisations
                err_trials = []
                for trial in range(5):
                    rng = np.random.default_rng(100*trial + n_dip)
                    anoms = Landscape.random_dipoles(
                        n_dip, extent, s, depth, rng=rng)
                    ls = Landscape(extent=extent, anomalies=anoms)
                    err_t, _, _ = anomaly_ensemble(
                        n_bugs=n_bugs, duration=duration, dt=dt,
                        kappa=2.0, sigma_theta=sigma_theta,
                        contrast=0.15, n_cry=50, sigma_sensor=0.02,
                        landscape=ls, seed=trial)
                    err_trials.append(err_t)
                errs.append(np.mean(err_trials))
            print(f'    n_dip={n_dip}  strength={s:.1f}  err={errs[-1]:.1f}°')
        ax.plot(dipole_strengths, errs, color=color, marker='o', ms=5, lw=2,
                label=f'{n_dip} dipole{"s" if n_dip>1 else ""}')

    ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
    ax.set_xlabel('Dipole strength (μT)', fontsize=12)
    ax.set_ylabel('Mean heading error (°)', fontsize=12)
    ax.set_title('Dipole Strength vs Navigation Error', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 60)

    # Right: error vs density at different strengths
    ax = axes3[1]
    strength_list = [1.0, 3.0, 5.0, 8.0]
    n_dip_range = np.array([0, 1, 3, 5, 10, 20, 50])
    colors3b = ['#2196F3', '#FF9800', '#F44336', '#9C27B0']

    for s, color in zip(strength_list, colors3b):
        errs = []
        for n_dip in n_dip_range:
            if n_dip == 0:
                err, _ = fast_ensemble(
                    n_bugs=n_bugs, duration=duration, dt=dt,
                    kappa=2.0, sigma_theta=sigma_theta,
                    contrast=0.15, n_cry=50, sigma_sensor=0.02)
                errs.append(err)
            else:
                err_trials = []
                for trial in range(5):
                    rng = np.random.default_rng(200*trial + int(s*10))
                    anoms = Landscape.random_dipoles(
                        n_dip, extent, s, depth, rng=rng)
                    ls = Landscape(extent=extent, anomalies=anoms)
                    err_t, _, _ = anomaly_ensemble(
                        n_bugs=n_bugs, duration=duration, dt=dt,
                        kappa=2.0, sigma_theta=sigma_theta,
                        contrast=0.15, n_cry=50, sigma_sensor=0.02,
                        landscape=ls, seed=trial)
                    err_trials.append(err_t)
                errs.append(np.mean(err_trials))
            print(f'    strength={s:.1f}  n_dip={n_dip}  err={errs[-1]:.1f}°')
        ax.plot(n_dip_range, errs, color=color, marker='s', ms=5, lw=2,
                label=f'{s:.0f} μT')

    ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4)
    ax.set_xlabel('Number of dipole anomalies', fontsize=12)
    ax.set_ylabel('Mean heading error (°)', fontsize=12)
    ax.set_title('Anomaly Density vs Navigation Error', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 60)

    fig3.suptitle('Navigation Degradation from Magnetic Anomalies '
                  f'(C=0.15, σ_θ={sigma_theta})', fontsize=14)
    plt.tight_layout()

    # ── Figure 4: Critical anomaly magnitude for different compasses ──
    print('  [4/4] Critical anomaly magnitude...')
    contrasts_test = [
        ('FAD-O₂ lit. (C=0.15)', 0.15, None),
        ('FAD-TrpH lit. (C=0.01)', 0.01, None),
        ('Relaxed FAD-O₂ (C=0.10)', 0.10, None),
    ]
    n_dip_crit = 10
    depth_crit = 80
    strengths_fine = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    sigma_thetas_crit = [0.3, 0.5]

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    compass_colors = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, sig in zip(axes4, sigma_thetas_crit):
        # Baseline (no anomaly) for each compass
        baselines = {}
        for label, C, my in contrasts_test:
            err_base, _ = fast_ensemble(
                n_bugs=200, duration=300, dt=0.02,
                kappa=2.0, sigma_theta=sig,
                contrast=C, n_cry=50, sigma_sensor=0.02,
                mean_yield=my)
            baselines[label] = err_base

        for (label, C, my), color in zip(contrasts_test, compass_colors):
            errs = []
            for s in strengths_fine:
                if s == 0:
                    errs.append(baselines[label])
                else:
                    trials = []
                    for trial in range(5):
                        rng = np.random.default_rng(300*trial)
                        anoms = Landscape.random_dipoles(
                            n_dip_crit, extent, s, depth_crit, rng=rng)
                        ls = Landscape(extent=extent, anomalies=anoms)
                        err_t, _, _ = anomaly_ensemble(
                            n_bugs=200, duration=300, dt=0.02,
                            kappa=2.0, sigma_theta=sig,
                            contrast=C, n_cry=50, sigma_sensor=0.02,
                            landscape=ls, seed=trial, mean_yield=my)
                        trials.append(err_t)
                    errs.append(np.mean(trials))
                print(f'    {label}  σ_θ={sig}  s={s:.1f}  '
                      f'err={errs[-1]:.1f}° (Δ={errs[-1]-baselines[label]:+.1f}°)')

            ax.plot(strengths_fine, errs, color=color, marker='o', ms=5,
                    lw=2, label=label)
            ax.axhline(baselines[label], color=color, ls=':', lw=1, alpha=0.3)

        ax.axhline(30, color='orange', ls=':', lw=1, alpha=0.4,
                   label='30° threshold')
        ax.set_xlabel('Dipole anomaly strength (μT)', fontsize=12)
        ax.set_ylabel('Mean heading error (°)', fontsize=12)
        ax.set_title(f'σ_θ = {sig} rad/√s', fontsize=13)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 70)

    fig4.suptitle(f'Critical Anomaly Magnitude by Compass Model '
                  f'({n_dip_crit} dipoles, depth={depth_crit} BL)',
                  fontsize=14)
    plt.tight_layout()

    if save_prefix:
        fig1.savefig(f'{save_prefix}anomaly_maps.png', dpi=150)
        fig2.savefig(f'{save_prefix}anomaly_traj.png', dpi=150)
        fig3.savefig(f'{save_prefix}anomaly_sweep.png', dpi=150)
        fig4.savefig(f'{save_prefix}anomaly_critical.png', dpi=150)
        print(f'Saved {save_prefix}anomaly_*.png')

    return fig1, fig2, fig3, fig4


# ── 12. Direction B: Path integration homing ─────────────────────

def pi_homing_ensemble(n_bugs, T_out, T_home, dt, kappa, sigma_theta,
                       contrast, n_cry, sigma_sensor, bias=0.0,
                       landscape=None, goal_out=3*np.pi/4, speed=1.0,
                       sigma_xy=0.05, seed=0, mean_yield=None,
                       use_pi=True, leak=0.0, mode='straight'):
    """Vectorised two-phase homing task.

    Two modes:
      'straight': outbound along fixed heading, then home
      'explore':  random foraging walk, then home via PI

    For 'straight' mode:
      Phase 1: navigate toward goal_out for T_out seconds
      Phase 2 (use_pi=True): steer toward CPU4 home vector
      Phase 2 (use_pi=False): reverse heading for T_home seconds

    For 'explore' mode:
      Phase 1: correlated random walk (no goal heading) for T_out seconds
      Phase 2: PI homing for T_home seconds (reversal is meaningless)

    CPU4 integrates the heading estimate (with noise + bias), not the
    true heading.  Compass noise averages out (√N benefit); systematic
    bias accumulates as rotation of the estimated displacement vector.
    For straight out-and-back with constant bias, the biases cancel
    (a null result). For exploration, they compound.

    Returns
    -------
    homing_errors : ndarray, shape (n_bugs,)
        Final distance from start (BL) per bug.
    mean_homing : float
        Mean distance from start (BL).
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    # Compass noise
    if mean_yield is None:
        mean_yield = 0.5
    delta = contrast * mean_yield
    n_per_ch = n_cry / 8.0
    sigma_compass = (sigma_sensor / (delta * np.sqrt(2 * n_per_ch))
                     if delta > 1e-10 else 10.0)

    # State
    x0, y0 = 500.0, 500.0
    if mode == 'straight':
        theta = np.full(n_bugs, goal_out)
    else:
        theta = rng.uniform(0, 2*np.pi, n_bugs)
    x = np.full(n_bugs, x0)
    y = np.full(n_bugs, y0)

    # CPU4 memory: (n_bugs, 8)
    n_cpu4 = 8
    cpu4_phi = np.linspace(0, 2*np.pi, n_cpu4, endpoint=False)
    memory = np.zeros((n_bugs, n_cpu4))

    # Landscape deviation grid (if anomalies)
    if landscape is not None and landscape.anomalies:
        xg, yg, dphi_grid = _build_deviation_grid(landscape, n_grid=150)
        has_anomalies = True
    else:
        has_anomalies = False

    def _do_step(goal_heading=None, use_home_vec=False, free_walk=False):
        """One timestep. Modifies theta, x, y, memory via closure."""
        nonlocal theta, x, y, memory

        # Field deviation from anomalies
        if has_anomalies:
            delta_phi = _interp_deviation(x, y, xg, yg, dphi_grid)
        else:
            delta_phi = 0.0

        # Compass: biased + noisy heading estimate in the LOCAL field frame.
        # The compass reads the angle between the body axis and the local
        # magnetic field direction.  In an anomaly field, the local direction
        # differs from the background by delta_phi(x,y).
        compass_noise = rng.normal(0, sigma_compass, n_bugs)
        heading_est = theta + compass_noise + bias + delta_phi

        # CPU4 update — integrates the compass-estimated heading, which
        # includes the spatially-varying anomaly bias.  For constant bias,
        # this rotates the CPU4 frame uniformly (cancels at readout).
        # For spatially-varying bias, the rotation is position-dependent
        # and does NOT cancel — this is where A×B matters.
        cos_vals = np.cos(heading_est[:, None] - cpu4_phi[None, :])
        drive = speed * np.maximum(cos_vals, 0.0) * dt
        if leak > 0:
            memory *= (1.0 - leak * dt)
        memory += drive

        # Steering
        if free_walk:
            # Pure random walk — no steering goal
            d_theta = sigma_theta * sqrt_dt * rng.standard_normal(n_bugs)
        elif use_home_vec:
            disp_x = np.sum(memory * np.cos(cpu4_phi), axis=1)
            disp_y = np.sum(memory * np.sin(cpu4_phi), axis=1)
            home_dir = np.arctan2(-disp_y, -disp_x)
            heading_error = home_dir - heading_est
            d_theta = kappa * np.sin(heading_error) * dt
            d_theta += sigma_theta * sqrt_dt * rng.standard_normal(n_bugs)
        else:
            # Goal is in geographic coordinates; heading_est already
            # includes delta_phi, so no extra correction needed.
            heading_error = goal_heading - heading_est
            d_theta = kappa * np.sin(heading_error) * dt
            d_theta += sigma_theta * sqrt_dt * rng.standard_normal(n_bugs)

        theta = (theta + d_theta) % (2 * np.pi)
        x += speed * np.cos(theta) * dt + sigma_xy * sqrt_dt * rng.standard_normal(n_bugs)
        y += speed * np.sin(theta) * dt + sigma_xy * sqrt_dt * rng.standard_normal(n_bugs)

    # Phase 1: outbound / exploration
    n_out = int(T_out / dt)
    if mode == 'explore':
        for _ in range(n_out):
            _do_step(free_walk=True)
    else:
        for _ in range(n_out):
            _do_step(goal_heading=goal_out)

    # Phase 2: homing
    n_home = int(T_home / dt)
    if use_pi:
        for _ in range(n_home):
            _do_step(use_home_vec=True)
    else:
        goal_return = (goal_out + np.pi) % (2 * np.pi)
        for _ in range(n_home):
            _do_step(goal_heading=goal_return)

    homing_errors = np.sqrt((x - x0)**2 + (y - y0)**2)
    return homing_errors, np.mean(homing_errors)


def path_integration_analysis(save_prefix=None):
    """Direction B: Path integration — what breaks the CPU4 home vector?

    Key findings from first-pass analysis:
    - Constant bias cancels perfectly in the CPU4 circuit: integration
      and readout share the same biased compass frame.  The displacement
      vector is rotated, but so is the homing readout → exact cancellation
      regardless of path tortuosity.
    - PI works remarkably well for short foraging bouts (200s → ~2 BL error).
    - The real threats to PI are: (a) memory leak/decay, (b) compass noise
      (via low contrast), (c) spatially varying anomalies (Direction A×B).

    Produces four figures:
      1. PI vs reversal (straight) + exploration PI vs T_explore
      2. Bias null result + CPU4 leak sweep (what doesn't vs does matter)
      3. Phase diagram: leak × T_explore → PI accuracy (real structure)
      4. Compass model comparison: FAD-O₂ vs FAD-TrpH PI performance
    """
    dt = 0.05
    n_bugs = 200
    speed = 1.0

    # ── Figure 1: PI enables homing ──
    print('  [1/4] PI vs reversal + exploration duration...')
    T_out = 150.0
    sigma_range = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: straight homing, PI vs reversal
    ax = axes1[0]
    for use_pi, label, color, marker in [
        (True, 'Path integration', '#2196F3', 'o'),
        (False, 'Heading reversal', '#FF9800', 's'),
    ]:
        errs = []
        for sig in sigma_range:
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_out, T_home=T_out, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                use_pi=use_pi, seed=42, mode='straight')
            errs.append(mean_err)
            print(f'    {label}  σ_θ={sig:.2f}  err={mean_err:.1f} BL')
        ax.plot(sigma_range, errs, color=color, marker=marker, ms=6,
                lw=2, label=label)
    ax.set_xlabel(r'$\sigma_\theta$ (rad/$\sqrt{s}$)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title(f'Straight out-and-back (T={T_out:.0f}s)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    # Right: exploration PI vs duration (enough homing time)
    ax = axes1[1]
    T_explores = np.array([50, 100, 200, 500, 1000, 2000])
    for sig, color, marker in [
        (0.3, '#2196F3', 'o'),
        (0.5, '#4CAF50', 's'),
        (1.0, '#FF9800', '^'),
    ]:
        errs = []
        for T_ex in T_explores:
            T_h = max(T_ex, 200)  # enough time to return
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_ex, T_home=T_h, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore')
            errs.append(mean_err)
        print(f'    explore σ={sig}  errs={[f"{e:.1f}" for e in errs]}')
        ax.plot(T_explores, errs, color=color, marker=marker, ms=6,
                lw=2, label=f'σ_θ={sig}')
    ax.set_xlabel('Exploration time (s)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title('Exploration + PI homing (leak=0)', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    fig1.suptitle('Path Integration Enables Homing from Exploration',
                  fontsize=14)
    plt.tight_layout()

    # ── Figure 2: What doesn't matter (bias) vs what does (leak) ──
    print('  [2/4] Bias null result + leak sweep...')
    T_explore = 500.0
    T_home = 500.0

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bias is irrelevant (the flat line IS the result)
    ax = axes2[0]
    bias_range_deg = np.array([0, 2, 5, 10, 15, 20, 30, 45])
    bias_range = np.radians(bias_range_deg)
    colors2 = ['#2196F3', '#4CAF50', '#FF9800']
    sigma_thetas_bias = [0.3, 0.5, 1.0]
    for sig, color in zip(sigma_thetas_bias, colors2):
        errs = []
        for b in bias_range:
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                bias=b, use_pi=True, seed=42, mode='explore')
            errs.append(mean_err)
        print(f'    bias null  σ={sig}  '
              f'range=[{errs[0]:.1f}, {errs[-1]:.1f}] BL')
        ax.plot(bias_range_deg, errs, color=color, marker='o',
                ms=5, lw=2, label=f'σ_θ={sig}')
    ax.set_xlabel('Constant compass bias (°)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title('Bias cancels (same-frame integration)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.text(0.5, 0.95, 'CPU4 integrates & reads\nin the same biased frame',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            style='italic', color='#666')

    # Right: leak matters
    ax = axes2[1]
    leak_rates = np.array([0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])
    for sig, color in zip(sigma_thetas_bias, colors2):
        errs = []
        for lk in leak_rates:
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                leak=lk, use_pi=True, seed=42, mode='explore')
            errs.append(mean_err)
            print(f'    leak={lk:.4f}  σ={sig}  err={mean_err:.1f} BL')
        leak_plot = np.array([max(lk, 1e-4) for lk in leak_rates])
        ax.plot(leak_plot, errs, color=color, marker='s', ms=5, lw=2,
                label=f'σ_θ={sig}')
    ax.set_xlabel('CPU4 leak rate λ (1/s)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title(f'Memory decay breaks PI (T_explore={T_explore:.0f}s)',
                 fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    # Integration window annotation
    ax.axvline(1.0/T_explore, color='gray', ls='--', alpha=0.5)
    ax.text(1.0/T_explore, ax.get_ylim()[1]*0.9, f'  λ=1/T_ex',
            color='gray', fontsize=9)

    fig2.suptitle('Constant Bias Is Irrelevant; Memory Leak Is Not',
                  fontsize=14)
    plt.tight_layout()

    # ── Figure 3: Phase diagram — leak × T_explore ──
    print('  [3/4] Phase diagram: leak × exploration time...')
    T_explore_grid = np.array([100, 200, 500, 1000, 2000])
    leak_grid = np.array([0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])
    sig_phase = 0.5  # moderate angular noise

    err_phase = np.zeros((len(leak_grid), len(T_explore_grid)))

    for i, lk in enumerate(leak_grid):
        for j, T_ex in enumerate(T_explore_grid):
            T_h = max(T_ex, 200)
            _, err_phase[i, j] = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_ex, T_home=T_h, dt=dt,
                kappa=2.0, sigma_theta=sig_phase,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                leak=lk, use_pi=True, seed=42, mode='explore')
            print(f'    leak={lk:.4f}  T_ex={T_ex}  '
                  f'err={err_phase[i,j]:.1f} BL')

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute error
    leak_plot_grid = np.array([max(lk, 2e-4) for lk in leak_grid])
    im0 = axes3[0].pcolormesh(T_explore_grid, leak_plot_grid, err_phase,
                               cmap='YlOrRd', shading='auto')
    plt.colorbar(im0, ax=axes3[0], label='Homing error (BL)')
    axes3[0].set_xlabel('Exploration time (s)', fontsize=12)
    axes3[0].set_ylabel('CPU4 leak rate λ (1/s)', fontsize=12)
    axes3[0].set_xscale('log')
    axes3[0].set_yscale('log')
    axes3[0].set_title(f'PI Homing Error (σ_θ={sig_phase})', fontsize=13)
    # Critical line: λ T_explore = 1
    t_crit = np.linspace(100, 2000, 50)
    axes3[0].plot(t_crit, 1.0/t_crit, 'w--', lw=2, alpha=0.7,
                  label='λT=1')
    axes3[0].legend(fontsize=10, loc='lower left')

    # Right: normalised by leak=0 baseline
    baseline = err_phase[0, :]  # leak=0 row
    ratio = err_phase / np.maximum(baseline[None, :], 1e-3)
    im1 = axes3[1].pcolormesh(T_explore_grid, leak_plot_grid, ratio,
                               cmap='YlOrRd', shading='auto',
                               vmin=1.0, vmax=max(ratio.max(), 3.0))
    plt.colorbar(im1, ax=axes3[1], label='Ratio to leak=0')
    axes3[1].set_xlabel('Exploration time (s)', fontsize=12)
    axes3[1].set_ylabel('CPU4 leak rate λ (1/s)', fontsize=12)
    axes3[1].set_xscale('log')
    axes3[1].set_yscale('log')
    axes3[1].set_title('Relative Degradation from Leak', fontsize=13)
    axes3[1].plot(t_crit, 1.0/t_crit, 'k--', lw=2, alpha=0.7,
                  label='λT=1')
    axes3[1].legend(fontsize=10, loc='lower left')
    try:
        axes3[1].contour(T_explore_grid, leak_plot_grid, ratio,
                         levels=[1.5, 2.0, 3.0],
                         colors=['orange', 'red', 'darkred'],
                         linewidths=2)
    except Exception:
        pass

    fig3.suptitle('Phase Diagram: When Does Memory Decay Break Homing?',
                  fontsize=14)
    plt.tight_layout()

    # ── Figure 4: Compass model comparison for PI ──
    print('  [4/4] Compass model comparison...')
    compass_configs = [
        ('FAD-O₂ (C=0.15)', 0.15, None),
        ('Relaxed FAD-O₂ (C=0.10)', 0.10, None),
        ('FAD-TrpH (C=0.01)', 0.01, None),
    ]
    compass_colors = ['#2196F3', '#4CAF50', '#FF9800']
    T_explore = 500.0
    T_home = 500.0

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: n_cry sweep (how many cryptochromes needed for PI?)
    ax = axes4[0]
    n_cry_range = np.array([5, 10, 20, 50, 100, 200, 500])
    for (label, C, my), color in zip(compass_configs, compass_colors):
        errs = []
        for nc in n_cry_range:
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=0.5,
                contrast=C, n_cry=nc, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore',
                mean_yield=my)
            errs.append(mean_err)
            print(f'    {label}  n_cry={nc}  err={mean_err:.1f} BL')
        ax.plot(n_cry_range, errs, color=color, marker='o', ms=5, lw=2,
                label=label)
    ax.set_xlabel('Number of cryptochromes', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title(f'PI needs good compass (T_ex={T_explore:.0f}s)', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # Right: exploration duration sweep per model
    ax = axes4[1]
    T_explore_range = np.array([100, 200, 500, 1000, 2000])
    for (label, C, my), color in zip(compass_configs, compass_colors):
        errs = []
        for T_ex in T_explore_range:
            T_h = max(T_ex, 200)
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_ex, T_home=T_h, dt=dt,
                kappa=2.0, sigma_theta=0.5,
                contrast=C, n_cry=50, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore',
                mean_yield=my)
            errs.append(mean_err)
        print(f'    {label}  errs={[f"{e:.1f}" for e in errs]}')
        ax.plot(T_explore_range, errs, color=color, marker='o', ms=5, lw=2,
                label=label)
    ax.set_xlabel('Exploration time (s)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title('PI vs exploration duration (n_cry=50)', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    fig4.suptitle('Compass Quality Sets the PI Performance Floor',
                  fontsize=14)
    plt.tight_layout()

    if save_prefix:
        fig1.savefig(f'{save_prefix}pi_homing.png', dpi=150)
        fig2.savefig(f'{save_prefix}pi_bias_leak.png', dpi=150)
        fig3.savefig(f'{save_prefix}pi_phase.png', dpi=150)
        fig4.savefig(f'{save_prefix}pi_compass.png', dpi=150)
        print(f'Saved {save_prefix}pi_*.png')

    return fig1, fig2, fig3, fig4


def anomaly_pi_analysis(save_prefix=None):
    """Direction A×B: anomalies + path integration.

    Spatially varying compass bias breaks the same-frame cancellation
    that makes constant bias irrelevant (Direction B).  The bug
    encounters different delta_phi(x,y) during outbound exploration
    and return homing, so the CPU4 memory accumulates a net rotated
    displacement vector that does NOT cancel at readout.

    Produces four figures:
      1. PI homing error vs anomaly strength + heading-follower comparison
      2. The cancellation breakdown: constant bias vs spatial anomaly
      3. Phase diagram: anomaly strength × T_explore
      4. Compass model comparison with anomalies
    """
    dt = 0.05
    n_bugs = 200
    extent = (1000, 1000)
    depth = 100.0  # dipole burial depth (BL)

    # ── Figure 1: PI homing error vs anomaly strength ──
    print('  [1/4] PI homing vs anomaly strength...')
    strengths = np.array([0, 0.5, 1, 2, 3, 5, 8, 12])
    n_dip = 10
    T_explore = 500.0
    T_home = 500.0

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PI homing error vs dipole strength
    ax = axes1[0]
    for sig, color, marker in [
        (0.3, '#2196F3', 'o'),
        (0.5, '#4CAF50', 's'),
        (1.0, '#FF9800', '^'),
    ]:
        errs = []
        for s in strengths:
            if s == 0:
                land = Landscape(extent=extent)
            else:
                land = Landscape(extent=extent)
                rng_land = np.random.default_rng(123)
                land.anomalies = Landscape.random_dipoles(
                    n_dip, extent, s, depth, rng=rng_land)
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore',
                landscape=land)
            errs.append(mean_err)
            print(f'    PI  σ={sig}  s={s}  err={mean_err:.1f} BL')
        ax.plot(strengths, errs, color=color, marker=marker, ms=6,
                lw=2, label=f'PI (σ_θ={sig})')

    # Overlay heading-follower from Direction A for comparison (σ=0.3 only)
    heading_errs = []
    for s in strengths:
        if s == 0:
            land = Landscape(extent=extent)
        else:
            land = Landscape(extent=extent)
            rng_land = np.random.default_rng(123)
            land.anomalies = Landscape.random_dipoles(
                n_dip, extent, s, depth, rng=rng_land)
        mean_err_deg, _, _ = anomaly_ensemble(
            n_bugs=n_bugs, duration=150, dt=dt,
            kappa=2.0, sigma_theta=0.3,
            contrast=0.15, n_cry=50, sigma_sensor=0.02,
            landscape=land, seed=42)
        heading_errs.append(mean_err_deg)
    ax.plot(strengths, heading_errs, color='#9C27B0', marker='d',
            ms=6, lw=2, ls='--', label='Heading-follower (σ_θ=0.3)')

    ax.set_xlabel('Dipole peak anomaly (μT)', fontsize=12)
    ax.set_ylabel('Error (BL or ° for heading)', fontsize=12)
    ax.set_title(f'{n_dip} dipoles, T_explore={T_explore:.0f}s', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # Right: more dipoles
    ax = axes1[1]
    n_dips_right = [5, 10, 20, 50]
    colors_nd = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for nd, color in zip(n_dips_right, colors_nd):
        errs = []
        for s in strengths:
            if s == 0:
                land = Landscape(extent=extent)
            else:
                land = Landscape(extent=extent)
                rng_land = np.random.default_rng(123)
                land.anomalies = Landscape.random_dipoles(
                    nd, extent, s, depth, rng=rng_land)
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=0.5,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore',
                landscape=land)
            errs.append(mean_err)
        print(f'    n_dip={nd}  errs={[f"{e:.1f}" for e in errs]}')
        ax.plot(strengths, errs, color=color, marker='o', ms=5, lw=2,
                label=f'n_dip={nd}')
    ax.set_xlabel('Dipole peak anomaly (μT)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title(f'PI homing vs dipole density (σ_θ=0.5)', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    fig1.suptitle('Anomalies Degrade Path Integration Homing',
                  fontsize=14)
    plt.tight_layout()

    # ── Figure 2: The cancellation breakdown ──
    print('  [2/4] Cancellation breakdown: constant vs spatial bias...')
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: constant bias (flat — from Direction B)
    ax = axes2[0]
    bias_range_deg = np.array([0, 2, 5, 10, 15, 20, 30, 45])
    bias_range = np.radians(bias_range_deg)
    for sig, color in [(0.3, '#2196F3'), (0.5, '#4CAF50'), (1.0, '#FF9800')]:
        errs = []
        for b in bias_range:
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                bias=b, use_pi=True, seed=42, mode='explore')
            errs.append(mean_err)
        ax.plot(bias_range_deg, errs, color=color, marker='o', ms=5, lw=2,
                label=f'σ_θ={sig}')
    ax.set_xlabel('Constant compass bias (°)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title('Constant bias: perfect cancellation', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.text(0.5, 0.95, 'Same-frame integration/readout',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            style='italic', color='#666')

    # Right: equivalent "bias" from spatial anomalies (grows!)
    # Map anomaly strength to approximate peak deviation in degrees
    # for comparison with the constant bias axis
    ax = axes2[1]
    for sig, color in [(0.3, '#2196F3'), (0.5, '#4CAF50'), (1.0, '#FF9800')]:
        errs = []
        for s in strengths:
            if s == 0:
                land = Landscape(extent=extent)
            else:
                land = Landscape(extent=extent)
                rng_land = np.random.default_rng(123)
                land.anomalies = Landscape.random_dipoles(
                    n_dip, extent, s, depth, rng=rng_land)
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=sig,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore',
                landscape=land)
            errs.append(mean_err)
            print(f'    spatial  σ={sig}  s={s}  err={mean_err:.1f} BL')
        ax.plot(strengths, errs, color=color, marker='s', ms=5, lw=2,
                label=f'σ_θ={sig}')
    ax.set_xlabel('Dipole peak anomaly (μT)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title('Spatial anomaly: cancellation broken', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    fig2.suptitle('Constant Bias Cancels; Spatial Bias Does Not',
                  fontsize=14)
    plt.tight_layout()

    # ── Figure 3: Phase diagram — anomaly strength × T_explore ──
    print('  [3/4] Phase diagram: anomaly × T_explore...')
    T_explore_grid = np.array([100, 200, 500, 1000, 2000])
    strength_grid = np.array([0, 1, 2, 3, 5, 8, 12])
    sig_phase = 0.5

    err_phase = np.zeros((len(strength_grid), len(T_explore_grid)))

    for i, s in enumerate(strength_grid):
        for j, T_ex in enumerate(T_explore_grid):
            T_h = max(T_ex, 200)
            if s == 0:
                land = Landscape(extent=extent)
            else:
                land = Landscape(extent=extent)
                rng_land = np.random.default_rng(123)
                land.anomalies = Landscape.random_dipoles(
                    n_dip, extent, s, depth, rng=rng_land)
            _, err_phase[i, j] = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_ex, T_home=T_h, dt=dt,
                kappa=2.0, sigma_theta=sig_phase,
                contrast=0.15, n_cry=50, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore',
                landscape=land)
            print(f'    s={s}  T_ex={T_ex}  '
                  f'err={err_phase[i,j]:.1f} BL')

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes3[0].pcolormesh(T_explore_grid, strength_grid, err_phase,
                               cmap='YlOrRd', shading='auto')
    plt.colorbar(im0, ax=axes3[0], label='Homing error (BL)')
    axes3[0].set_xlabel('Exploration time (s)', fontsize=12)
    axes3[0].set_ylabel('Dipole strength (μT)', fontsize=12)
    axes3[0].set_xscale('log')
    axes3[0].set_title(f'PI Homing Error ({n_dip} dipoles, σ_θ={sig_phase})',
                       fontsize=13)

    # Normalised by anomaly=0 baseline
    baseline = err_phase[0, :]  # strength=0 row
    ratio = err_phase / np.maximum(baseline[None, :], 1e-3)
    im1 = axes3[1].pcolormesh(T_explore_grid, strength_grid, ratio,
                               cmap='YlOrRd', shading='auto',
                               vmin=1.0, vmax=max(ratio.max(), 3.0))
    plt.colorbar(im1, ax=axes3[1], label='Ratio to clean field')
    axes3[1].set_xlabel('Exploration time (s)', fontsize=12)
    axes3[1].set_ylabel('Dipole strength (μT)', fontsize=12)
    axes3[1].set_xscale('log')
    axes3[1].set_title('Relative Degradation from Anomalies', fontsize=13)
    try:
        axes3[1].contour(T_explore_grid, strength_grid, ratio,
                         levels=[1.5, 2.0, 3.0],
                         colors=['orange', 'red', 'darkred'],
                         linewidths=2)
    except Exception:
        pass

    fig3.suptitle('Phase Diagram: When Do Anomalies Break PI Homing?',
                  fontsize=14)
    plt.tight_layout()

    # ── Figure 4: Compass model comparison with anomalies ──
    print('  [4/4] Compass model comparison with anomalies...')
    compass_configs = [
        ('FAD-O₂ (C=0.15)', 0.15, None),
        ('Relaxed FAD-O₂ (C=0.10)', 0.10, None),
        ('FAD-TrpH (C=0.01)', 0.01, None),
    ]
    compass_colors = ['#2196F3', '#4CAF50', '#FF9800']

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))

    # Left: anomaly strength sweep per model
    ax = axes4[0]
    for (label, C, my), color in zip(compass_configs, compass_colors):
        errs = []
        for s in strengths:
            if s == 0:
                land = Landscape(extent=extent)
            else:
                land = Landscape(extent=extent)
                rng_land = np.random.default_rng(123)
                land.anomalies = Landscape.random_dipoles(
                    n_dip, extent, s, depth, rng=rng_land)
            _, mean_err = pi_homing_ensemble(
                n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
                kappa=2.0, sigma_theta=0.5,
                contrast=C, n_cry=50, sigma_sensor=0.02,
                use_pi=True, seed=42, mode='explore',
                landscape=land, mean_yield=my)
            errs.append(mean_err)
            print(f'    {label}  s={s}  err={mean_err:.1f} BL')
        ax.plot(strengths, errs, color=color, marker='o', ms=5, lw=2,
                label=label)
    ax.set_xlabel('Dipole peak anomaly (μT)', fontsize=12)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title(f'PI with anomalies ({n_dip} dipoles)', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # Right: clean vs anomaly per model (bar-style comparison)
    ax = axes4[1]
    strength_comparison = 5.0  # μT — moderate anomaly
    x_pos = np.arange(len(compass_configs))
    width = 0.35
    clean_errs = []
    anom_errs = []
    land_anom = Landscape(extent=extent)
    rng_land = np.random.default_rng(123)
    land_anom.anomalies = Landscape.random_dipoles(
        n_dip, extent, strength_comparison, depth, rng=rng_land)
    for (label, C, my) in compass_configs:
        _, err_clean = pi_homing_ensemble(
            n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
            kappa=2.0, sigma_theta=0.5,
            contrast=C, n_cry=50, sigma_sensor=0.02,
            use_pi=True, seed=42, mode='explore',
            mean_yield=my)
        clean_errs.append(err_clean)
        _, err_anom = pi_homing_ensemble(
            n_bugs=n_bugs, T_out=T_explore, T_home=T_home, dt=dt,
            kappa=2.0, sigma_theta=0.5,
            contrast=C, n_cry=50, sigma_sensor=0.02,
            use_pi=True, seed=42, mode='explore',
            landscape=land_anom, mean_yield=my)
        anom_errs.append(err_anom)
        print(f'    {label}  clean={err_clean:.1f}  '
              f'anom={err_anom:.1f} BL')
    bars1 = ax.bar(x_pos - width/2, clean_errs, width, label='Clean field',
                   color='#E3F2FD', edgecolor='#2196F3', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, anom_errs, width,
                   label=f'{n_dip} dipoles, {strength_comparison:.0f} μT',
                   color='#FFECB3', edgecolor='#FF9800', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['FAD-O₂\n(C=0.15)', 'Relaxed\nFAD-O₂',
                        'FAD-TrpH\n(C=0.01)'], fontsize=10)
    ax.set_ylabel('Mean homing error (BL)', fontsize=12)
    ax.set_title('Clean vs anomaly field', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    # Add percentage increase labels
    for i, (c, a) in enumerate(zip(clean_errs, anom_errs)):
        if c > 0.1:
            pct = (a - c) / c * 100
            ax.text(i + width/2, a + 0.3, f'+{pct:.0f}%',
                    ha='center', fontsize=9, color='#E65100')

    fig4.suptitle('Compass Quality × Anomaly Interaction',
                  fontsize=14)
    plt.tight_layout()

    if save_prefix:
        fig1.savefig(f'{save_prefix}axb_strength.png', dpi=150)
        fig2.savefig(f'{save_prefix}axb_cancellation.png', dpi=150)
        fig3.savefig(f'{save_prefix}axb_phase.png', dpi=150)
        fig4.savefig(f'{save_prefix}axb_compass.png', dpi=150)
        print(f'Saved {save_prefix}axb_*.png')

    return fig1, fig2, fig3, fig4


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Deeper analysis of magnetic bug')
    parser.add_argument('--peclet', action='store_true')
    parser.add_argument('--differentiate', action='store_true')
    parser.add_argument('--harmonics', action='store_true')
    parser.add_argument('--critical-noise', action='store_true')
    parser.add_argument('--ncry', action='store_true')
    parser.add_argument('--validate-fast', action='store_true')
    parser.add_argument('--relax-nav', action='store_true')
    parser.add_argument('--uneq-rates', action='store_true')
    parser.add_argument('--orient', action='store_true')
    parser.add_argument('--anomaly', action='store_true')
    parser.add_argument('--pi', action='store_true')
    parser.add_argument('--axb', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--save', type=str, default='fig_',
                        help='Save prefix (default: fig_)')
    args = parser.parse_args()

    run_all = args.all or not any([args.peclet, args.differentiate,
                                    args.harmonics, args.critical_noise,
                                    args.ncry, args.validate_fast,
                                    args.relax_nav, args.uneq_rates,
                                    args.orient, args.anomaly, args.pi,
                                    args.axb])

    if args.peclet or run_all:
        print('=== Peclet number study ===')
        peclet_study(save_prefix=args.save)

    if args.harmonics or run_all:
        print('\n=== Harmonic decomposition ===')
        harmonic_decomposition(save_prefix=args.save)

    if args.differentiate or run_all:
        print('\n=== Model differentiation at boundary ===')
        model_differentiation(save_prefix=args.save)

    if args.critical_noise or run_all:
        print('\n=== Critical noise per model ===')
        critical_noise(save_prefix=args.save)

    if args.ncry or run_all:
        print('\n=== N_cry sweep ===')
        ncry_sweep(save_prefix=args.save)

    if args.validate_fast or run_all:
        print('\n=== Validate fast vs full simulation ===')
        validate_fast_vs_full(save_prefix=args.save)

    if args.relax_nav or run_all:
        print('\n=== Relaxation + navigation ===')
        relaxation_navigation(save_prefix=args.save)

    if args.uneq_rates or run_all:
        print('\n=== Unequal recombination rates ===')
        unequal_rates(save_prefix=args.save)

    if args.orient or run_all:
        print('\n=== Orientational disorder ===')
        orientational_disorder(save_prefix=args.save)

    if args.anomaly or run_all:
        print('\n=== Magnetic anomaly navigation ===')
        anomaly_navigation(save_prefix=args.save)

    if args.pi or run_all:
        print('\n=== Path integration (Direction B) ===')
        path_integration_analysis(save_prefix=args.save)

    if args.axb or run_all:
        print('\n=== Anomaly × Path integration (A×B) ===')
        anomaly_pi_analysis(save_prefix=args.save)

    print('\nDone.')


if __name__ == '__main__':
    main()
