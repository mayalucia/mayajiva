"""
Quantum spin dynamics of radical pairs.

Computes the singlet yield Φ_S(θ) from first principles: the spin
Hamiltonian (Zeeman + hyperfine + exchange), Haberkorn recombination,
and the initial singlet-born density matrix.

The toy model (2 electrons + 1 nucleus, dim=8) captures the essential
physics of the radical-pair compass.  The intermediate model (2 electrons
+ 2 nuclei, dim=16) captures ~90% of the FAD anisotropy.

Units: HFC tensors in Tesla, B0 in Tesla, rates in s⁻¹.
Internally, angular frequencies are gamma_e × (field in Tesla).

References:
  - Haberkorn (1976) Mol. Phys. 32:1491
  - Hore & Mouritsen (2016) Annu. Rev. Biophys. 45:299
  - Hiscock et al. (2016) PNAS 113:4634
"""

import numpy as np

# ── Physical constants ──────────────────────────────────────────────

GAMMA_E = 1.761e11   # electron gyromagnetic ratio (rad/s/T)
B0_EARTH = 50e-6     # typical geomagnetic field (T)

# ── Pauli matrices and spin-½ operators ─────────────────────────────

_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)

_S_HALF = {'x': _SIGMA_X / 2, 'y': _SIGMA_Y / 2, 'z': _SIGMA_Z / 2}
_COMP = ('x', 'y', 'z')


# ── Operators in product Hilbert space ──────────────────────────────

def embed_operator(op, site, n_sites):
    """Embed a single-site operator into the full product space.

    Parameters
    ----------
    op : ndarray (2, 2)
        Single spin-½ operator.
    site : int
        Subsystem index (0 = electron A, 1 = electron B, 2+ = nuclei).
    n_sites : int
        Total number of spin-½ subsystems.

    Returns
    -------
    ndarray (d, d) with d = 2**n_sites
    """
    factors = [_I2] * n_sites
    factors[site] = op
    result = factors[0]
    for f in factors[1:]:
        result = np.kron(result, f)
    return result


def spin_operators(site, n_sites):
    """Return {Sx, Sy, Sz} for subsystem `site` in the full space."""
    return {c: embed_operator(_S_HALF[c], site, n_sites) for c in _COMP}


def singlet_projector(n_sites):
    """Singlet projector P_S = |S><S| ⊗ I_nuclear.

    Uses the identity P_S = 1/4 I - S_A · S_B (exact for two spin-½).
    """
    SA = spin_operators(0, n_sites)
    SB = spin_operators(1, n_sites)
    d = 2 ** n_sites
    SdotS = sum(SA[c] @ SB[c] for c in _COMP)
    return 0.25 * np.eye(d, dtype=complex) - SdotS


def initial_state(n_sites):
    """Initial density matrix: singlet electrons ⊗ maximally mixed nuclei.

    ρ₀ = |S><S| ⊗ I_nuc / d_nuc,  with Tr[ρ₀] = 1.
    """
    d_nuc = 2 ** (n_sites - 2)
    return singlet_projector(n_sites) / d_nuc


# ── Hamiltonian construction ────────────────────────────────────────

def hfc_tensor_axial(a_iso, a_aniso, site, electron=0):
    """Build an axial HFC tensor specification.

    Parameters
    ----------
    a_iso : float
        Isotropic hyperfine coupling (Tesla).
    a_aniso : float
        Anisotropy Δ = a_∥ − a_⊥ (Tesla).
    site : int
        Nuclear subsystem index (≥ 2).
    electron : int
        Which electron (0 or 1) this nucleus is coupled to.

    Returns
    -------
    dict with keys 'site', 'electron', 'A' (3×3 ndarray in Tesla).
    """
    a_perp = a_iso - a_aniso / 3.0
    a_par = a_iso + 2.0 * a_aniso / 3.0
    A = np.diag([a_perp, a_perp, a_par])
    return {'site': site, 'electron': electron, 'A': A}


def build_hamiltonian(theta, B0, hfc_tensors, J=0.0, n_sites=None):
    """Construct the radical-pair spin Hamiltonian.

    Parameters
    ----------
    theta : float
        Angle between B₀ and the molecular z-axis (rad).
    B0 : float
        Magnetic field magnitude (Tesla).
    hfc_tensors : list of dict
        Each: {'site': int, 'electron': int, 'A': (3,3) ndarray in Tesla}.
    J : float
        Exchange coupling (rad/s).
    n_sites : int or None
        Total spin-½ subsystems.  Inferred from hfc_tensors if None.

    Returns
    -------
    H : ndarray (d, d), complex, Hermitian.  Units: rad/s.
    """
    if n_sites is None:
        n_sites = 2 + len(hfc_tensors)
    d = 2 ** n_sites

    # Field direction in molecular frame (phi = 0 by axial symmetry)
    B_hat = np.array([np.sin(theta), 0.0, np.cos(theta)])

    SA = spin_operators(0, n_sites)
    SB = spin_operators(1, n_sites)

    # Zeeman: -γₑ B · (Sₐ + S_b)
    H = np.zeros((d, d), dtype=complex)
    for k, c in enumerate(_COMP):
        H += -GAMMA_E * B0 * B_hat[k] * (SA[c] + SB[c])

    # Hyperfine: γₑ Sₑ · A · Iₖ  (A in Tesla → multiply by γₑ for rad/s)
    for hfc in hfc_tensors:
        Se = spin_operators(hfc['electron'], n_sites)
        Ik = spin_operators(hfc['site'], n_sites)
        A = hfc['A']
        for a in range(3):
            for b in range(3):
                if abs(A[a, b]) > 1e-30:
                    H += GAMMA_E * A[a, b] * (Se[_COMP[a]] @ Ik[_COMP[b]])

    # Exchange: J(1/4 + Sₐ · S_b)
    if abs(J) > 1e-30:
        SdotS = sum(SA[c] @ SB[c] for c in _COMP)
        H += J * (0.25 * np.eye(d, dtype=complex) + SdotS)

    return H


# ── Singlet yield computation ───────────────────────────────────────

def singlet_yield_eq(H, P_S, rho0, k):
    """Singlet yield for equal recombination rates k_S = k_T = k.

    Algebraic formula via eigendecomposition (no time integration):

        Φ_S = k Σ_{n,m} ⟨m|P_S|n⟩⟨n|ρ₀|m⟩ / (k + i(Eₙ − Eₘ))

    Parameters
    ----------
    H : (d, d) Hamiltonian.
    P_S : (d, d) singlet projector.
    rho0 : (d, d) initial density matrix.
    k : float, recombination rate (s⁻¹).

    Returns
    -------
    float : Φ_S ∈ [0, 1].
    """
    E, V = np.linalg.eigh(H)
    P_eig = V.conj().T @ P_S @ V      # P_eig[m,n] = <m|P_S|n>
    rho_eig = V.conj().T @ rho0 @ V   # rho_eig[n,m] = <n|rho0|m>

    dE = E[:, None] - E[None, :]      # dE[n,m] = E_n - E_m
    L = 1.0 / (k + 1j * dE)            # Lorentzian kernel

    return (k * np.sum(P_eig.T * rho_eig * L)).real


def singlet_yield_uneq(H, P_S, rho0, k_S, k_T):
    """Singlet yield for unequal rates via Liouvillian inversion.

    Vectorises the density matrix (column-stacking, order='F') and
    solves  L |σ⟩ = −|ρ₀⟩  where L is the Haberkorn Liouvillian.

    Parameters
    ----------
    H, P_S, rho0 : (d, d) arrays.
    k_S, k_T : float, singlet and triplet recombination rates (s⁻¹).

    Returns
    -------
    float : Φ_S ∈ [0, 1].
    """
    d = H.shape[0]
    Id = np.eye(d, dtype=complex)
    P_T = Id - P_S

    # Liouvillian superoperator (d² × d²)
    # Column-stacking: vec(A ρ B) = (B^T ⊗ A) vec(ρ)
    L = (-1j * (np.kron(H, Id) - np.kron(Id, H.T))
         - 0.5 * k_S * (np.kron(P_S, Id) + np.kron(Id, P_S.T))
         - 0.5 * k_T * (np.kron(P_T, Id) + np.kron(Id, P_T.T)))

    rho_vec = rho0.flatten(order='F')
    try:
        sigma_vec = np.linalg.solve(L, -rho_vec)
    except np.linalg.LinAlgError:
        # Singular when one rate is zero — use least-squares
        sigma_vec, _, _, _ = np.linalg.lstsq(L, -rho_vec, rcond=None)
    sigma = sigma_vec.reshape((d, d), order='F')

    return (k_S * np.trace(P_S @ sigma)).real


# ── Spin relaxation ────────────────────────────────────────────────

def relaxation_superoperator(n_sites, k_relax_A=0.0, k_relax_B=0.0):
    """Isotropic random-field relaxation Liouvillian for both electrons.

    Each electron α experiences independent, isotropic random fields
    causing T1 = T2 = 1/k_relax_α.  The Lindblad dissipators are:

        D[√k S_α^q](ρ) = k (S^q ρ S^q − ρ/4)

    summed over q ∈ {x,y,z} and α ∈ {A,B}.

    In vectorised form (column-stacking):

        L_relax_α = k_α [Σ_q (S_α^{qT} ⊗ S_α^q) − (3/4) I_{d²}]

    Parameters
    ----------
    n_sites : int
        Total spin-½ subsystems.
    k_relax_A, k_relax_B : float
        Relaxation rates for electrons A (site 0) and B (site 1) in s⁻¹.
        T1 = T2 = 1/k_relax.  Set to 0 for no relaxation.

    Returns
    -------
    L_relax : ndarray (d², d²), complex.
        Relaxation superoperator to add to the Haberkorn Liouvillian.
    """
    d = 2 ** n_sites
    d2 = d * d
    L_relax = np.zeros((d2, d2), dtype=complex)

    for site, k_r in [(0, k_relax_A), (1, k_relax_B)]:
        if k_r == 0.0:
            continue
        S = spin_operators(site, n_sites)
        dissip = np.zeros((d2, d2), dtype=complex)
        for c in _COMP:
            Sq = S[c]
            # vec(S ρ S) = (S^T ⊗ S) vec(ρ)  [column-stacking]
            dissip += np.kron(Sq.T, Sq)
        dissip -= 0.75 * np.eye(d2, dtype=complex)
        L_relax += k_r * dissip

    return L_relax


def singlet_yield_relaxed(H, P_S, rho0, k_S, k_T,
                          k_relax_A=0.0, k_relax_B=0.0, n_sites=None):
    """Singlet yield with Haberkorn recombination + spin relaxation.

    Uses the full Liouvillian inversion: L σ = −ρ₀, Φ_S = k_S Tr[P_S σ].
    Falls back to singlet_yield_eq or singlet_yield_uneq if no relaxation.

    Parameters
    ----------
    H, P_S, rho0 : (d, d) arrays.
    k_S, k_T : float
    k_relax_A, k_relax_B : float
        Spin relaxation rates (s⁻¹).
    n_sites : int or None

    Returns
    -------
    float : Φ_S ∈ [0, 1].
    """
    # No relaxation → use fast paths
    if k_relax_A == 0.0 and k_relax_B == 0.0:
        if np.isclose(k_S, k_T):
            return singlet_yield_eq(H, P_S, rho0, k_S)
        else:
            return singlet_yield_uneq(H, P_S, rho0, k_S, k_T)

    d = H.shape[0]
    if n_sites is None:
        n_sites = int(np.log2(d))
    Id = np.eye(d, dtype=complex)
    P_T = Id - P_S

    # Haberkorn Liouvillian
    L = (-1j * (np.kron(H, Id) - np.kron(Id, H.T))
         - 0.5 * k_S * (np.kron(P_S, Id) + np.kron(Id, P_S.T))
         - 0.5 * k_T * (np.kron(P_T, Id) + np.kron(Id, P_T.T)))

    # Add relaxation
    L += relaxation_superoperator(n_sites, k_relax_A, k_relax_B)

    rho_vec = rho0.flatten(order='F')
    try:
        sigma_vec = np.linalg.solve(L, -rho_vec)
    except np.linalg.LinAlgError:
        sigma_vec, _, _, _ = np.linalg.lstsq(L, -rho_vec, rcond=None)
    sigma = sigma_vec.reshape((d, d), order='F')

    return (k_S * np.trace(P_S @ sigma)).real


# ── Predefined radical pair models ──────────────────────────────────

# FAD hyperfine parameters (Tesla)
_FAD_N5_AISO = 523e-6
_FAD_N5_ANISO = 700e-6     # a_∥ − a_⊥, from DFT (order-of-magnitude)

_FAD_N10_AISO = 189e-6
_FAD_N10_ANISO = 250e-6

# TrpH β-proton HFC (Tesla), approximately isotropic
_TRPH_H1_AISO = 0.71e-3
_TRPH_H1_ANISO = 0.0

_TRPH_H2_AISO = 1.07e-3
_TRPH_H2_ANISO = 0.0


def toy_fad_o2():
    """[FAD·⁻ O₂·⁻] toy: 2 electrons + N5 nucleus.  dim = 8."""
    return {
        'name': '[FAD·⁻ O₂·⁻] toy (N5)',
        'n_sites': 3,
        'hfc_tensors': [
            hfc_tensor_axial(_FAD_N5_AISO, _FAD_N5_ANISO, site=2, electron=0),
        ],
        'J': 0.0,
    }


def toy_fad_trp():
    """[FAD·⁻ TrpH·⁺] toy: 2 electrons + N5 + H1.  dim = 16."""
    return {
        'name': '[FAD·⁻ TrpH·⁺] toy (N5 + H1)',
        'n_sites': 4,
        'hfc_tensors': [
            hfc_tensor_axial(_FAD_N5_AISO, _FAD_N5_ANISO, site=2, electron=0),
            hfc_tensor_axial(_TRPH_H1_AISO, _TRPH_H1_ANISO, site=3, electron=1),
        ],
        'J': 0.0,
    }


def intermediate_fad_o2():
    """[FAD·⁻ O₂·⁻] intermediate: 2 electrons + N5 + N10.  dim = 16."""
    return {
        'name': '[FAD·⁻ O₂·⁻] (N5 + N10)',
        'n_sites': 4,
        'hfc_tensors': [
            hfc_tensor_axial(_FAD_N5_AISO, _FAD_N5_ANISO, site=2, electron=0),
            hfc_tensor_axial(_FAD_N10_AISO, _FAD_N10_ANISO, site=3, electron=0),
        ],
        'J': 0.0,
    }


def intermediate_fad_trp():
    """[FAD·⁻ TrpH·⁺] intermediate: N5+N10 + H1+H2.  dim = 64."""
    return {
        'name': '[FAD·⁻ TrpH·⁺] (N5+N10 + H1+H2)',
        'n_sites': 6,
        'hfc_tensors': [
            hfc_tensor_axial(_FAD_N5_AISO, _FAD_N5_ANISO, site=2, electron=0),
            hfc_tensor_axial(_FAD_N10_AISO, _FAD_N10_ANISO, site=3, electron=0),
            hfc_tensor_axial(_TRPH_H1_AISO, _TRPH_H1_ANISO, site=4, electron=1),
            hfc_tensor_axial(_TRPH_H2_AISO, _TRPH_H2_ANISO, site=5, electron=1),
        ],
        'J': 0.0,
    }


# ── RadicalPairCompass: pre-computed lookup table ───────────────────

class RadicalPairCompass:
    """Quantum radical-pair compass with pre-computed Φ_S(θ) profile.

    At init, solves the spin Hamiltonian at n_theta angles to build a
    lookup table.  During simulation, reads are fast interpolations.

    Parameters
    ----------
    model : dict or None
        From one of the factory functions.  Default: toy_fad_o2().
    B0 : float
        Magnetic field magnitude (Tesla).
    k : float
        Recombination rate (s⁻¹).  Default 1e6 (1 μs lifetime).
    k_S, k_T : float or None
        Singlet/triplet rates.  If None, both set to k.
    k_relax_A, k_relax_B : float
        Spin relaxation rates for electrons A and B (s⁻¹).
        T1 = T2 = 1/k_relax.  Default 0 (no relaxation).
    n_theta : int
        Lookup table resolution.
    """

    def __init__(self, model=None, B0=B0_EARTH, k=1e6,
                 k_S=None, k_T=None,
                 k_relax_A=0.0, k_relax_B=0.0,
                 n_theta=360):
        if model is None:
            model = toy_fad_o2()

        self.model = model
        self.B0 = B0
        self.k_S = k_S if k_S is not None else k
        self.k_T = k_T if k_T is not None else k
        self.k_relax_A = k_relax_A
        self.k_relax_B = k_relax_B
        self.n_sites = model['n_sites']

        # Pre-compute singlet yield profile on [0, π]
        self._thetas = np.linspace(0, np.pi, n_theta, endpoint=True)
        self._yields = np.empty(n_theta)

        P_S = singlet_projector(self.n_sites)
        rho0 = initial_state(self.n_sites)
        has_relax = (k_relax_A > 0.0 or k_relax_B > 0.0)
        equal = np.isclose(self.k_S, self.k_T)

        for i, th in enumerate(self._thetas):
            H = build_hamiltonian(
                th, B0, model['hfc_tensors'],
                J=model.get('J', 0.0), n_sites=self.n_sites)
            if has_relax:
                self._yields[i] = singlet_yield_relaxed(
                    H, P_S, rho0, self.k_S, self.k_T,
                    k_relax_A, k_relax_B, self.n_sites)
            elif equal:
                self._yields[i] = singlet_yield_eq(H, P_S, rho0, self.k_S)
            else:
                self._yields[i] = singlet_yield_uneq(
                    H, P_S, rho0, self.k_S, self.k_T)

        # Derived quantities
        self.mean_yield = float(np.mean(self._yields))
        self.max_yield = float(np.max(self._yields))
        self.min_yield = float(np.min(self._yields))
        self.contrast = (self.max_yield - self.min_yield) / self.mean_yield

    def singlet_yield(self, theta):
        """Interpolated Φ_S(θ).  θ folded to [0, π] by symmetry.

        Parameters
        ----------
        theta : float or ndarray
            Angle between field and molecular z-axis (rad).

        Returns
        -------
        float or ndarray
        """
        t = np.abs(theta) % np.pi
        return np.interp(t, self._thetas, self._yields)

    def yield_curve(self):
        """Return (thetas, yields) for the full lookup table."""
        return self._thetas.copy(), self._yields.copy()
