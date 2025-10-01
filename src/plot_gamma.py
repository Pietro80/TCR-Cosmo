import os
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tcr_core import gamma_of_z, dgamma_dz
import tcr_core as tcr


# ---------- Parametri di lavoro (modificabili) ----------
# Late-time TCR (per le curve e la griglia)
OM_LATE = 0.295
H0_TCR  = 66.8
tcr.set_tcr_late_params(H0=H0_TCR, Om=OM_LATE)

# Griglia per heatmap/contours (Ωm, H0)
OM_GRID = np.linspace(0.28, 0.32, 9)
H0_GRID = np.arange(62, 69)

# ---------- Helper: costruisci args EARLY per rs_reference ----------
def make_args_early_tcr():
    # campi richiesti da tcr_core.early_params_from_args + r_s_reference + H_early_dispatch
    return SimpleNamespace(
        early_mode="tcr",
        omega_b_early=0.02237,
        omega_c_early=0.0,
        N_eff=3.046,
        h_early=0.62,
        rs_from="z_drag",     # per BAO
        z_star=1089.0,
        z_drag=1059.0,
        alpha_rs=1.0,
        # parametri extra usati in H_TCR_early
        okeff_early=0.0,
        A_early=0.0,
        p_early=1.0,
        zt_early=3000.0,
        beta_early=3.0,
        A_infl=6.0,
        zinfl_on=3e3,
        zinfl_off=3e4,
        beta_infl=3.0,
        z_bbn=1e8,
        dNeff_early=0.0,
        theta_prior="none",
        theta100_P=1.0411,
        sigma_theta100=0.0003
    )

def compute_rs_from_args(args_early):
    pars = tcr.early_params_from_args(args_early)
    return tcr.r_s_reference(args_early, pars), pars

args_early_TCR = make_args_early_tcr()
rs_TCR, _ = compute_rs_from_args(args_early_TCR)

# ---------- Paths (same folder as this script) ----------
BASE = os.path.dirname(os.path.abspath(__file__))

# --- Example parameters (you can modify them)
Ok_eff = 0.0      # effective curvature
zt_mix = 0.7      # transition redshift
beta_mix = 5.0    # steepness of the transition
A = 0.5           # dressing amplitude
p = 1.0           # exponent

# --- Grid in z
z_vals = np.linspace(0, 5, 200)

# --- Compute gamma and its derivative
gamma_vals = [gamma_of_z(z, Ok_eff, zt_mix, beta_mix, A, p) for z in z_vals]
dgamma_vals = [dgamma_dz(z, Ok_eff, zt_mix, beta_mix, A, p) for z in z_vals]

# --- Plot γ(z)
plt.figure(figsize=(8,5))
plt.plot(z_vals, gamma_vals, label=r'$\gamma(z)$', color='blue')
plt.axhline(1.0, color='gray', linestyle='--', label=r'$\gamma=1$ (limit $\Lambda$CDM)')
plt.xlabel("Redshift z")
plt.ylabel(r'$\gamma(z)$')
plt.title("Relational dressing function γ(z)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "gamma_z.png"), dpi=300)

# --- Plot γ′(z)
plt.figure(figsize=(8,5))
plt.plot(z_vals, dgamma_vals, label=r"$\gamma'(z)$", color='red')
plt.axhline(0.0, color='gray', linestyle='--')
plt.xlabel("Redshift z")
plt.ylabel(r"$d\gamma/dz$")
plt.title("Derivative of γ(z)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "dgamma_dz.png"), dpi=300)

plt.show()
