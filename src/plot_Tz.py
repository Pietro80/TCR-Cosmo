import os
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tcr_core import gamma_of_z, dgamma_dz, A_default, p_default, Ok_eff_star, zt_mix_star, beta_mix_star
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

# Default parameters (you can modify them to test other cases)
Ok_eff = Ok_eff_star
zt_mix = zt_mix_star
beta_mix = beta_mix_star
A = A_default
p = p_default

# Definition of the relational expansion factor T(z)
def T_of_z(z):
    g = gamma_of_z(z, Ok_eff, zt_mix, beta_mix, A, p)
    dg = dgamma_dz(z, Ok_eff, zt_mix, beta_mix, A, p)
    return g / (1.0 + (1.0 + z) * (dg / g))

# Grid in z
z_vals = np.linspace(0, 3.0, 400)
T_vals = np.array([T_of_z(z) for z in z_vals])

# Valore a z=0
T0 = T_of_z(0.0)

# Plot
plt.figure(figsize=(8,5))
plt.plot(z_vals, T_vals, label="T(z)")
plt.axhline(1.0, linestyle="--", color="gray", label="T(z)=1 (limit without dressing)")

# marker a z=0
plt.scatter([0], [T0], color="red", zorder=5, label=f"T(0) = {T0:.3f}")

# sposto il testo un po’ a destra e leggermente più in alto
plt.text(0.15, T0-0.01, f"{T0:.3f}", color="red", ha="left", va="bottom")

plt.xlabel("Redshift z")
plt.ylabel("T(z) = γ / [1 + (1+z)γ'/γ]")
plt.title("Relational expansion factor T(z)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "T_of_z.png"), dpi=300)

plt.show()
