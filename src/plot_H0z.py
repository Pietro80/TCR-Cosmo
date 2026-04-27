import os
import numpy as np
import matplotlib.pyplot as plt
from tcr_core import gamma_of_z, dgamma_dz, A_default, p_default, Ok_eff_star, zt_mix_star, beta_mix_star
import tcr_core as tcr

# ---------- Parametri ----------
OM_LATE = 0.295
H0_TCR  = 66.8

tcr.set_tcr_late_params(H0=H0_TCR, Om=OM_LATE)

BASE = os.path.dirname(os.path.abspath(__file__))

Ok_eff = Ok_eff_star
zt_mix = zt_mix_star
beta_mix = beta_mix_star
A = A_default
p = p_default

# ---------- T(z) ----------
def T_of_z(z):
    g = gamma_of_z(z, Ok_eff, zt_mix, beta_mix, A, p)
    dg = dgamma_dz(z, Ok_eff, zt_mix, beta_mix, A, p)
    return g / (1.0 + (1.0 + z) * (dg / g))

# ---------- H0 * T(z) ----------
def H0T_of_z(z):
    return H0_TCR * T_of_z(z)

# Grid
z_vals = np.linspace(0, 3.0, 400)
H0T_vals = np.array([H0T_of_z(z) for z in z_vals])

# Valore a z=0
H0T0 = H0T_of_z(0.0)

# ---------- Plot ----------
plt.figure(figsize=(8,5))

plt.plot(z_vals, H0T_vals, label="H0 · T(z)")

# riferimento H0 costante
plt.axhline(H0_TCR, linestyle="--", color="gray", label=f"H0 = {H0_TCR}")

# punto z=0
plt.scatter([0], [H0T0], color="red", zorder=5, label=f"H0·T(0) = {H0T0:.2f}")
plt.text(0.15, H0T0-1, f"{H0T0:.2f}", color="red", ha="left", va="bottom")

plt.xlabel("Redshift z")
plt.ylabel("H0 · T(z) [km/s/Mpc]")
plt.title("H0-scaled relational expansion")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(BASE, "H0T_of_z.png"), dpi=300)
plt.show()