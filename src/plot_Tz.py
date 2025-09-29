import os
import numpy as np
import matplotlib.pyplot as plt
from tcr_core import gamma_of_z, dgamma_dz, A_default, p_default, Ok_eff_star, zt_mix_star, beta_mix_star

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

# Plot
plt.figure(figsize=(8,5))
plt.plot(z_vals, T_vals, label="T(z)")
plt.axhline(1.0, linestyle="--", color="gray", label="T(z)=1 (limit without dressing)")
plt.xlabel("Redshift z")
plt.ylabel("T(z) = γ / [1 + (1+z)γ'/γ]")
plt.title("Relational expansion factor T(z)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "T_of_z.png"), dpi=300)

plt.show()
