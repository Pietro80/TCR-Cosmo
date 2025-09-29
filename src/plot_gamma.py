import os
import numpy as np
import matplotlib.pyplot as plt
from tcr_core import gamma_of_z, dgamma_dz

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
