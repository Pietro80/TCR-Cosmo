#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_Hz_TCR_LCDM_z0_z3.py
---------------------------
Genera due grafici da z=0 a z=3:
1) H(z) totale sovrapposto: TCR-Cosmo vs ΛCDM
2) scarto relativo percentuale: 100 * (H_TCR/H_LCDM - 1)

Output:
- Hz_TCR_LCDM_overlay_z0_z3.png
- Hz_TCR_LCDM_relative_difference_z0_z3.png

Uso base:
    python3 plot_Hz_TCR_LCDM_z0_z3.py

Uso con parametri modificati:
    python3 plot_Hz_TCR_LCDM_z0_z3.py --H0-tcr 66.8 --Om-late 0.295 --n 800 --show

Nota:
- Se tcr_core.py è disponibile e importabile, usa direttamente le sue funzioni.
- Se tcr_core richiede dipendenze non installate, usa il fallback late-time equivalente
  alle formule presenti in tcr_core: H_LCDM, gamma(z), H_bare, H_TCR_obs.
"""

import os
import math
import argparse
import importlib.util
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Fallback late-time: formule equivalenti a tcr_core.py
# =========================================================
H0_LCDM = 67.4
Om_LCDM = 0.315
Ok_LCDM = 0.0

H0_b = 62.0
Om_b = 0.32
wbh0 = 0.01

A_default = 0.29
p_default = 0.94
q = 3.0
r_exp = 3.0
Bcoef = 1.0
Ccoef = 1.0
fdeep0 = 0.15
fshal0 = 0.55
wgal0 = 0.045
xi = 2.0
zt_void = 0.7
beta_void = 3.0
zt_gal = 0.7
beta_gal = 4.0
Ok_eff_star = 0.00
zt_mix_star = 0.75
beta_mix_star = 3.2


def set_tcr_late_params_fallback(H0=None, Om=None):
    global H0_b, Om_b
    if H0 is not None:
        H0_b = float(H0)
    if Om is not None:
        Om_b = float(Om)


def H_LCDM_fallback(z):
    Ol = 1.0 - Om_LCDM - Ok_LCDM
    return H0_LCDM * math.sqrt(Om_LCDM * (1.0 + z)**3 + Ok_LCDM * (1.0 + z)**2 + Ol)


def logistic_dec(z, value0, zt, beta):
    return value0 / (1.0 + math.exp(beta * (z - zt)))


def f_v_deep_z(z):
    return logistic_dec(z, fdeep0, zt_void, beta_void)


def f_v_shallow_z(z):
    return logistic_dec(z, fshal0, zt_void, beta_void)


def w_gal_z(z):
    return logistic_dec(z, wgal0, zt_gal, beta_gal)


def w_bh_const(z):
    return wbh0


def w_d(z, zt_mix, beta_mix, w0=0.85, w1=0.10):
    return w1 + (w0 - w1) / (1.0 + math.exp(-beta_mix * (zt_mix - z)))


def f_v_eff(z, zt_mix, beta_mix):
    fd, fs = f_v_deep_z(z), f_v_shallow_z(z)
    wd = w_d(z, zt_mix, beta_mix)
    inner = wd * (fd**xi) + (1.0 - wd) * (fs**xi)
    return inner**(1.0 / xi)


def H_bare(z, Ok_eff):
    Ol_eff = 1.0 - Om_b - Ok_eff
    return H0_b * math.sqrt(Om_b * (1.0 + z)**3 + Ok_eff * (1.0 + z)**2 + Ol_eff)


def gamma_of_z_fallback(z, Ok_eff, zt_mix, beta_mix, A, p):
    fv = f_v_eff(z, zt_mix, beta_mix)
    return 1.0 + A * (fv**p) * (1.0 - Bcoef * (w_bh_const(z)**q) - Ccoef * (w_gal_z(z)**r_exp))


def dgamma_dz_fallback(z, Ok_eff, zt_mix, beta_mix, A, p, eps_rel=1e-4):
    h = max(1e-4, eps_rel * max(1.0, z))
    return (
        gamma_of_z_fallback(z + h, Ok_eff, zt_mix, beta_mix, A, p)
        - gamma_of_z_fallback(z - h, Ok_eff, zt_mix, beta_mix, A, p)
    ) / (2.0 * h)


def H_TCR_obs_fallback(z, Ok_eff, zt_mix, beta_mix, A, p, den_floor=1e-4):
    gam = gamma_of_z_fallback(z, Ok_eff, zt_mix, beta_mix, A, p)
    dgam = dgamma_dz_fallback(z, Ok_eff, zt_mix, beta_mix, A, p)
    den = 1.0 + (1.0 + z) * dgam
    if den <= den_floor:
        den = den_floor
    return gam * H_bare(z, Ok_eff) / den


class FallbackTCR:
    H0_LCDM = H0_LCDM
    A_default = A_default
    p_default = p_default
    Ok_eff_star = Ok_eff_star
    zt_mix_star = zt_mix_star
    beta_mix_star = beta_mix_star
    set_tcr_late_params = staticmethod(set_tcr_late_params_fallback)
    H_LCDM = staticmethod(H_LCDM_fallback)
    H_TCR_obs = staticmethod(H_TCR_obs_fallback)


def load_tcr_core_or_fallback():
    base = os.path.dirname(os.path.abspath(__file__))
    for filename in ("tcr_core.py", "tcr_core(8).py"):
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("tcr_core", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, f"usato {filename}"
        except Exception as exc:
            print(f"[WARN] Non riesco a importare {filename}: {exc}")
            print("[WARN] Uso il fallback late-time equivalente per H(z).")
            return FallbackTCR, "fallback late-time"
    print("[WARN] tcr_core.py non trovato. Uso il fallback late-time interno.")
    return FallbackTCR, "fallback late-time"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-min", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--H0-tcr", type=float, default=66.8)
    ap.add_argument("--Om-late", type=float, default=0.295)
    ap.add_argument("--A", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--okeff", type=float, default=None)
    ap.add_argument("--zt", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--show", action="store_true")
    return ap


def main():
    args = build_parser().parse_args()
    tcr, source = load_tcr_core_or_fallback()

    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or base
    os.makedirs(out_dir, exist_ok=True)

    tcr.set_tcr_late_params(H0=args.H0_tcr, Om=args.Om_late)

    Ok_eff = tcr.Ok_eff_star if args.okeff is None else args.okeff
    zt_mix = tcr.zt_mix_star if args.zt is None else args.zt
    beta_mix = tcr.beta_mix_star if args.beta is None else args.beta
    A = tcr.A_default if args.A is None else args.A
    p = tcr.p_default if args.p is None else args.p

    z = np.linspace(args.z_min, args.z_max, args.n)
    H_lcdm = np.array([tcr.H_LCDM(float(zz)) for zz in z])
    H_tcr = np.array([tcr.H_TCR_obs(float(zz), Ok_eff, zt_mix, beta_mix, A, p) for zz in z])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(z, H_lcdm, label=rf"$\Lambda$CDM  ($H_0={tcr.H0_LCDM:.1f}$)", linewidth=2)
    ax.plot(z, H_tcr, label=rf"TCR-Cosmo  ($H_0={args.H0_tcr:.1f}$, $\Omega_m={args.Om_late:.3f}$)", linewidth=2)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title(r"$H(z)$ totale sovrapposto: TCR-Cosmo vs $\Lambda$CDM, $0 \leq z \leq 3$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out1 = os.path.join(out_dir, "Hz_TCR_LCDM_overlay_z0_z3.png")
    fig.savefig(out1, dpi=300)

    rel_diff = 100.0 * (H_tcr / H_lcdm - 1.0)
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    ax2.plot(z, rel_diff, linewidth=2, label=r"$100\,(H_{TCR}/H_{\Lambda CDM}-1)$")
    ax2.axhline(0.0, linestyle="--", linewidth=1)
    ax2.set_xlabel("Redshift z")
    ax2.set_ylabel("Differenza relativa [%]")
    ax2.set_title(r"Scarto relativo di $H(z)$: TCR-Cosmo rispetto a $\Lambda$CDM")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    out2 = os.path.join(out_dir, "Hz_TCR_LCDM_relative_difference_z0_z3.png")
    fig2.savefig(out2, dpi=300)

    print(f"[OK] Fonte modello: {source}")
    print("[OK] Grafici salvati:")
    print(" -", out1)
    print(" -", out2)
    print("\nValori di controllo:")
    for zz in [0.0, 0.5, 1.0, 2.0, 3.0]:
        hl = float(tcr.H_LCDM(zz))
        ht = float(tcr.H_TCR_obs(zz, Ok_eff, zt_mix, beta_mix, A, p))
        print(f"z={zz:>3.1f} | LCDM={hl:8.3f} | TCR={ht:8.3f} | diff={100*(ht/hl-1):+7.3f}%")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
