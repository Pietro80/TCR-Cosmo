
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results_standalone.py
--------------------------
- Pantheon+SH0ES.dat, Pantheon+SH0ES_STAT+SYS.cov
- data_MM20.dat, HzTable_MM_BC03.dat
- BAO_consensus_results_dM_Hz.txt, BAO_consensus_covtot_dM_Hz.txt

Output PNG salvati nella stessa cartella:
- plot_sn_hubble.png
- plot_sn_residuals.png
- plot_cc_Hz.png
- plot_bao_DM.png
- plot_bao_Hz.png
- plot_tcr_chi2_heatmap_light.png
- plot_tcr_chi2_contours_light.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace

import tcr_core as tcr

# ---------- Percorsi (stessa cartella dello script) ----------
BASE = os.path.dirname(os.path.abspath(__file__))
PATH_SN      = os.path.join(BASE, "Pantheon+SH0ES.dat")
PATH_SN_COV  = os.path.join(BASE, "Pantheon+SH0ES_STAT+SYS.cov")
PATH_CC      = os.path.join(BASE, "data_MM20.dat")
PATH_CC_TAB  = os.path.join(BASE, "HzTable_MM_BC03.dat")
PATH_BAO_RES = os.path.join(BASE, "BAO_consensus_results_dM_Hz.txt")
PATH_BAO_COV = os.path.join(BASE, "BAO_consensus_covtot_dM_Hz.txt")

# ---------- Parametri di lavoro (modificabili) ----------
# Late-time TCR (per le curve e la griglia)
OM_LATE = 0.295
H0_TCR  = 66.8

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

def make_args_early_lcdm(rs_from="z_drag"):
    return SimpleNamespace(
        early_mode="std",
        omega_b_early=0.02237,
        omega_c_early=0.12,
        N_eff=3.046,
        h_early=tcr.H0_LCDM/100.0,
        rs_from=rs_from,
        z_star=1089.0,
        z_drag=1059.0,
        alpha_rs=1.0,
        # i seguenti non usati in std ma lascio placeholder
        okeff_early=0.0, A_early=0.0, p_early=1.0, zt_early=3000.0, beta_early=3.0,
        A_infl=0.0, zinfl_on=3e3, zinfl_off=3e4, beta_infl=3.0,
        z_bbn=1e8, dNeff_early=0.0,
        theta_prior="none", theta100_P=1.0411, sigma_theta100=0.0003
    )

def compute_rs_from_args(args_early):
    pars = tcr.early_params_from_args(args_early)
    return tcr.r_s_reference(args_early, pars), pars

# ---------- Costruisci funzioni modello late-time ----------
def get_models():
    # LCDM
    H_LCDM_f = lambda z: tcr.H_LCDM(z)
    # TCR (usa parametri late impostati da set_tcr_late_params)
    H_TCR_f  = lambda z: tcr.H_TCR_obs(z, tcr.Ok_eff_star, tcr.zt_mix_star, tcr.beta_mix_star, tcr.A_default, tcr.p_default)
    return H_LCDM_f, H_TCR_f

# ---------- Figure: SN ----------
def figure_sn(H_LCDM_f, H_TCR_f):
    sn = pd.read_csv(PATH_SN, sep=r"\s+", engine="python", comment="#")
    z = sn["zHD"].values
    mu_obs = sn["MU_SH0ES"].values

    def mu_theory(zv, H_func, Ok, H0):
        return np.array([5*np.log10((1+zz)*tcr.D_M(zz, H_func, Ok, H0)*1e6/10.0) for zz in zv])

    mu_LCDM = mu_theory(z, H_LCDM_f, 0.0, tcr.H0_LCDM)
    mu_TCR  = mu_theory(z, H_TCR_f,  tcr.Ok_eff_star, H0_TCR)

    # Hubble diagram con colori distinti
    plt.figure()
    plt.plot(z, mu_LCDM, label="LCDM", color="tab:orange")
    plt.plot(z, mu_TCR,  label="TCR",  color="tab:blue")
    plt.scatter(z, mu_obs, s=10, label="Pantheon+SH0ES", color="black", alpha=0.7)
    plt.xlabel("z")
    plt.ylabel(r"$\mu$ (distance modulus)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE,"plot_sn_hubble.png"), dpi=160)
    plt.close()

    # Residui
    dmu_LCDM = mu_LCDM - mu_obs
    dmu_TCR  = mu_TCR  - mu_obs
    plt.figure()
    plt.scatter(z, dmu_LCDM, s=12, label="LCDM - data", marker="x", color="tab:orange")
    plt.scatter(z, dmu_TCR,  s=12, label="TCR - data",  marker="x", color="tab:blue")
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.xlabel("z")
    plt.ylabel(r"$\Delta \mu$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE,"plot_sn_residuals.png"), dpi=160)
    plt.close()

# ---------- Figure: CC ----------

def _read_cc_points_flexible():
    """
    Ritorna (z, H, sigma) leggendo:
    - prima prova con data_MM20.dat (formato atteso: z, H, sigma);
    - se i valori non sembrano H(z) realistici, ripiega su HzTable_MM_BC03.dat
      leggendo le prime 3 colonne (z, Hz, errHz).
    """
    try:
        z, H, sig = tcr.load_CC_data_MM20(PATH_CC)
        # Heuristica: se H mediano < 10 (sospetto) o rapporto sig/H > 1,
        # probabilmente non è il file giusto → fallback
        import numpy as _np
        medH = float(_np.nanmedian(H))
        medRatio = float(_np.nanmedian(_np.abs(sig) / _np.clip(_np.abs(H), 1e-6, None)))
        if medH < 10.0 or medRatio > 1.0:
            raise RuntimeError("PATH_CC non sembra contenere H(z). Fallback a HzTable_MM_BC03.dat")
        return z, H, sig
    except Exception:
        # Fallback: usa HzTable_MM_BC03.dat (z, Hz, errHz) separato da virgole
        import pandas as _pd
        cc = _pd.read_csv(PATH_CC_TAB, comment="#", engine="python",
                          sep=",", header=None, usecols=[0,1,2],
                          names=["z","H","sigma"])
        return cc["z"].values, cc["H"].values, cc["sigma"].values


def figure_cc(H_LCDM_f, H_TCR_f):
    z_cc, Hobs, sig_stat = _read_cc_points_flexible()
    zz = np.linspace(0.0, 2.1, 400)
    H_LCDM = np.array([H_LCDM_f(zi) for zi in zz])
    H_TCR  = np.array([H_TCR_f(zi)  for zi in zz])
    plt.figure()
    plt.errorbar(z_cc, Hobs, yerr=sig_stat, fmt="o", ms=4, label="CC data")
    plt.plot(zz, H_LCDM, label="LCDM"); plt.plot(zz, H_TCR,  label="TCR")
    plt.xlabel("z"); plt.ylabel("H(z) [km/s/Mpc]"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE,"plot_cc_Hz.png"), dpi=160); plt.close()

# ---------- Figure: BAO ----------
def figure_bao(rs_ref_LCDM, rs_ref_TCR, H_LCDM_f, H_TCR_f):
    rows_bao, Cbao = tcr.read_bao_consensus(PATH_BAO_RES, PATH_BAO_COV)
    Cbao = np.array(Cbao, dtype=float)
    # separa dati e incertezze (use diag for visualization)
    z_dm, y_dm, z_hz, y_hz = [], [], [], []
    err_dm, err_hz = [], []
    diag = np.sqrt(np.diag(Cbao))
    for idx, (z, label, val) in enumerate(rows_bao):
        if label.startswith("dM"):
            z_dm.append(z); y_dm.append(val); err_dm.append(diag[idx])
        else:
            z_hz.append(z); y_hz.append(val); err_hz.append(diag[idx])

    def pred_dm_hz(rows, H_func, Ok, H0, rs_ref):
        vec = tcr.bao_predict_vector(rows, H_func, Ok, H0, rs_ref)
        v_dm, v_hz = [], []
        for (z,label,_), v in zip(rows, vec):
            (v_dm if label.startswith("dM") else v_hz).append(v)
        return np.array(v_dm), np.array(v_hz)

    ydm_LCDM, yhz_LCDM = pred_dm_hz(rows_bao, H_LCDM_f, 0.0, tcr.H0_LCDM, rs_ref_LCDM)
    ydm_TCR,  yhz_TCR  = pred_dm_hz(rows_bao, H_TCR_f,  tcr.Ok_eff_star, H0_TCR, rs_ref_TCR)

    # dM con barre d'errore
    plt.figure()
    plt.plot(z_dm, ydm_LCDM, "s-", label="LCDM")
    plt.plot(z_dm, ydm_TCR,  "^-", label="TCR")
    plt.errorbar(z_dm, y_dm, yerr=err_dm, fmt="o", label="Data")
    plt.xlabel("z"); plt.ylabel(r"$D_M(z)\,(r_d^{fid}/r_s)$")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_bao_DM.png"), dpi=160); plt.close()

    # Hz con barre d'errore
    plt.figure()
    plt.plot(z_hz, yhz_LCDM, "s-", label="LCDM")
    plt.plot(z_hz, yhz_TCR,  "^-", label="TCR")
    plt.errorbar(z_hz, y_hz, yerr=err_hz, fmt="o", label="Data")
    plt.xlabel("z"); plt.ylabel(r"$H(z)\,(r_s/r_d^{fid})$")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE, "plot_bao_Hz.png"), dpi=160); plt.close()

# ---------- Heatmap/Contours χ² per TCR su griglia (BAO+SN+CC) ----------
def figure_grid(rs_ref_TCR, H_TCR_f):
    # Pre-carica SN cov e BAO per efficienza
    Csn  = tcr.load_sn_cov_auto(PATH_SN_COV)
    rows_bao, Cbao = tcr.read_bao_consensus(PATH_BAO_RES, PATH_BAO_COV)
    Cinv_bao = np.linalg.inv(Cbao)

    G = np.zeros((len(OM_GRID), len(H0_GRID)))
    for i, OM in enumerate(OM_GRID):
        for j, H in enumerate(H0_GRID):
            # set parametri late-time del TCR
            tcr.set_tcr_late_params(H0=H, Om=OM)
            # BAO
            chi2_bao_T,_ = tcr.chi2_BAO(rows_bao, Cinv_bao, H_TCR_f, tcr.Ok_eff_star, H, rs_ref_TCR)
            # SN
            chi2_sn_T,_,_ = tcr.chi2_SN_fullcov(PATH_SN, Csn, H_TCR_f, tcr.Ok_eff_star, H)
            # CC fullcov
            chi2_cc_T,_  = tcr.chi2_CC_fullcov_from_files(PATH_CC, PATH_CC_TAB, H_TCR_f)
            G[i,j] = chi2_bao_T + chi2_sn_T + chi2_cc_T

    H0g, OMg = np.meshgrid(H0_GRID, OM_GRID)

    # Heatmap
    plt.figure()
    im = plt.imshow(G, origin="lower", aspect="auto",
                    extent=[min(H0_GRID), max(H0_GRID), min(OM_GRID), max(OM_GRID)])
    cb = plt.colorbar(im); cb.set_label(r"$\chi^2_{\rm TCR}$")
    plt.xlabel(r"$H_0$ (TCR)"); plt.ylabel(r"$\Omega_m$ (late)")
    plt.tight_layout(); plt.savefig(os.path.join(BASE,"plot_tcr_chi2_heatmap_light.png"), dpi=160); plt.close()

    # Contours
    plt.figure()
    CS = plt.contour(H0g, OMg, G, levels=7)
    plt.clabel(CS, inline=True, fmt="%.0f")
    plt.xlabel(r"$H_0$ (TCR)"); plt.ylabel(r"$\Omega_m$ (late)")
    plt.tight_layout(); plt.savefig(os.path.join(BASE,"plot_tcr_chi2_contours_light.png"), dpi=160); plt.close()

def main():
    # imposta parametri late-time per TCR
    tcr.set_tcr_late_params(H0=H0_TCR, Om=OM_LATE)

    # funzioni H(z)
    H_LCDM_f, H_TCR_f = get_models()

    # rs: LCDM usa z_drag; TCR idem (consistente con BAO)
    args_early_LCDM = make_args_early_lcdm(rs_from="z_drag")
    rs_LCDM, _ = compute_rs_from_args(args_early_LCDM)

    args_early_TCR = make_args_early_tcr()
    rs_TCR, _ = compute_rs_from_args(args_early_TCR)

    # figure
    figure_sn(H_LCDM_f, H_TCR_f)
    figure_cc(H_LCDM_f, H_TCR_f)
    figure_bao(rs_LCDM, rs_TCR, H_LCDM_f, H_TCR_f)
    figure_grid(rs_TCR, H_TCR_f)

    print("[OK] Figure salvate in", BASE)

if __name__ == "__main__":
    main()