#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import tcr_core as tcr

from tcr_core import (
    A_default, p_default, Ok_eff_star, zt_mix_star, beta_mix_star,
    early_params_from_args, r_s_reference, chi2_theta_star,
    H_LCDM, H_TCR_obs, H0_LCDM,
    read_bao_consensus, chi2_BAO, chi2_SN_fullcov, chi2_CC, load_sn_cov_auto,
    chi2_CC_fullcov_from_files
)

def parse_list_of_floats(s):
    return [float(x) for x in s.split(",")]

def add_early_tcr_args(ap):
    ap.add_argument("--okeff-early", type=float, default=0.0)
    ap.add_argument("--A-early",   type=float, default=0.0)
    ap.add_argument("--p-early",   type=float, default=1.0)
    ap.add_argument("--zt-early",  type=float, default=3000.0)
    ap.add_argument("--beta-early",type=float, default=3.0)
    ap.add_argument("--A-infl",    type=float, default=6.0)
    ap.add_argument("--zinfl-on",  type=float, default=3e3)
    ap.add_argument("--zinfl-off", type=float, default=3e4)
    ap.add_argument("--beta-infl", type=float, default=3.0)
    ap.add_argument("--z-bbn",       type=float, default=1e8)
    ap.add_argument("--dNeff-early", type=float, default=0.0)

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn", required=True)
    ap.add_argument("--sn-cov", required=True)
    ap.add_argument("--cc", required=True)
    ap.add_argument("--bao-results", required=True)
    ap.add_argument("--bao-cov", required=True)

    ap.add_argument("--early-mode", choices=["std","tcr"], default="tcr")
    ap.add_argument("--omega-b-early", type=float, default=0.02237)
    ap.add_argument("--omega-c-early", type=float, default=0.0)
    ap.add_argument("--N_eff", type=float, default=3.046)
    ap.add_argument("--h-early", type=float, default=0.62)
    ap.add_argument("--rs-from", choices=["z_star","z_drag"], default="z_drag")
    ap.add_argument("--z-star", type=float, default=1089.0)
    ap.add_argument("--z-drag", type=float, default=1059.0)
    ap.add_argument("--alpha-rs", type=float, default=1.0)

    ap.add_argument("--theta-prior", choices=["none","planck"], default="none")
    ap.add_argument("--theta100-P", dest="theta100_P", type=float, default=1.0411)
    ap.add_argument("--sigma-theta100", type=float, default=0.0003)

    ap.add_argument("--A", type=float, default=A_default)
    ap.add_argument("--p", type=float, default=p_default)
    ap.add_argument("--okeff", type=float, default=Ok_eff_star)
    ap.add_argument("--zt", type=float, default=zt_mix_star)
    ap.add_argument("--beta", type=float, default=beta_mix_star)

    add_early_tcr_args(ap)

    # fair comparison options
    ap.add_argument("--H0-tcr", type=float, default=62.0,
                    help="H0 del ramo TCR late-time")
    ap.add_argument("--Om-late", type=float, default=0.32,
                    help="Omega_m del ramo TCR late-time; se None, derivata da early ( (omega_b+omega_c)/h^2 )")

    ap.add_argument("--shared-rs", action="store_true",
                    help="Forza r_s TCR = r_s LCDM (diagnostica)")

    # debug BAO
    ap.add_argument("--bao-debug", action="store_true",
                    help="Stampa contributi BAO punto-per-punto e test Ok=0")

    # mini grid-search
    ap.add_argument("--scan", action="store_true",
                    help="Esegue una mini grid-search su Om-late e H0-tcr")
    ap.add_argument("--scan-om", type=parse_list_of_floats, default=None,
                    help="Lista valori per Om-late (es: 0.26,0.28,0.30,0.32,0.34)")
    ap.add_argument("--scan-h0", type=parse_list_of_floats, default=None,
                    help="Lista valori per H0-tcr (es: 60,62,64,66,67.4)")
    
    ap.add_argument("--cc-mode", choices=["diag","fullcov"], default="diag",
                    help="Usa covarianza completa per CC se 'fullcov' e se --cc-cov-tab è fornito.")
    ap.add_argument("--cc-cov-tab", default=None,
                    help="Percorso a HzTable_MM_*.dat oppure a una matrice NxN di covarianza sistematica per CC.")


    # i/o
    ap.add_argument("--out", default="tcr_fullcov_compare_results.csv")
    ap.add_argument("--show", action="store_true")
    return ap

def compute_rs(args):
    pars_early = early_params_from_args(args)
    return r_s_reference(args, pars_early), pars_early

def build_models_and_data(args, rs_ref_LCDM, rs_ref_TCR, pars_early, args_LCDM, pars_early_LCDM):
    rows_bao, Cbao = read_bao_consensus(args.bao_results, args.bao_cov)
    Cinv_bao = np.linalg.inv(Cbao)
    Csn = load_sn_cov_auto(args.sn_cov)

    # Prior CMB separato per modello (MODIFICA MINIMA: solo qui)
    chi2_cmb_LCDM,_ = chi2_theta_star(args_LCDM, pars_early_LCDM)
    chi2_cmb_TCR,_  = chi2_theta_star(args, pars_early)

    # H(z)
    H_LCDM_f = lambda z: H_LCDM(z)
    H_TCR_f  = lambda z: H_TCR_obs(z, args.okeff, args.zt, args.beta, args.A, args.p)
    return rows_bao, Cinv_bao, Csn, chi2_cmb_LCDM, chi2_cmb_TCR, H_LCDM_f, H_TCR_f

def one_eval(args, rs_ref_LCDM, rs_ref_TCR, rows_bao, Cinv_bao, Csn, chi2_cmb_L, chi2_cmb_T, H_LCDM_f, H_TCR_f):
    
    rows_bao, Cbao = read_bao_consensus(args.bao_results, args.bao_cov)
    Cinv_bao = np.linalg.inv(Cbao)
    Csn = load_sn_cov_auto(args.sn_cov)

    # funzioni H(z)
    H_LCDM_f = lambda z: H_LCDM(z)
    H_TCR_f  = lambda z: H_TCR_obs(z, args.okeff, args.zt, args.beta, args.A, args.p)
    
    chi2_bao_L,_ = chi2_BAO(rows_bao, Cinv_bao, H_LCDM_f, 0.0, H0_LCDM, rs_ref_LCDM)
    chi2_bao_T,_ = chi2_BAO(rows_bao, Cinv_bao, H_TCR_f, args.okeff, args.H0_tcr, rs_ref_TCR)
    chi2_sn_L,_,dmu_L = chi2_SN_fullcov(args.sn, Csn, H_LCDM_f, 0.0, H0_LCDM)
    chi2_sn_T,_,dmu_T = chi2_SN_fullcov(args.sn, Csn, H_TCR_f, args.okeff, args.H0_tcr)

    # --- QUI: scelta CC diagonale vs covarianza piena ---
    if args.cc_mode == "fullcov" and args.cc_cov_tab is not None:
        chi2_cc_L,_ = chi2_CC_fullcov_from_files(args.cc, args.cc_cov_tab, H_LCDM_f)
        chi2_cc_T,_ = chi2_CC_fullcov_from_files(args.cc, args.cc_cov_tab, H_TCR_f)
    else:
        chi2_cc_L,_ = chi2_CC(args.cc, H_LCDM_f)
        chi2_cc_T,_ = chi2_CC(args.cc, H_TCR_f)

    #tot_L = chi2_cmb_L + chi2_bao_L + chi2_sn_L + chi2_cc_L
    #tot_T = chi2_cmb_T + chi2_bao_T + chi2_sn_T + chi2_cc_T

    tot_L = chi2_bao_L + chi2_sn_L + chi2_cc_L
    tot_T = chi2_bao_T + chi2_sn_T + chi2_cc_T

    summary = pd.DataFrame({
        "Model": ["LCDM","TCR"],
        "r_s_ref_Mpc":[round(rs_ref_LCDM,3), round(rs_ref_TCR,3)],
        "chi2_BAO":[chi2_bao_L, chi2_bao_T],
        "chi2_SN_fullcov":[chi2_sn_L, chi2_sn_T],
        "chi2_CC":[chi2_cc_L, chi2_cc_T],
        "chi2_TOTAL":[tot_L, tot_T],
        "Delta_mu_best":[dmu_L, dmu_T]
    }).round(3)
    return summary

def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    # 1) r_s per TCR dall'early corrente  ← (DEVE venire prima!)
    rs_ref_TCR, pars_early = compute_rs(args)

    # 2) r_s per LCDM: prior CMB su z_star, BAO su z_drag
    from types import SimpleNamespace
    args_LCDM = SimpleNamespace(**vars(args))
    args_LCDM.early_mode = "std"
    args_LCDM.omega_c_early = 0.12
    args_LCDM.h_early = H0_LCDM/100.0

    # PRIOR CMB su z_star
    args_LCDM.rs_from = "z_star"
    args.theta_prior = "planck"
    args_LCDM.theta_prior = "planck"
    rs_ref_LCDM_CMB, pars_early_LCDM = compute_rs(args_LCDM)

    # BAO su z_drag
    args_LCDM_bao = SimpleNamespace(**vars(args_LCDM))
    args_LCDM_bao.rs_from = "z_drag"
    rs_ref_LCDM_BAO, _ = compute_rs(args_LCDM_bao)

    # usa questo per i BAO
    rs_ref_LCDM = rs_ref_LCDM_BAO

    # 3) Costruzione dati e modelli
    rows_bao, Cinv_bao, Csn, chi2_cmb_L, chi2_cmb_T, H_LCDM_f, H_TCR_f = \
        build_models_and_data(args, rs_ref_LCDM, rs_ref_TCR, pars_early, args_LCDM, pars_early_LCDM)
    
    # GRID-SEARCH --------------------------------------------------------------
    if args.scan:
        # build grids
        om_grid = args.scan_om if args.scan_om is not None else [0.26,0.28,0.30,0.32,0.34]
        h0_grid = args.scan_h0 if args.scan_h0 is not None else [60,62,64,66,67.4]

        records = []
        for OM in om_grid:
            for H in h0_grid:
                # set late params
                tcr.set_tcr_late_params(H0=H, Om=OM)
                # eval
                s = one_eval(args, rs_ref_LCDM, rs_ref_TCR, rows_bao, Cinv_bao, Csn, chi2_cmb_L, chi2_cmb_T, H_LCDM_f, H_TCR_f)
                # pick TCR row
                row = s[s["Model"]=="TCR"].iloc[0].to_dict()
                row.update({"Om_late": OM, "H0_tcr": H})
                records.append(row)
        grid = pd.DataFrame.from_records(records)

        # Add AIC/BIC for TCR
        N = 1701 + 15 + 30  # approx number of data points
        k_TCR = 6  # Om_late, H0_tcr, beta, zt, A, p
        grid["AIC"] = grid["chi2_TOTAL"] + 2*k_TCR
        grid["BIC"] = grid["chi2_TOTAL"] + k_TCR*np.log(N)

        grid = grid.sort_values("chi2_TOTAL").reset_index(drop=True)

        # print best
        best = grid.iloc[0]
        print("\n=== MINI GRID-SEARCH (TCR) ===")
        print("Migliore per χ² totale:  Om_late={Om_late:.3f}, H0_tcr={H0_tcr:.3f},  χ²={chi2_TOTAL:.3f}".format(**best))

        # save both grid and best
        grid_out = args.out.replace(".csv","_grid.csv")
        best_out = args.out.replace(".csv","_best.csv")
        grid.to_csv(grid_out, index=False)
        best.to_frame().T.to_csv(best_out, index=False)
        print(f"[OK] Grid salvata: {grid_out}")
        print(f"[OK] Best salvato: {best_out}")

        # mini report top-3
        top3 = grid.head(3).copy()
        print("\nTop-3 per χ² (con AIC/BIC):")
        for i, r in enumerate(top3.itertuples(index=False), start=1):
            print(f"  {i}) Om={r.Om_late:.3f}, H0={r.H0_tcr:.3f} | χ²={r.chi2_TOTAL:.3f} | AIC={r.AIC:.3f} | BIC={r.BIC:.3f}")

        # show (optional)
        if args.show:
            with pd.option_context("display.width", 160,
                                   "display.max_rows", None,
                                   "display.float_format", lambda x: f"{x:,.3f}"):
                print("\nPrime 10 combinazioni:")
                print(grid.head(10).to_string(index=False))
        return

    # SINGLE EVAL --------------------------------------------------------------
    # Set late-time params (fair). If Om-late is None, derive from early
    if args.Om_late is None:
        Om_from_early = (args.omega_b_early + args.omega_c_early)/(args.h_early**2)
    else:
        Om_from_early = args.Om_late
    tcr.set_tcr_late_params(H0=args.H0_tcr, Om=Om_from_early)

    summary = one_eval(args, rs_ref_LCDM, rs_ref_TCR, rows_bao, Cinv_bao, Csn, chi2_cmb_L, chi2_cmb_T, H_LCDM_f, H_TCR_f)
    summary.to_csv(args.out, index=False)
    print(f"[OK] CSV scritto: {args.out}")

    if args.show:
        with pd.option_context("display.width", 140,
                               "display.max_rows", None,
                               "display.float_format", lambda x: f"{x:,.3f}"):
            print("\n=== RISULTATI ===")
            print(summary.to_string(index=False))

if __name__ == "__main__":
    main()