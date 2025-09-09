#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from scipy import integrate, linalg
from scipy.special import expit
from astropy import units as u
from astropy import constants as const
import io, re

# =========================================================
# Costanti
# =========================================================
c = const.c.to(u.km/u.s).value  # speed of light [km/s] from astropy
Tcmb = 2.7255         # K
r_d_fid = 147.09      # Mpc (valore fiduciale per scaling BAO)

def omega_gamma_from_T(Tcmb=Tcmb):
    return 2.469e-5*(Tcmb/2.7255)**4


# =========================================================
# Utility per caricamento file di testo numerici
# =========================================================
def _load_txt_flexible(path):
    """Caricatore robusto per file testo con separatori misti (spazi/virgole).
    - Ignora righe vuote o commentate con '#'
    - Usa pandas con sep r"[,\s]+" per gestire virgole e spazi
    - Converte a numerico, mette a NaN i token non numerici (es. 'Moresco'),
      poi elimina colonne/righe completamente NaN.
    - Fallback manuale se pandas dovesse fallire.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]

    try:
        df = pd.read_csv(
            io.StringIO("".join(raw)),
            sep=r"[,\s]+",          # <— differenza chiave: gestisce spazi O virgole
            engine="python",
            header=None
        )
        # forza numerico, scarta testi
        df = df.apply(pd.to_numeric, errors="coerce")
        # drop colonne tutte NaN (es. colonne con sole stringhe scartate)
        df = df.dropna(axis=1, how="all")
        # drop righe tutte NaN (righe vuote post-conversione)
        df = df.dropna(axis=0, how="all")
        arr = df.values
        # se ancora vuoto, lascia al fallback
        if arr.size > 0:
            return arr.astype(float)
    except Exception:
        pass

    # Fallback manuale ultra-robusto
    rows = []
    for ln in raw:
        toks = re.split(r"[,\s]+", ln.strip())
        vals = []
        for t in toks:
            try:
                vals.append(float(t))
            except ValueError:
                continue
        if vals:
            rows.append(vals)
    return np.array(rows, dtype=float)

# =========================================================
# Argomenti e parametri EARLY
# =========================================================
def early_params_from_args(args):
    return {
        "omega_b": args.omega_b_early,
        "omega_c": args.omega_c_early,
        "N_eff": args.N_eff,
        "h_early": args.h_early,
        "z_star": args.z_star,
        "z_drag": args.z_drag,
        "alpha_rs": args.alpha_rs,
    }

# =========================================================
# Early standard (LCDM-like per confronto)
# =========================================================
def E_early_std(z, pars):
    omega_gamma = omega_gamma_from_T(Tcmb)
    omega_r = omega_gamma*(1.0 + 0.2271*pars["N_eff"])
    Om = (pars["omega_b"] + pars["omega_c"]) / pars["h_early"]**2
    Or = omega_r / pars["h_early"]**2
    Ol = max(0.0, 1.0 - Om - Or)
    return np.sqrt(Or*(1+z)**4 + Om*(1+z)**3 + Ol)

def H_early_std(z, pars):
    return 100.0*pars["h_early"]*E_early_std(z, pars)

# Plasma primordiale (comune)
def Rb(z, pars):
    return 31.5*pars["omega_b"]*((Tcmb/2.7)**-4.0)*(1e3/(1.0+z))

def cs_z(z, pars):
    return c/np.sqrt(3.0*(1.0 + Rb(z, pars)))

# =========================================================
# Early TCR: gamma_early (recomb) + gamma_infl (banda log z)
# =========================================================
def w_d_early(z, zt, beta, w0=0.85, w1=0.10):
    """
    Logistica numericamente stabile in z:
    w = w1 + (w0 - w1) * expit(beta*(zt - z))
    """
    return w1 + (w0 - w1) * expit(beta*(zt - z))

def dw_d_early_dz(z, zt, beta, w0=0.85, w1=0.10):
    # derivata analitica della logistica stabile
    u = beta*(zt - z)
    sig = expit(u)
    dsig_dz = -beta * sig * (1.0 - sig)
    return (w0 - w1) * dsig_dz

def gamma_early_of_z(z, zt, beta, A, p, z_bbn):
    if z >= z_bbn:
        return 1.0
    w = w_d_early(z, zt, beta)
    return 1.0 + A*(w**p)

def dgamma_early_dz(z, zt, beta, A, p, z_bbn):
    if z >= z_bbn or A == 0.0:
        return 0.0
    w = w_d_early(z, zt, beta)
    dw = dw_d_early_dz(z, zt, beta)
    if w <= 0.0:
        return 0.0
    return A * p * (w**(p-1.0)) * dw

def Neff_of_z(z, Neff0, dNeff, zt, beta, z_bbn):
    """
    N_eff(z) dinamico, safe per BBN: per z>=z_bbn ritorna Neff0.
    Transizione logistica in z, numericamente stabile.
    """
    if (dNeff == 0.0) or (z >= z_bbn):
        return Neff0
    w = expit(beta*(zt - z))
    return Neff0 + dNeff*w

def _sigm(x):
    return expit(x)

def smooth_window_logz(z, z_on, z_off, beta):
    """
    Finestra liscia in log10 z: w ~ 0 fuori [z_on, z_off], ~1 dentro.
    Implementata come differenza di due logistic in L = log10(z).
    Nessun clipping necessario.
    """
    zz = max(z, 1.0 + 1e-12)
    L = np.log10(zz)
    Lon, Loff = np.log10(z_on), np.log10(z_off)
    w_on  = _sigm(beta*(L - Lon))
    w_off = _sigm(beta*(L - Loff))
    w = w_on - w_off
    if w < 0.0:  # sicurezza numerica estrema
        w = 0.0
    if w > 1.0:
        w = 1.0
    return w

def dwindow_logz_dz(z, z_on, z_off, beta):
    """
    Derivata analitica della finestra in log10 z:
    dw/dz = beta/(z ln 10) * [σ(1-σ)|on - σ(1-σ)|off]
    """
    zz = max(z, 1.0 + 1e-12)
    L = np.log10(zz)
    Lon, Loff = np.log10(z_on), np.log10(z_off)
    s_on  = _sigm(beta*(L - Lon))
    s_off = _sigm(beta*(L - Loff))
    dsig_dL_on  = beta * s_on  * (1.0 - s_on)
    dsig_dL_off = beta * s_off * (1.0 - s_off)
    dL_dz = 1.0 / (zz * math.log(10.0))
    return (dsig_dL_on - dsig_dL_off) * dL_dz

def gamma_infl_of_z(z, z_on, z_off, beta_infl, A_infl):
    w = smooth_window_logz(z, z_on, z_off, beta_infl)
    return 1.0 + A_infl*w

def dgamma_infl_dz(z, z_on, z_off, beta_infl, A_infl):
    if A_infl == 0.0:
        return 0.0
    dw = dwindow_logz_dz(z, z_on, z_off, beta_infl)
    return A_infl * dw

def H_TCR_early(z, pars,
                okeff_early,
                A_early, p_early, zt_early, beta_early,
                A_infl, zinfl_on, zinfl_off, beta_infl,
                z_bbn, dNeff,
                den_floor=1e-4):
    """
    Early TCR senza CDM:
    - base bare: barioni + radiazione + curvatura eff. opzionale
    - gamma_infl(z): finestra inflazionaria su banda log(z)
    - gamma_early(z): dressing vicino al recombination
    """
    hE = pars["h_early"]
    Neff_here = Neff_of_z(z, pars["N_eff"], dNeff, zt_early, beta_early, z_bbn)

    omega_gamma = omega_gamma_from_T(Tcmb)
    Or = (omega_gamma*(1.0 + 0.2271*Neff_here)) / hE**2
    Om_b = pars["omega_b"]/hE**2
    Ol = max(0.0, 1.0 - Om_b - Or - okeff_early)

    H0 = 100.0*hE
    H_bare = H0*np.sqrt(Or*(1+z)**4 + Om_b*(1+z)**3 + okeff_early*(1+z)**2 + Ol)

    gam_infl  = gamma_infl_of_z(z, zinfl_on, zinfl_off, beta_infl, A_infl)
    dgam_infl = dgamma_infl_dz(z, zinfl_on, zinfl_off, beta_infl, A_infl)

    gam_early  = gamma_early_of_z(z, zt_early, beta_early, A_early, p_early, z_bbn)
    dgam_early = dgamma_early_dz(z, zt_early, beta_early, A_early, p_early, z_bbn)

    gam_tot = gam_infl * gam_early
    dgam_tot = dgam_infl * gam_early + dgam_early * gam_infl

    den = 1.0 + (1.0+z)*dgam_tot
    if den <= den_floor:
        # protezione fisica: meglio segnalare se scende troppo
        den = den_floor
    return gam_tot*H_bare/den

def H_early_dispatch(z, args, pars):
    if args.early_mode == "std":
        return H_early_std(z, pars)
    return H_TCR_early(
        z, pars,
        args.okeff_early,
        args.A_early, args.p_early, args.zt_early, args.beta_early,
        args.A_infl, args.zinfl_on, args.zinfl_off, args.beta_infl,
        args.z_bbn, args.dNeff_early
    )

# =========================================================
# Caricamento dataset Cosmic Chronometers (MM20)
# =========================================================
def load_CC_data_MM20(data_path):
    """
    data_MM20.dat atteso come >=3 colonne: z, H(z), sigma_stat (in km/s/Mpc o %).
    Ritorna (z, H_obs, sigma_stat_abs).
    """
    arr = _load_txt_flexible(data_path)
    if arr.shape[1] < 3:
        raise ValueError("data_MM20.dat: mi aspetto almeno 3 colonne (z, H, sigma_stat).")
    z  = arr[:,0].astype(float)
    H  = arr[:,1].astype(float)
    s3 = arr[:,2].astype(float)

    # Heuristica robusta: se la 3a colonna sembra percentuale, converti → assoluta
    # (es. valori tipici pochi unità e s3/H << 1)
    med_val   = float(np.nanmedian(s3))
    med_ratio = float(np.nanmedian(s3 / np.clip(H, 1e-6, None)))
    looks_percent = (med_val < 50.0) and (med_ratio < 0.5)
    sigma_abs = (s3*0.01*H) if looks_percent else s3

    # Evita zeri/sottostime patologiche
    sigma_abs = np.clip(sigma_abs, 1e-3, None)
    return z, H, sigma_abs


# =========================================================
# χ² dei cronometri cosmici con covarianza completa
# =========================================================
def chi2_CC_fullcov_from_files(cc_data_path, cc_table_path, H_func):
    """
    χ² CC usando C = C_stat + C_sys.
    - cc_data_path = data_MM20.dat
    - cc_table_path = HzTable_MM_BC03.dat (o matrice NxN già pronta)
    """
    z, Hobs, sig_stat = load_CC_data_MM20(cc_data_path)
    N = len(z)
    Cstat = np.diag(sig_stat**2)
    # PASSA z come riferimento per l'allineamento
    Csys  = load_CC_cov_or_table(cc_table_path, N, z_ref=z)
    C = Cstat + Csys

    Hth = np.array([H_func(zz) for zz in z])
    d = Hth - Hobs

    cho = linalg.cho_factor(C, lower=True, check_finite=False)
    alpha = linalg.cho_solve(cho, d, check_finite=False)
    chi2 = float(d @ alpha)
    return chi2, N

# =========================================================
# Costruzione matrice di covarianza sistematica per i CC
# =========================================================
def load_CC_cov_or_table(table_path, N_expected, z_ref=None, z_tol=5e-2):
    """
    Costruisce la covarianza sistematica per i CC.
    - Se il file è NxN: la ritorna direttamente.
    - Altrimenti assume tabella per-punto: [z, H, sigma_stat, comp1, comp2, ...].
      Ogni 'compk' è un contributo sistematico per-punto (spesso in % di H).
      Modello: ciascun componente è 100% correlato tra redshift ⇒
               C_sys = Σ_k (v_k v_k^T), con v_k i vettori (assoluti) per componente k.
    """
    arr = _load_txt_flexible(table_path)

    # Caso 1: cov piena già pronta NxN
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] == N_expected:
        return arr.astype(float)

    # Caso 2: tabella per-punto
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((N_expected, N_expected), dtype=float)

    z_tab = arr[:, 0].astype(float)
    H_tab = arr[:, 1].astype(float)

    # Componenti sistematiche: dalla quarta colonna in poi
    comps = arr[:, 3:] if arr.shape[1] > 3 else np.zeros((arr.shape[0], 0))
    M = comps.shape[1]

    # Se non ho componenti, niente sistematici
    if M == 0:
        return np.zeros((N_expected, N_expected), dtype=float)

    # Decide se i comps sono percentuali (tipico) o assoluti
    # Heuristica robusta: se mediana(comp) < 50 e comp/H mediana < 0.5 ⇒ % di H
    med_comp = float(np.nanmedian(comps))
    ratio_med = float(np.nanmedian(comps / np.clip(H_tab[:, None], 1e-6, None)))
    comps_are_percent = (med_comp < 50.0) and (ratio_med < 0.5)

    # Matching ai redshift dei CC
    if z_ref is None:
        # senza z_ref non posso riallineare: fallback a diagonale nulla
        return np.zeros((N_expected, N_expected), dtype=float)

    z_ref = np.asarray(z_ref, dtype=float)

    # Costruisco i vettori v_k (dimensione N_expected) per ciascun componente k
    V = np.zeros((N_expected, M), dtype=float)
    for i, z0 in enumerate(z_ref):
        j = np.argmin(np.abs(z_tab - z0))
        if abs(z_tab[j] - z0) <= z_tol:
            if M > 0:
                if comps_are_percent:
                    # % di H → assoluto per ciascun componente
                    V[i, :] = (comps[j, :] * 0.01) * H_tab[j]
                else:
                    # già in km/s/Mpc
                    V[i, :] = comps[j, :]
        # altrimenti riga resta a zero (conservativo)

    # C_sys piena = somma dei rank-1 (outer product) per componente
    # equivalente a: C_sys = V @ V^T
    Csys = V @ V.T
    return Csys

# Distanza comovente early e sound horizon (USANO il dispatcher!)
def comoving_distance_early(z, args, pars):
    integ,_ = integrate.quad(lambda zz: c/H_early_dispatch(zz, args, pars),
                             0.0, z, limit=500, epsabs=1e-8, epsrel=1e-8)
    return integ

def r_s_reference(args, pars):
    z_ref   = pars["z_star"] if args.rs_from == "z_star" else pars["z_drag"]
    # Upper bound ragionevole e dipendente dalla finestra
    z_upper = max(1e6, getattr(args, "zinfl_off", 1e6)*10.0)
    f = lambda zz: cs_z(zz, pars)/H_early_dispatch(zz, args, pars)

    # Spezzatura per stabilità numerica
    z_mid = min(1e5, z_upper)
    integ1,_ = integrate.quad(f, z_ref, z_mid, limit=500, epsabs=1e-6, epsrel=1e-6)
    integ2 = 0.0
    if z_upper > z_mid:
        integ2,_ = integrate.quad(f, z_mid, z_upper, limit=500, epsabs=1e-6, epsrel=1e-6)
    return pars["alpha_rs"]*(integ1 + integ2)

# Vincolo CMB (theta*)
def chi2_theta_star(args, pars):
    if args.theta_prior == "none":
        return 0.0, 0
    rs = r_s_reference(args, pars)
    D_C_star = comoving_distance_early(pars["z_star"], args, pars)
    th = rs/D_C_star
    return ((100.0*th - args.theta100_P)/args.sigma_theta100)**2, 1

# =========================================================
# Late-time: LCDM e TCR (come base)
# =========================================================
# =========================================================
# Varianti con Astropy Units (fail-fast) per H(z) e distanze
# =========================================================
def _z_quantity(z):
    """Converte z in Quantity adimensionale (u.one)."""
    if isinstance(z, u.Quantity):
        return z.to(u.one)
    return (np.float64(z)) * u.one

def H_LCDM_Q(z):
    """H(z) in LCDM come Quantity [km/s/Mpc]."""
    zz = _z_quantity(z).value  # z è adimensionale
    Ol = 1.0 - Om_LCDM - Ok_LCDM
    H = H0_LCDM*np.sqrt(Om_LCDM*(1+zz)**3 + Ok_LCDM*(1+zz)**2 + Ol)
    return np.float64(H) * (u.km/u.s/u.Mpc)

def H_TCR_obs_Q(z, Ok_eff, zt_mix, beta_mix, A, p):
    """H(z) TCR come Quantity [km/s/Mpc]."""
    val = H_TCR_obs(np.float64(z), Ok_eff, zt_mix, beta_mix, A, p)
    return np.float64(val) * (u.km/u.s/u.Mpc)

def D_C_Q(z, H_func_Q):
    """Distanza comovente come Quantity [Mpc].
    H_func_Q deve restituire Quantity [km/s/Mpc]."""
    zf = _z_quantity(z).value
    def integrand(zz):
        Hz = H_func_Q(zz).to_value(u.km/u.s/u.Mpc)
        return c/Hz
    integ,_ = integrate.quad(integrand, 0.0, zf, limit=400, epsabs=1e-8, epsrel=1e-8)
    return np.float64(integ) * u.Mpc

def D_M_Q(z, H_func_Q, Ok, H0):
    """Distanza trasversa D_M come Quantity [Mpc].
    Ok adimensionale, H0 in km/s/Mpc."""
    Dc = D_C_Q(z, H_func_Q).to_value(u.Mpc)
    if abs(Ok) < 1e-12:
        return np.float64(Dc) * u.Mpc
    s = math.sqrt(abs(Ok))
    x = s*(H0*Dc/c)
    val = (c/H0)*(math.sinh(x)/s if Ok>0 else math.sin(x)/s)
    return np.float64(val) * u.Mpc

H0_LCDM, Om_LCDM, Ok_LCDM = 67.4, 0.315, 0.0
def H_LCDM(z):
    Ol = 1.0 - Om_LCDM - Ok_LCDM
    return H0_LCDM*np.sqrt(Om_LCDM*(1+z)**3 + Ok_LCDM*(1+z)**2 + Ol)

H0_b, Om_b = 62.0, 0.32
wbh0 = 0.01

def logistic_dec(z, value0, zt, beta):
    # decrescente con z, stabile
    return value0/(1.0 + math.exp(beta*(z-zt)))

A_default, p_default = 0.29, 0.94
q, r_exp, Bcoef, Ccoef = 3.0, 3.0, 1.0, 1.0
fdeep0, fshal0 = 0.15, 0.55
wgal0 = 0.045
xi = 2.0
zt_void, beta_void = 0.7, 3.0
zt_gal,  beta_gal  = 0.7, 4.0
Ok_eff_star, zt_mix_star, beta_mix_star = -0.02, 0.75, 3.2

def f_v_deep_z(z): return logistic_dec(z, fdeep0, zt_void, beta_void)
def f_v_shallow_z(z): return logistic_dec(z, fshal0, zt_void, beta_void)
def w_gal_z(z): return logistic_dec(z, wgal0, zt_gal, beta_gal)
def w_bh_const(z): return wbh0

def w_d(z, zt_mix, beta_mix, w0=0.85, w1=0.10):
    return w1 + (w0 - w1)/(1.0 + math.exp(-beta_mix*(zt_mix - z)))

def f_v_eff(z, zt_mix, beta_mix):
    fd, fs = f_v_deep_z(z), f_v_shallow_z(z)
    inner = w_d(z, zt_mix, beta_mix)*(fd**xi) + (1.0 - w_d(z, zt_mix, beta_mix))*(fs**xi)
    return inner**(1.0/xi)

def H_bare(z, Ok_eff):
    Ol_eff = 1.0 - Om_b - Ok_eff
    return H0_b*np.sqrt(Om_b*(1+z)**3 + Ok_eff*(1+z)**2 + Ol_eff)

def gamma_of_z(z, Ok_eff, zt_mix, beta_mix, A, p):
    fv = f_v_eff(z, zt_mix, beta_mix)
    return 1.0 + A*(fv**p)*(1.0 - Bcoef*(w_bh_const(z)**q) - Ccoef*(w_gal_z(z)**r_exp))

def dgamma_dz(z, Ok_eff, zt_mix, beta_mix, A, p, eps_rel=1e-4):
    # Derivata numerica con passo relativo robusto
    h = max(1e-4, eps_rel*max(1.0, z))
    return (gamma_of_z(z+h, Ok_eff, zt_mix, beta_mix, A, p) -
            gamma_of_z(z-h, Ok_eff, zt_mix, beta_mix, A, p))/(2*h)

def H_TCR_obs(z, Ok_eff, zt_mix, beta_mix, A, p, den_floor=1e-4):
    gam = gamma_of_z(z, Ok_eff, zt_mix, beta_mix, A, p)
    dgam = dgamma_dz(z, Ok_eff, zt_mix, beta_mix, A, p)
    den = 1.0 + (1.0+z)*dgam
    if den <= den_floor:
        den = den_floor
    return gam*H_bare(z, Ok_eff)/den

# =========================================================
# Distanze e BAO (senza s_rd)
# =========================================================
def D_C(z, H_func):
    integ,_ = integrate.quad(lambda zz: c/H_func(zz), 0, z, limit=400, epsabs=1e-8, epsrel=1e-8)
    return integ

def D_M(z, H_func, Ok, H0):
    Dc = D_C(z, H_func)
    if abs(Ok) < 1e-12:
        return Dc
    s = math.sqrt(abs(Ok))
    x = s*(H0*Dc/c)
    return (c/H0)*(math.sinh(x)/s if Ok>0 else math.sin(x)/s)

def read_bao_consensus(path_results, path_cov):
    df = pd.read_csv(path_results, sep=r"\s+", engine="python", header=None, comment="#")
    df.columns = ["z","label","value"]
    rows = [(float(r.z), str(r.label), float(r.value)) for _,r in df.iterrows()]
    C = np.loadtxt(path_cov)
    return rows, C

def bao_predict_vector(rows, H_func, Ok, H0, rs_ref):
    vec = []
    for z, lab, _ in rows:
        if lab.startswith("dM"):
            DM = D_M(z, H_func, Ok, H0) * (r_d_fid/rs_ref)
            vec.append(DM)
        elif lab.startswith("Hz"):
            Hz = H_func(z) * (rs_ref/r_d_fid)
            vec.append(Hz)
    return np.array(vec)

def chi2_BAO(rows, Cinv, H_func, Ok, H0, rs_ref):
    y = np.array([val for _,_,val in rows])
    pred = bao_predict_vector(rows, H_func, Ok, H0, rs_ref)
    d = pred - y
    return float(d @ Cinv @ d), len(y)

# =========================================================
# CC e SN
# =========================================================
def chi2_CC(cc_path, H_func):
    # Prova: CSV a virgole con 3 colonne numeriche (ignora eventuale "reference")
    try:
        cc = pd.read_csv(
            cc_path, comment="#", engine="python",
            sep=",", header=None, usecols=[0,1,2],
            names=["z","H","sigma"]
        )
    except Exception:
        # Fallback: separatore a spazi
        cc = pd.read_csv(
            cc_path, comment="#", engine="python",
            sep=r"\s+", header=None, usecols=[0,1,2],
            names=["z","H","sigma"]
        )

    H_th = np.array([H_func(zz) for zz in cc["z"].values])
    chi2 = float(np.sum(((H_th - cc["H"].values)/cc["sigma"].values)**2))
    return chi2, len(cc)

def load_sn_cov_auto(path):
    with open(path, "r") as f:
        txt = f.read()
    toks = pd.core.tools.numeric._FLOAT_RE.findall(txt) if hasattr(pd.core.tools.numeric, "_FLOAT_RE") else None
    if toks is None:
        # fallback robusto
        import re
        toks = re.split(r"[,\s]+", txt.strip())
    vals = np.array([float(t) for t in toks if t != ""], dtype=float)
    try:
        first_int = int(round(vals[0]))
    except Exception:
        first_int = -1
    if first_int > 100 and len(vals) == first_int*first_int + 1:
        N = first_int
        return vals[1:].reshape(N, N)
    L = len(vals)
    N = int(round(np.sqrt(L)))
    if N*N == L:
        return vals.reshape(N, N)
    raise ValueError("Unrecognized covariance layout")

def chi2_SN_fullcov(sn_dat_path, C, H_func, Ok, H0):
    sn = pd.read_csv(sn_dat_path, sep=r"\s+", engine="python", comment="#")
    z = sn["zHD"].values
    mu_obs = sn["MU_SH0ES"].values
    N = len(z)
    mu_th = np.array([5*np.log10((1+zz)*D_M(zz, H_func, Ok, H0)*1e6/10.0) for zz in z])
    Delta = mu_th - mu_obs
    cho = linalg.cho_factor(C, lower=True, check_finite=False)
    alpha = linalg.cho_solve(cho, Delta, check_finite=False)
    ones = np.ones(N)
    beta = linalg.cho_solve(cho, ones, check_finite=False)
    term1 = float(Delta @ alpha)
    Aterm = float(ones @ beta)
    Bterm = float(ones @ alpha)
    chi2 = term1 - (Bterm*Bterm)/Aterm
    return chi2, N, Bterm/Aterm

# =========================================================
# Scan inflazione (utility)
# =========================================================
def scan_inflation_band(args, pars, A_vals=(3,6,9), on_vals=(2e3,5e3,1e4), off_factors=(3,5,10,20,30)):
    rows = []
    for A in A_vals:
        for z_on in on_vals:
            for k in off_factors:
                z_off = z_on * k
                class _Tmp: pass
                t = _Tmp()
                for name in vars(args):
                    setattr(t, name, getattr(args, name))
                t.A_infl, t.zinfl_on, t.zinfl_off = A, z_on, z_off
                rs = r_s_reference(t, pars)
                rows.append((A, z_on, z_off, rs))
    df = pd.DataFrame(rows, columns=["A_infl","zinfl_on","zinfl_off","r_s_Mpc"])
    df["delta_rs"] = (df["r_s_Mpc"] - 147.0).abs()
    df = df.sort_values("delta_rs")
    print(df.head(20).to_string(index=False))
    df.to_csv("tcr_inflation_rs_scan.csv", index=False)


# =========================================================
# Helper: set late-time TCR constants from the runner (fair comparison)
# =========================================================
def set_tcr_late_params(H0=None, Om=None):
    """
    Aggiorna i parametri globali del ramo bare TCR in modo esplicito dal runner.
    Questo evita 'barare' fissando H0/Ωm in modo rigido quando si confronta con ΛCDM.
    """
    global H0_b, Om_b
    if H0 is not None:
        H0_b = float(H0)
    if Om is not None:
        Om_b = float(Om)

# =========================================================
# BAO: versione con dettagli per debug (contributi ~diag)
# =========================================================
def chi2_BAO_with_details(rows, Cinv, H_func, Ok, H0, rs_ref):
    """
    Restituisce anche un elenco di dettagli per punto: (z, label, residuo, contributo approx).
    Nota: con covarianza piena, i contributi punto-per-punto non sono unici; qui usiamo
    d * (Cinv @ d) come proxy utile per la diagnostica.
    """
    y = np.array([val for _,_,val in rows])
    pred = bao_predict_vector(rows, H_func, Ok, H0, rs_ref)
    d = pred - y
    chi2 = float(d @ Cinv @ d)
    contrib = d * (Cinv @ d)  # proxy
    details = []
    for (z,label,_), resid, contr in zip(rows, d, contrib):
        details.append((float(z), str(label), float(resid), float(contr)))
    return chi2, len(y), details