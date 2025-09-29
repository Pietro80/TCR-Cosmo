# TCR-Cosmo

> **Purpose**  
> This repository **is not a general-purpose framework**: it does not include automated tests, it does not accept “all” inputs, and it does not provide stable APIs.  
> It is a **minimal tool** to **reproduce the results and figures** of the low-redshift TCR-Cosmo preprint using SN Ia, compressed BAO, and Cosmic Chronometers.

---

## Why this repo

- Simple pipeline to recompute the main quantities (χ², AIC/BIC) and generate the preprint’s plots.  
- Clarity about **what is required** (data, scripts) and **how to run** the scripts to obtain the same outputs.  
- Code/text separation: the preprint PDF is in `docs/`.

---

> **Data note**: for licensing/rights reasons, the data files are included in this repository.

---

## Requirements

- Python **3.9+**
- Python dependencies (in `requirements.txt`):
  - `numpy`, `pandas`, `scipy`, `astropy`, `matplotlib`

## Required data

- **Supernovae (Pantheon+SH0ES)**
  - `Pantheon+SH0ES.dat`  
    https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat
  - `Pantheon+SH0ES_STAT+SYS.cov` — full covariance  
    https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov

- **Cosmic Chronometers (CC)**
  - `HzTable_MM_BC03.dat` — H(z) measurements
  - `data_MM20.dat` — table/covariance for `fullcov` mode  
    https://gitlab.com/mmoresco/CCcovariance

- **BAO DR12 (compressed)**
  - `BAO_consensus_results_dM_Hz.txt`
  - `BAO_consensus_covtot_dM_Hz.txt`  
    https://www.sdss3.org/science/boss_publications.php

## Run scripts

Main:

```bash
python3 tcr_run.py \
  --sn Pantheon+SH0ES.dat \
  --sn-cov Pantheon+SH0ES_STAT+SYS.cov \
  --cc HzTable_MM_BC03.dat \
  --cc-mode fullcov \
  --cc-cov-tab data_MM20.dat \
  --bao-results BAO_consensus_results_dM_Hz.txt \
  --bao-cov BAO_consensus_covtot_dM_Hz.txt \
  --Om-late 0.295 --H0-tcr 66.8 \
  --out final_single.csv \
  --show
```

Generate plots:

```bash
python3 plot_results.py
python3 plot_gamma.py
python3 plot_Tz.py
```

## What it does / what it doesn’t do

**Does**
- Loads the listed datasets and computes χ² contributions for **SN**, **BAO**, and **CC**.
- Produces the main preprint figures at **low redshift**.

**Doesn’t**
- It is not a general-purpose cosmo-fitting framework.
- It does not cover high-z nor all possible datasets/surveys.
- It does not expose stable APIs, does not include a test suite, and does not guarantee compatibility with arbitrary inputs.

---

## Reproducibility

- To obtain the same numbers/figures as in the preprint, use **the same data files** and **the same commands** shown above.
- For traceability, report **the repository version** and the versions of **Python/packages**.

---

## Citation

If you use this code or reproduce its results, please cite:
- The *TCR-Cosmo* preprint (author, title, year, arXiv/DOI if available).
- This repository (URL and version).

---

## License

Released under the **MIT License**. See `LICENSE`.

---

## Author & contact

- For issues or technical questions, open an **Issue** or submit a focused **Pull Request** (e.g., minor fixes to paths/figures).
