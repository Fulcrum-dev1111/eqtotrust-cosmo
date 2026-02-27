# ingest_pantheon.py
# Downloads official Pantheon+ data and covariance matrix from the official repo.
# Ensures row ordering is preserved so data and covariance align perfectly.

import pandas as pd
import numpy as np
import urllib.request
import os

DATA_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
COV_URL = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"

DATA_CSV = "pantheon_plus.csv"
COV_FILE = "pantheon_plus_cov.txt"

if os.path.exists(DATA_CSV) and os.path.exists(COV_FILE):
    print(f"{DATA_CSV} and {COV_FILE} already exist, skipping download.")
else:
    print("Downloading official Pantheon+ data...")
    urllib.request.urlretrieve(DATA_URL, "pantheon_raw.txt")

    print("Processing data (preserving original row order for covariance alignment)...")
    df = pd.read_csv("pantheon_raw.txt", sep=r'\s+')

    out_df = pd.DataFrame({
        'z': df['zHD'],
        'mu': df['m_b_corr'],
        'sigma_mu': df['m_b_corr_err_DIAG']
    })

    out_df.to_csv(DATA_CSV, index=False)
    print(f"Saved {len(out_df)} rows to {DATA_CSV}")
    os.remove("pantheon_raw.txt")

    print("Downloading official Pantheon+ STAT+SYS covariance matrix...")
    urllib.request.urlretrieve(COV_URL, "pantheon_cov_raw.txt")

    print("Processing covariance matrix...")
    with open("pantheon_cov_raw.txt", "r") as f:
        lines = f.readlines()

    n_cov = int(lines[0].strip())
    print(f"  Covariance matrix dimension from file header: {n_cov}")

    if n_cov != len(out_df):
        raise ValueError(
            f"FATAL: Covariance dimension ({n_cov}) != data rows ({len(out_df)}). "
            "Row alignment broken — aborting."
        )

    vals = np.array([float(x.strip()) for x in lines[1:] if x.strip()])
    expected = n_cov * n_cov
    if len(vals) != expected:
        raise ValueError(
            f"FATAL: Expected {expected} covariance values, got {len(vals)}."
        )

    cov = vals.reshape((n_cov, n_cov))

    if not np.allclose(cov, cov.T, atol=1e-12):
        print("  WARNING: Covariance matrix not perfectly symmetric, symmetrizing.")
        cov = 0.5 * (cov + cov.T)

    np.savetxt(COV_FILE, cov)
    print(f"Saved {n_cov}x{n_cov} covariance matrix to {COV_FILE}")
    os.remove("pantheon_cov_raw.txt")

    print(f"\nAlignment check: {len(out_df)} data rows, {n_cov}x{n_cov} covariance — MATCHED")
