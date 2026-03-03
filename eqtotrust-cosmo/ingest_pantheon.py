# ingest_pantheon.py
# Run this once to pull the raw data from the official repo and format it.

import pandas as pd
import urllib.request
import os

url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2B_Magnitude_Format/Pantheon%2B_Stats.txt"

print("Downloading official Pantheon+ data...")
urllib.request.urlretrieve(url, "pantheon_raw.txt")

# The official file is space-separated
print("Processing data...")
df = pd.read_csv("pantheon_raw.txt", delim_whitespace=True)

# Extract redshift (zHD), distance modulus (m_b_corr), and error (m_b_corr_err)
# Note: In Pantheon terminology, m_b_corr acts as the distance modulus mu
out_df = pd.DataFrame({
    'z': df['zHD'],
    'mu': df['m_b_corr'],
    'sigma_mu': df['m_b_corr_err']
})

out_df.to_csv("pantheon_plus.csv", index=False)
print("Saved clean data to pantheon_plus.csv")
os.remove("pantheon_raw.txt")
