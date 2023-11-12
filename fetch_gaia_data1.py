import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Fetch data from Gaia Archive
def fetch_gaia_data(query, output_format="csv"):
    GAIA_ARCHIVE_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
    
    response = requests.post(GAIA_ARCHIVE_URL, data={
        "request": "doQuery",
        "lang": "adql",
        "format": output_format,
        "query": query
    })
    response.raise_for_status()
    return response.content

# Preprocess the data
def preprocess_data(file_path, output_file_path):
    df = pd.read_csv(file_path)
    print(f"Initial data shape: {df.shape}")

    # Drop rows with any missing values
    df.dropna(inplace=True)
    print(f"Data shape after dropping rows with missing values: {df.shape}")

    # Replace infinite values with NaN, then drop those rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Data shape after dropping infinite values: {df.shape}")

    features_to_scale = ['phot_g_mean_mag', 'parallax', 'bp_rp', 'radial_velocity']
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    df.to_csv(output_file_path, index=False)
    print(f"Data preprocessed and saved to '{output_file_path}'. Final data shape: {df.shape}")

# Define the query
query = """
SELECT 
    source_id, 
    ra, 
    dec, 
    pmra, 
    pmdec, 
    parallax, 
    phot_g_mean_mag, 
    phot_bp_mean_mag, 
    phot_rp_mean_mag, 
    bp_rp, 
    radial_velocity
FROM 
    gaiadr2.gaia_source 
WHERE 
    phot_g_mean_mag < 15 AND 
    parallax_over_error > 10
"""

# Set the paths
windows_desktop_path = '/mnt/c/Users/serge/Desktop/Trainers/'

# Fetch and preprocess the data
try:
    # Fetch the data and save it with a new name to avoid overwriting existing files
    data = fetch_gaia_data(query)
    new_raw_file_path = windows_desktop_path + 'gaia_stars_raw_new.csv'
    with open(new_raw_file_path, 'wb') as file:
        file.write(data)
    print("Data fetched and saved to 'gaia_stars_raw_new.csv'.")

    # Preprocess the data with verbose output and save with a new distinct name
    new_preprocessed_file_path = windows_desktop_path + 'gaia_stars_preprocessed_verbose_new.csv'
    preprocess_data(new_raw_file_path, new_preprocessed_file_path)

except Exception as e:
    print(f"An error occurred: {e}")
