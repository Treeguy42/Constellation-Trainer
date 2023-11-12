import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import DBSCAN

# Constants
MAX_MAGNITUDE_VISIBLE = 6.8  # Adjusted visibility threshold for the naked eye
EPSILON_RAD = 0.1 * np.pi / 180  # DBSCAN radius in radians (0.1 degrees)

# Load and clean data
def load_and_clean_data(file_path, ra_column='ra', dec_column='dec'):
    data = pd.read_csv(file_path)
    # Clean and parse RA and Dec
    data = data.dropna(subset=[ra_column, dec_column])
    data[ra_column] = pd.to_numeric(data[ra_column], errors='coerce')
    data[dec_column] = pd.to_numeric(data[dec_column], errors='coerce')
    data.dropna(subset=[ra_column, dec_column], inplace=True)
    return data

# Filter visible stars
def filter_visible_stars(star_data, mag_column='phot_g_mean_mag'):
    return star_data[star_data[mag_column] <= MAX_MAGNITUDE_VISIBLE]

# Apply DBSCAN clustering
def apply_dbscan(star_data, ra_column='ra', dec_column='dec', epsilon_rad=EPSILON_RAD):
    # Additional cleaning to ensure 'ra' and 'dec' are numeric and there are no NaN values
    star_data[ra_column] = pd.to_numeric(star_data[ra_column], errors='coerce')
    star_data[dec_column] = pd.to_numeric(star_data[dec_column], errors='coerce')
    star_data.dropna(subset=[ra_column, dec_column], inplace=True)

    # Convert to radians for clustering
    star_data_rad = star_data.copy()
    star_data_rad[ra_column] = np.deg2rad(star_data_rad[ra_column])
    star_data_rad[dec_column] = np.deg2rad(star_data_rad[dec_column])
    
    # Stack the coordinates for DBSCAN
    coords = np.stack((star_data_rad[ra_column], star_data_rad[dec_column]), axis=1)
    
    # Run DBSCAN
    db = DBSCAN(eps=epsilon_rad, min_samples=1, metric='haversine').fit(coords)
    
    # Assign cluster labels
    star_data.loc[:, 'cluster_label'] = db.labels_
    return star_data

# Load datasets
gaia_data = load_and_clean_data('/mnt/c/Users/serge/Desktop/Trainers/gaia_stars_with_constellations.csv')
hyg_data = load_and_clean_data('/mnt/c/Users/serge/Desktop/Trainers/hyg_v37.csv')
matched_data = pd.read_csv('/mnt/c/Users/serge/Desktop/Trainers/matched_constellations_fast.csv')

# Make sure the 'gaia_source_id' and 'source_id' are both strings
matched_data['gaia_source_id'] = matched_data['gaia_source_id'].astype(str)
gaia_data['source_id'] = gaia_data['source_id'].astype(str)

# Filter visible stars
visible_gaia = filter_visible_stars(gaia_data, 'phot_g_mean_mag')
visible_hyg = filter_visible_stars(hyg_data, 'mag')

# Merge matched data with visible stars from both Gaia and HYG datasets
matched_gaia = pd.merge(matched_data, visible_gaia, left_on='gaia_source_id', right_on='source_id', how='left')
matched_hyg = pd.merge(matched_data, visible_hyg, left_on='hyg_id', right_on='id', how='left')

# Combine matched data into a single DataFrame
combined_matched_data = pd.concat([matched_gaia, matched_hyg])

# Apply DBSCAN clustering to combined matched data
combined_matched_clusters = apply_dbscan(combined_matched_data, 'ra', 'dec')

# Save the visible stars per constellation to CSV
combined_matched_clusters.to_csv('/mnt/c/Users/serge/Desktop/Trainers/constellations_visible.csv', index=False)

# Save all visible stars to CSV (both Gaia and HYG)
combined_visible_stars = pd.concat([visible_gaia, visible_hyg], ignore_index=True)
combined_visible_stars.to_csv('/mnt/c/Users/serge/Desktop/Trainers/stars_all_visible.csv', index=False)

# Save clusters of stars around constellations to CSV
clusters_csv = '/mnt/c/Users/serge/Desktop/Trainers/visible_constellation_clusters.csv'
with open(clusters_csv, 'w') as file:
    for constellation, group_df in combined_matched_clusters.groupby('constellation'):
        group_df.to_csv(file, index=False)
        file.write("\n")

print("Data has been processed and saved.")
