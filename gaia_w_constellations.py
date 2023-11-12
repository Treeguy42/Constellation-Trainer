from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd

# Load the Gaia and HYG data
gaia_data_path = '/mnt/c/Users/serge/Desktop/Trainers/gaia_stars_raw.csv'
hyg_data_path = '/mnt/c/Users/serge/Desktop/Trainers/hyg_v37.csv'

gaia_data = pd.read_csv(gaia_data_path)
hyg_data = pd.read_csv(hyg_data_path)

# Convert the RA and DEC to SkyCoord objects
gaia_coords = SkyCoord(ra=gaia_data['ra'].values*u.degree, dec=gaia_data['dec'].values*u.degree)
hyg_coords = SkyCoord(ra=hyg_data['ra'].values*u.degree, dec=hyg_data['dec'].values*u.degree)

# Find the closest HYG stars to each Gaia star
idx, d2d, _ = gaia_coords.match_to_catalog_sky(hyg_coords)

# Add the constellation information from HYG to Gaia data
gaia_data['constellation'] = hyg_data.iloc[idx]['con'].values

# Save the new Gaia data with constellation information
output_path = '/mnt/c/Users/serge/Desktop/Trainers/gaia_stars_with_constellations.csv'
gaia_data.to_csv(output_path, index=False)

print(f"Updated Gaia data with constellation information saved to {output_path}")
