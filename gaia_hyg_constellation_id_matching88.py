import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from sklearn.neighbors import KDTree

def load_data(file_path):
    return pd.read_csv(file_path)

def build_kdtree(hyg_data):
    # Clean the data to ensure there are no non-numeric values
    hyg_data = hyg_data.dropna(subset=['ra', 'dec'])
    hyg_data = hyg_data[(hyg_data['ra'].apply(lambda x: isinstance(x, (int, float)))) & (hyg_data['dec'].apply(lambda x: isinstance(x, (int, float))))]

    # Convert to radians for the k-d tree
    coords = SkyCoord(ra=hyg_data['ra'].values*u.degree, dec=hyg_data['dec'].values*u.degree)
    tree = KDTree(list(zip(coords.ra.rad, coords.dec.rad)))
    return tree

def find_closest_hyg_star(gaia_coord, kdtree, hyg_data):
    dist, idx = kdtree.query([[gaia_coord.ra.rad, gaia_coord.dec.rad]], k=1)
    if idx[0][0] < len(hyg_data):
        return hyg_data.iloc[idx[0][0]]
    return None

def cross_reference_constellations(gaia_data, kdtree, hyg_data, constellations_of_interest):
    constellation_matches = []
    for index, gaia_star in gaia_data.iterrows():
        gaia_coord = SkyCoord(ra=gaia_star['ra']*u.degree, dec=gaia_star['dec']*u.degree)
        closest_hyg_star = find_closest_hyg_star(gaia_coord, kdtree, hyg_data)
        if closest_hyg_star is not None and closest_hyg_star['con'] in constellations_of_interest:
            constellation_matches.append({
                'gaia_source_id': gaia_star['source_id'],
                'hyg_id': closest_hyg_star['id'],
                'constellation': closest_hyg_star['con']
            })
    return pd.DataFrame(constellation_matches)

# Define the constellations of interest
constellations_of_interest = [
    'And', 'Ant', 'Aps', 'Aql', 'Aqr', 'Ara', 'Ari', 'Aur', 'Boo', 'Cae', 'Cam', 'Cnc', 'CVn',
    'CMa', 'CMi', 'Cap', 'Car', 'Cas', 'Cen', 'Cep', 'Cet', 'Cha', 'Cir', 'Col', 'Com', 'CrA',
    'CrB', 'Crv', 'Crt', 'Cru', 'Cyg', 'Del', 'Dor', 'Dra', 'Equ', 'Eri', 'For', 'Gem', 'Gru',
    'Her', 'Hor', 'Hya', 'Hyi', 'Ind', 'Lac', 'Leo', 'LMi', 'Lep', 'Lib', 'Lup', 'Lyn', 'Lyr',
    'Men', 'Mic', 'Mon', 'Mus', 'Nor', 'Oct', 'Oph', 'Ori', 'Pav', 'Peg', 'Per', 'Phe', 'Pic',
    'Psc', 'PsA', 'Pup', 'Pyx', 'Ret', 'Sge', 'Sgr', 'Sco', 'Scl', 'Sct', 'Ser', 'Sex', 'Tau',
    'Tel', 'Tri', 'TrA', 'Tuc', 'UMa', 'UMi', 'Vel', 'Vir', 'Vol', 'Vul'
]

# Define file paths
gaia_raw_path = '/mnt/c/Users/serge/Desktop/Trainers/gaia_stars_raw_new.csv'
hyg_data_path = '/mnt/c/Users/serge/Desktop/Trainers/hyg_v37.csv'

# Load the datasets
gaia_data = load_data(gaia_raw_path)
hyg_data = load_data(hyg_data_path)

# Build the k-d tree from the HYG data
kdtree = build_kdtree(hyg_data)

# Perform the cross-referencing
constellation_matches = cross_reference_constellations(gaia_data, kdtree, hyg_data, constellations_of_interest)

# Save the matched constellation data to a CSV file
constellation_matches.to_csv('/mnt/c/Users/serge/Desktop/Trainers/matched_constellations_fast.csv', index=False)

print(f"Matched constellation data saved to '/mnt/c/Users/serge/Desktop/Trainers/matched_constellations_fast.csv'")
