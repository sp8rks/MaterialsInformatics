import pandas as pd
from pymatgen.ext.matproj import MPRester
import os

filename = r'C:\Users\taylo\Google Drive\teaching\5050 Materials Informatics\apikey.txt'


def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)


Sparks_API = get_file_contents(filename)

# %%

mpr = MPRester(Sparks_API)
cwd = os.getcwd()
cifs_folder = os.path.join(cwd, "6_cifs")
files = os.listdir(cifs_folder)

df = pd.DataFrame(columns=('pretty_formula', 'band_gap',
                           "density", 'formation_energy_per_atom', 'volume'))

mpid_list = []
for f in files:
    file_to_read = os.path.join(cifs_folder, f)
    data1 = mpr.find_structure(file_to_read)
    print(data1)
    mpid_list.append(data1[0])


# grab some props for those mpids
criteria = {'material_id': {'$in': mpid_list}}


props = ['pretty_formula', 'band_gap', "density",
         'formation_energy_per_atom', 'volume']
entries = mpr.query(criteria=criteria, properties=props)

i = 0
for entry in entries:
    df.loc[i] = [entry['pretty_formula'], entry['band_gap'], entry['density'],
                 entry['formation_energy_per_atom'], entry['volume']]
    i += 1

df['metal?'] = df['band_gap'] == 0
df = df.drop(['band_gap'], axis=1)


# create first tree with main node density, next node formation energy
# create 2nd tree with main node volume, next node

# show need for ensemble by doing bootstrap of following

# bootstrap A 2,0,2,4,5,5 using features density, volume
dfA = df.iloc[[2, 0, 2, 4, 5, 5], :]
# bootstrap B 2,1,3,1,4,4 using features volume, formation energy
dfB = df.iloc[[2, 1, 3, 1, 4, 4], :]
# bootstrap C 4,1,3,0,0,2 using features density, volume
dfC = df.iloc[[4, 1, 3, 0, 0, 2], :]
# bootstrap D 3,3,2,5,1,2 using features formation energy, density
dfD = df.iloc[[3, 3, 2, 5, 1, 2], :]

# how many features should be used? we want sqrt of total # features
