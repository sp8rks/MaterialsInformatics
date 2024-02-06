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

df = pd.DataFrame(columns=('pretty_formula', 'band_gap',
                           "density", 'formation_energy_per_atom', 'volume'))

# grab some props for stable oxides
criteria = {'e_above_hull': {'$lte': 0.02},'elements':{'$all':['O']}}
# criteria2 = {'e_above_hull': {'$lte': 0.02},'elements':{'$all':['O']},
#              'band_gap':{'$ne':0}}

props = ['pretty_formula', 'band_gap', "density",
         'formation_energy_per_atom', 'volume']
entries = mpr.query(criteria=criteria, properties=props)

i = 0
for entry in entries:
    df.loc[i] = [entry['pretty_formula'], entry['band_gap'], entry['density'],
                 entry['formation_energy_per_atom'], entry['volume']]
    i += 1


# %% 
#cell for making some nice bootstrap examples
cwd = os.getcwd()
cifs_folder = os.path.join(cwd, "6_cifs")
files = os.listdir(cifs_folder)


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


df = pd.DataFrame(entries)

#drop metals
df['metal?'] = df['band_gap'] == 0
df = df.drop(['band_gap'], axis=1)

df2 = df.drop(df.loc[df['metal?']==True].index,axis=0)

# create first tree with main node density, next node formation energy
# create 2nd tree with main node volume, next node

# show need for ensemble by doing bootstrap of following

# bootstrap A 2,0,2,4,5,5 using features density, volume
dfA = df2.iloc[[2, 0, 2, 4, 5, 5], :]
# bootstrap B 2,1,3,1,4,4 using features volume, formation energy
dfB = df.iloc[[2, 1, 3, 1, 4, 4], :]
# bootstrap C 4,1,3,0,0,2 using features density, volume
dfC = df.iloc[[4, 1, 3, 0, 0, 2], :]
# bootstrap D 3,3,2,5,1,2 using features formation energy, density
dfD = df.iloc[[3, 3, 2, 5, 1, 2], :]

# how many features should be used? we want sqrt of total # features

# %%
#Let's build a simple random forest without any composition or structure features
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

RNG_SEED = 42
np.random.seed(seed=RNG_SEED)

X = df[['band_gap','formation_energy_per_atom','volume']]
y = df['density']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RNG_SEED)
rf = RandomForestRegressor(max_depth=2, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse_val = mean_squared_error(y_test, y_pred, squared=False)


# %%
#now let's add CBFV
from CBFV import composition

rename_dict = {'density': 'target', 'pretty_formula':'formula'}
df = df.rename(columns=rename_dict)


RNG_SEED = 42
np.random.seed(seed=RNG_SEED)

X = df[['formula','band_gap','formation_energy_per_atom','volume']]
y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RNG_SEED)

X_train, y_train, formulae_train, skipped_train = composition.generate_features(df, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)
X_test, y_test, formulae_train, skipped_train = composition.generate_features(df, elem_prop='oliynyk', drop_duplicates=False, extend_features=True, sum_feat=True)


#technically we should scale and normalize our data here... but lets skip it for now

#we should perform hyperparameter tuning via cross-validation on a validation dataset too...

rf = RandomForestRegressor(max_depth=4, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse_val = mean_squared_error(y_test, y_pred, squared=False)


# %%
#we can visualize the trees!
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
feature_list = list(X_train.columns)
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')








