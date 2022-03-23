import pandas as pd
from pymatgen.ext.matproj import MPRester
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import matplotlib.gridspec as gridspec


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

# grab some props for stable insulators
criteria = {'e_above_hull': {'$lte': 0.02},'band_gap':{'$gt':0}}

props = ['pretty_formula', 'band_gap', "density",
         'formation_energy_per_atom', 'volume']
entries = mpr.query(criteria=criteria, properties=props)

df_insulators = pd.DataFrame(entries)

print(df_insulators['density'].mean())
print(df_insulators['density'].std())

# grab some props for stable metals
criteria = {'e_above_hull': {'$lte': 0.02},'band_gap':{'$eq':0}}

props = ['pretty_formula', 'band_gap', "density",
         'formation_energy_per_atom', 'volume']
entries = mpr.query(criteria=criteria, properties=props)

df_metals = pd.DataFrame(entries)

print(df_metals['density'].mean())
print(df_metals['density'].std())

#https://homepage.divms.uiowa.edu/~mbognar/applets/normal.html

# %%

#let's plot our gaussians
fig = plt.figure(1, figsize=(5,5))
gs = gridspec.GridSpec(3,1)
gs.update(wspace=0.2, hspace=0.25)

#Generate first panel
xtr_subsplot= fig.add_subplot(gs[0:1,0:1])
x=np.arange(0,20,0.1)
#y=scipy.stats.norm(4.4, 1.72).cdf(x) #cumulative distribution function
y_metals=scipy.stats.norm(df_metals['density'].mean(), df_metals['density'].std()).pdf(x) #probability distribution function
y_ins=scipy.stats.norm(df_insulators['density'].mean(), df_insulators['density'].std()).pdf(x) #probability distribution function
plt.plot(x,y_metals)
plt.plot(x,y_ins)
plt.ylabel(r'$\rho\,g/cc$')

#Generate second panel
xtr_subsplot= fig.add_subplot(gs[1:2,0:1])
x=np.arange(-1000,5000,0.1)
#y=scipy.stats.norm(4.4, 1.72).cdf(x) #cumulative distribution function
y_metals=scipy.stats.norm(df_metals['volume'].mean(), df_metals['volume'].std()).pdf(x) #probability distribution function
y_ins=scipy.stats.norm(df_insulators['volume'].mean(), df_insulators['volume'].std()).pdf(x) #probability distribution function
plt.plot(x,y_metals)
plt.plot(x,y_ins)
plt.ylabel('$V$ Angstroms')

#Generate third panel
xtr_subsplot= fig.add_subplot(gs[2:3,0:1])
x=np.arange(-4,2,0.1)
#y=scipy.stats.norm(4.4, 1.72).cdf(x) #cumulative distribution function
y_metals=scipy.stats.norm(df_metals['formation_energy_per_atom'].mean(), df_metals['formation_energy_per_atom'].std()).pdf(x) #probability distribution function
y_ins=scipy.stats.norm(df_insulators['formation_energy_per_atom'].mean(), df_insulators['formation_energy_per_atom'].std()).pdf(x) #probability distribution function
plt.plot(x,y_metals,label='metal')
plt.plot(x,y_ins,label='insulator')
plt.ylabel('$\Delta H/atom$ eV')

plt.legend()


# %%
#introduce a new mystery material with following values
density = 4
volume = 800
formation_energy = -2

#we do classification by adding up probabilities for each
#initial guess based on proportion of metals v insulators
prior_metals = df_metals['density'].count()/(df_insulators['density'].count()+df_metals['density'].count())
prior_insulators = 1-prior_metals

#now probability based on density
density_metals = scipy.stats.norm(df_metals['density'].mean(), df_metals['density'].std()).pdf(density)
density_insulators = scipy.stats.norm(df_insulators['density'].mean(), df_insulators['density'].std()).pdf(density)

#now probability based on volume
volume_metals = scipy.stats.norm(df_metals['volume'].mean(), df_metals['volume'].std()).pdf(volume)
volume_insulators = scipy.stats.norm(df_insulators['volume'].mean(), df_insulators['volume'].std()).pdf(volume)

#now probability based on formation energy
energy_metals = scipy.stats.norm(df_metals['formation_energy_per_atom'].mean(), df_metals['formation_energy_per_atom'].std()).pdf(formation_energy)
energy_insulators = scipy.stats.norm(df_insulators['formation_energy_per_atom'].mean(), df_insulators['formation_energy_per_atom'].std()).pdf(formation_energy)

#now we add up the log of these probabilities and compare
odds_of_metal = np.log(prior_metals)+np.log(density_metals)+np.log(volume_metals)+np.log(energy_metals)
odds_of_insulator = np.log(prior_insulators)+np.log(density_insulators)+np.log(volume_insulators)+np.log(energy_insulators)

if odds_of_metal > odd_of_insulator:
    print('new material is probably a metal!')
else:
    print('new material is an insulator!')