import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns

# %%
#start with a linear model (3rd order polynomial)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size']=14

filename = r'ElectronMobility.csv'
#data taken from https://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
df = pd.read_csv(filename)
y = df['Mobility'].dropna()
x = df['Density Ln'].dropna()

fig = plt.figure(1, figsize=[5,5])

colors = sns.cubehelix_palette(5, start=2)

def third_order_poly(x,c,c1,c2,c3):
    return c+c1*x+c2*x**2+c3*x**3

popt,pcov = curve_fit(third_order_poly,x,y)
c=popt[0]
c1=popt[1]
c2=popt[2]
c3=popt[3]
plt.plot(x,y,marker='o', color=colors[2], linestyle='', markersize=11, mfc='white',label='NIST data')
plt.plot(x,third_order_poly(x,*popt),marker='None', linestyle='-',color=colors[4], markersize=11, mfc='white',label=f'{c:.0f}+{c1:.1f}x+{c2:.2f}x$^2$+{c3:.2f}x$^3$')



plt.minorticks_on()
plt.tick_params(direction='in',right=True, top=True, left=True, bottom=True)
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)  
plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
plt.xlim([-3.25,2.25])
plt.ylim([0,1600])
plt.xlabel('Density Ln')
plt.ylabel('Mobiliy $cm^2/(Vs)$') 
plt.legend(bbox_to_anchor=(1.05, 1.0))




plt.savefig('third_order_poly_fit.png', dpi=300,bbox_inches="tight")
# %%
#plot the residual to show bias
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size']=14

filename = r'ElectronMobility.csv'
#data taken from https://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
df = pd.read_csv(filename)
y = df['Mobility'].dropna()
x = df['Density Ln'].dropna()

fig = plt.figure(1, figsize=[5,5])

colors = sns.cubehelix_palette(5, start=2)

def third_order_poly(x,c,c1,c2,c3):
    return c+c1*x+c2*x**2+c3*x**3

popt,pcov = curve_fit(third_order_poly,x,y)
c=popt[0]
c1=popt[1]
c2=popt[2]
c3=popt[3]
z=76
#plt.plot(x,y,marker='o', color=colors[2], linestyle='', markersize=11, mfc='white',label='NIST data')
plt.plot(third_order_poly(x,*popt),y-third_order_poly(x,*popt),marker='s', linestyle='',color=colors[4], markersize=11, mfc='white',label=f'residual')
plt.axhline(0,linestyle='--',color='grey')



plt.minorticks_on()
plt.tick_params(direction='in',right=True, top=True, left=True, bottom=True)
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)  
plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
plt.xlim([0,1600])
plt.ylim([-150,150])
plt.xlabel('fitted value')
plt.ylabel('residual') 
plt.legend()




plt.savefig('third_order_poly_fit_bias.png', dpi=300,bbox_inches="tight")

# %%
#nonlinear fit example

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size']=14

filename = r'ElectronMobility.csv'
#data taken from https://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
df = pd.read_csv(filename)
y = df['Mobility'].dropna()
x = df['Density Ln'].dropna()

fig = plt.figure(1, figsize=[5,5])

colors = sns.cubehelix_palette(5, start=2)

def rational_function(x,c,c1,c2,c3,c4,c5,c6):
    return (c + c1*x + c2*x**2 + c3*x**3) / (1 + c4*x + c5*x**2 + c6*x**3)

popt,pcov = curve_fit(rational_function,x,y)
c=popt[0]
c1=popt[1]
c2=popt[2]
c3=popt[3]
c4=popt[4]
c5=popt[5]
c6=popt[6]
plt.plot(x,y,marker='o', color=colors[2], linestyle='', markersize=11, mfc='white',label='NIST data')
plt.plot(x,rational_function(x,*popt),marker='None', linestyle='-',color=colors[4], markersize=11, mfc='white',label=f'rational function')


plt.minorticks_on()
plt.tick_params(direction='in',right=True, top=True, left=True, bottom=True)
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)  
plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
plt.xlim([-3.25,2.25])
plt.ylim([0,1600])
plt.xlabel('Density Ln')
plt.ylabel('Mobiliy $cm^2/(Vs)$') 
plt.legend(bbox_to_anchor=(1.05, 1.0))




plt.savefig('rational_function_fit.png', dpi=300,bbox_inches="tight")

# %%
#nonlinear residual plot

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.size']=14

filename = r'ElectronMobility.csv'
#data taken from https://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
df = pd.read_csv(filename)
y = df['Mobility'].dropna()
x = df['Density Ln'].dropna()

fig = plt.figure(1, figsize=[5,5])

colors = sns.cubehelix_palette(5, start=2)

def rational_function(x,c,c1,c2,c3,c4,c5,c6):
    return (c + c1*x + c2*x**2 + c3*x**3) / (1 + c4*x + c5*x**2 + c6*x**3)

popt,pcov = curve_fit(rational_function,x,y)
c=popt[0]
c1=popt[1]
c2=popt[2]
c3=popt[3]
c4=popt[4]
c5=popt[5]
c6=popt[6]
#plt.plot(x,y,marker='o', color=colors[2], linestyle='', markersize=11, mfc='white',label='NIST data')
plt.plot(rational_function(x,*popt),y-rational_function(x,*popt),marker='^', linestyle='',color=colors[3], markersize=11, mfc='white',label=f'residual')
plt.axhline(0,linestyle='--',color='grey')

plt.minorticks_on()
plt.tick_params(direction='in',right=True, top=True, left=True, bottom=True)
plt.tick_params(labelbottom=True, labeltop=False, labelright=False, labelleft=True)  
plt.tick_params(direction='in',which='minor', length=5, bottom=True, top=True, left=True, right=True)
plt.tick_params(direction='in',which='major', length=10, bottom=True, top=True, left=True, right=True)
plt.xlim([0,1600])
plt.ylim([-150,150])
plt.xlabel('fitted value')
plt.ylabel('residual') 
plt.legend(bbox_to_anchor=(1.05, 1.0))




plt.savefig('rational_function_residual.png', dpi=300,bbox_inches="tight")