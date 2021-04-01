# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 22:52:24 2021

@author: Cazaban clément
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv('Meteorite_Landings.csv') 
df = df.drop("GeoLocation", axis=1)
df = df.dropna()
df = df[df["mass (g)"]!=0]
df = df.reset_index()
for i in range(len(df)):
    df.year[i]= int(df.year[i][6:10])
df = df.drop(df[df.year > 2016].index)
df = df.drop(df[df.year < 861].index)

#%% QUESTION 1
plt.rc('font', **{
      'size': 30,
      'family': 'sans-serif'})

plt.figure(0, figsize=(15,10))
plt.hist(df['mass (g)'], color = 'royalblue', edgecolor = 'black',
          bins = 20) 
plt.yscale("log")
plt.title("mass distribution of meteorites", y=1.02)
plt.xlabel("mass (g)")
plt.ylabel("Nr of meteorites (log scale)")

plt.figure(1,figsize=(15,10))
df_less50kg = df[df["mass (g)"]<=50000]
plt.hist(df_less50kg['mass (g)'], color = 'royalblue', edgecolor = 'black',
         bins = 20) 
plt.yscale("log") 
plt.title("mass distribution of meteorites <= 50 Kg", y=1.02)
plt.xlabel("mass (g)")
plt.ylabel("Nr of meteorites (log scale)")

#%% QUESTION 2
from collections import Counter
count = Counter(df.year)
tab = np.array(sorted(list(count.items())))
tab = tab[tab[:,0]>1973]
cumsum = np.cumsum(tab[:,1], axis=0)

plt.figure(0, figsize=(15,10))
plt.plot(tab[:,0],cumsum,'s', color = 'royalblue',marker='o')
plt.title("number of meteorites as a function of time (by year)", y=1.02)
plt.xlabel("Years")
plt.ylabel("Nr of meteorites") 

from scipy import stats
res = stats.linregress(tab[:,0], cumsum)
print(f"R-squared: {res.rvalue**2:.6f}")
print(f"Regression formula: {res.slope:.2f}*x + {res.intercept:.2f}")
plt.plot(tab[:,0], res.intercept + res.slope*tab[:,0], 'r', label='fitted line')
plt.legend()

#%% QUESTION 3 
import geopandas
from shapely.geometry import Point, MultiPolygon
gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.reclong, df.reclat))
points = []

#####################################################################################################
df = df.reset_index()   # PLEASE COMMENT THIS LINE IF YOU ARE'NT RUNNING THIS CELL FOR THE FIRST TIME
#####################################################################################################

for i in range(len(df)):
    points.append(Point(df.reclong[i],df.reclat[i]))
    
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
oman = world[world.name == "Oman"]
poly = oman.geometry
mpoly = MultiPolygon(poly.item())

in_ = []
for i in range(len(points)):
    if points[i].within(mpoly):
        in_.append(True)
    else:
        in_.append(False)
in_ = np.asarray(in_)
gdf = gdf[in_]

plt.rc('font', **{
      'size': 45,
      'family': 'sans-serif'})
ax = world[world.name == 'Oman'].plot(
    color='white', edgecolor='black', figsize=(18,18))
gdf.plot(ax=ax, color='red')
plt.title("Oman meteorites",size=60)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

#%% QUESTION 4
plt.figure(3, figsize=(20,15))
plt.hist(gdf['reclat'], color = 'royalblue', edgecolor = 'black',
          bins = 15)  
plt.title("Latitude distribution of oman's meteorites",size=50, y=1.02)
plt.xlabel("latitude ")
plt.ylabel("Nr of meteorites")


plt.figure(4, figsize=(20,15))
plt.hist(gdf['reclong'], color = 'royalblue', edgecolor = 'black',
          bins = 10)  
plt.title("Longitude distribution of oman's meteorites",size=50, y=1.02)
plt.xlabel("Longitude")
plt.ylabel("Nr of meteorites")

density = np.vstack((gdf['reclong'],gdf['reclat'])).T

fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(20, 15))
hb = ax.hexbin(gdf['reclong'], gdf['reclat'], gridsize=50, bins='log', cmap='inferno')
# ax.axis([xmin, xmax, ymin, ymax])
ax.set_title("2D distribution")
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(Nr of meteorites)')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
from sklearn.neighbors import KernelDensity

def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape), kde_skl

xx, yy, zz, estimator = kde2D(gdf['reclong'],gdf['reclat'], 1.0) 
plt.figure(5, figsize=(20,15))
plt.pcolormesh(xx, yy, zz,cmap='inferno')
plt.colorbar()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Probablity density plot (Gaussian Kernel)",size=50, y=1.02)
# %% QUESTION 5
#Make a prediction : 
start_i = 18.9644 - 0.9043
end_i = 18.9644 + 0.9043
start_j = 53.9555 - 0.8923
end_j = 53.9555 + 0.8923
step_i = 100
step_j = 100
step = step_i*step_j

###### APROXIMATION OF THE AREA WITH A SQUARE ######
proba_ligne = []
for y in np.linspace(start_j,end_j,step_j):
    ligne_y = np.vstack((np.linspace(start_i,end_i,step_i),np.ones(step_i)*y)).T
    kd_vals = np.exp(estimator.score_samples(ligne_y))
    proba = np.sum(kd_vals * ((end_i-start_i)/(step_i-1))) 
    proba_ligne.append(proba)
proba_ligne = np.asarray(proba_ligne)
probability = np.sum(proba_ligne * ((end_j-start_j)/(step_j-1)))
pred = np.asarray([[19,55],[19.001,55.001]])
predictions = np.exp(estimator.score_samples(pred))
print(f"The probability (for a square) is : {probability}")

###### APROXIMATION OF THE AREA WITH A CIRCLE ######
proba_cercle = []
points =[]
dr = (0.9/step_i)
proba_points =[]
for r in np.linspace(0,0.9,step_j):
    for teta in np.linspace(0,2*np.pi,step_i):
        point = np.vstack((18.9644+0.9043*np.sin(teta),53.9555+0.8923*np.cos(teta))).T
        proba_points.append(np.exp(estimator.score_samples(point)))
    dteta = r*np.tan((np.pi/step_j))
    proba_cercle.append(np.sum(np.asarray(proba_points) * dteta))
proba_cercle = np.asarray(proba_ligne)
probability = np.sum(proba_cercle * dr)
print(f"The probability (for a circle) is : {probability}")

