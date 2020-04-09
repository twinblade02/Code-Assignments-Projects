import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer

# get data
dataset = pd.read_excel('C:/Users/ldmag/Downloads/HBAT.xls')
selected = dataset[['x6','x7','x8','x9','x10','x11','x12','x13','x14','x16','x18']]

# Getting Eigenvalues 
fa = FactorAnalyzer(rotation = None)
fa.fit(selected)
ev, v = fa.get_eigenvalues()

# Scree plot
plt.scatter(range(1,selected.shape[1]+1),ev)
plt.plot(range(1,selected.shape[1]+1),ev)
plt.axhline(y=1, color='r', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

# Factor pattern - unrotated
fa_no_rotation = FactorAnalyzer(n_factors = 4, rotation = None)
fa_no_rotation.fit(selected)
loadings_no_rotation = fa_no_rotation.loadings_
df_Loadings = pd.DataFrame(data= loadings_no_rotation, index= ['x6','x7','x8','x9','x10','x11','x12','x13','x14','x16','x18'],
             columns = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4'])

## Factor pattern - with rotation
fa_promax = FactorAnalyzer(n_factors = 4, rotation = 'promax')
fa_promax.fit(selected)
loadings_promax = fa_promax.loadings_
df_promax_loadings = pd.DataFrame(data= loadings_promax, index= ['x6','x7','x8','x9','x10','x11','x12','x13','x14','x16','x18'],
             columns = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4'])

## Factor pattern - varimax rotation - seems to be the best fit
fa_varimax = FactorAnalyzer(n_factors = 4, rotation = 'varimax')
fa_varimax.fit(selected)
loadings_varimax = fa_varimax.loadings_
df_varimax_loadings = pd.DataFrame(data= loadings_varimax, index= ['x6','x7','x8','x9','x10','x11','x12','x13','x14','x16','x18'],
             columns = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4'])

## Significant results - Promax
df_promax_loadings[(df_promax_loadings[['Factor 1','Factor 2','Factor 3','Factor 4']] >= 0.5)]
df_promax_loadings[(df_promax_loadings[['Factor 1','Factor 2','Factor 3','Factor 4']] <= -0.5)]

## Significant results - Varimax
df_varimax_loadings[(df_varimax_loadings[['Factor 1','Factor 2','Factor 3','Factor 4']] >= 0.5)]
df_varimax_loadings[(df_varimax_loadings[['Factor 1','Factor 2','Factor 3','Factor 4']] <= -0.5)]