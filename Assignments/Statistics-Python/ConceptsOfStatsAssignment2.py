''' Code was compiled in Spyder IDE.
***WARNING: Do not run code in its entirety or the plots will be overridden; recommend running in the following order:
    import statements, read data, skewness, normal probability : run ax2, ax3, ax4 and ax5 one after another, scatter plots: ax6 and ax7 one by one
    Correlation, Boxplot: ax8 then ax9 one by one. ***
'''
# Import 
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

# read data
dataset = pd.read_excel('C:/Users/ldmag/Downloads/HBAT(3).xls')

sns.set(color_codes=True)
X = dataset[['x6','x7','x16','x17']] 
ax1 = sns.distplot(X)

# skewness and kurtosis tests, correlation: Pearson, normality test
from scipy.stats import kurtosis, skew
from scipy.stats import normaltest
k = kurtosis(X)
s = skew(X)
corr = X.corr()

# test for normality 
alpha = 0.01
k2, p = normaltest(X)
for num in p:
    if num < alpha:
        print("Reject null hypothesis")
    else:
        print("Unable to reject null hypothesis")
        
# normal probability plots 
from scipy.stats import norm
from scipy.stats import probplot
x6 = dataset['x6']
x7 = dataset['x7']
x16 = dataset['x16']
x17 = dataset['x17']

ax2 = probplot(x6, dist = "norm", plot = plt)
ax3 = probplot(x7, dist = "norm", plot = plt)
ax4 = probplot(x16, dist = "norm", plot = plt)
ax5 = probplot(x17, dist = "norm", plot = plt)

# scatter plots and ellipse: Ellipse function for matplotlib too complex to implement 
data1 = dataset[['x7','x19']]
ax6 = sns.scatterplot(data=data1)
data2 = dataset[['x6','x19']]
ax7 = sns.scatterplot(data=data2)

# Correlation
data_corr = dataset[['x6','x7','x8','x12']]
corr2 = data_corr.corr()

#Boxplot
data3 = dataset[['x6','x1']]
ax8 = sns.boxplot(data=data3)
data4 = dataset[['x7','x1']]
ax9 = sns.boxplot(data=data4)