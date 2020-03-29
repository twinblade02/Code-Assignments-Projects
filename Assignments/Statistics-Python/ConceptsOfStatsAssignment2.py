import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

# read data
dataset = pd.read_excel('C:/Users/ldmag/Downloads/HBAT(3).xls')

sns.set(color_codes=True)
X = dataset[['x6','x7','x16','x17']]

sns.distplot(X)

# skewness and kurtosis tests, correlation: Pearson, normality test
from scipy.stats import kurtosis, skew
from scipy.stats import normaltest
k = kurtosis(X)
s = skew(X)
corr = X.corr()
# test for normality 
alpha = 0.05
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

probplot(x6, dist = "norm", plot = plt)
probplot(x7, dist = "norm", plot = plt)
probplot(x16, dist = "norm", plot = plt)
probplot(x17, dist = "norm", plot = plt)

# scatter plots and ellipse 
