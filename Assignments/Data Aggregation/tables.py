import pandas as pd
import numpy as np

# pandas settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# get dataset
dataset = pd.read_csv("C:/Users/ldmag/Downloads/ss13hil.csv")

# Table 1. HINCP grouped by HHT
selected = dataset[['HINCP', 'HHT']]
hht_dict = {1.0: 'Married couple household', 2.0:'Other family household:Male householder, no wife present',
            3.0:'Other family household:Female householder, no husband present',
            4.0:'Nonfamily household:Male householder: Living alone',
            5.0:'Nonfamily household:Male householder: Not living alone',
            6.0:'Nonfamily household:Female householder: Living alone',
            7.0: 'Nonfamily household:Female householder: Not living alone'}

selected['HHT'] = selected['HHT'].map(hht_dict).copy()
query = selected['HINCP'].groupby(selected['HHT']).describe()
query = query.drop(columns=['25%', '50%', '75%'])

# Table 2
selected2 = dataset[['WGTP', 'ACCESS','HHL']]
access_dict = {1.0: 'Yes w/ subsrc. ', 2.0: 'Yes w/o subsrc', 3.0 : 'No'}
hhl_dict = {1.0: 'English only', 2.0: 'Spanish', 3.0: 'Other Indo-European languages',
            4.0: 'Asian and Pacific Island languages', 5.0: 'Other'}

selected2['ACCESS'] = selected2['ACCESS'].map(access_dict).copy()
selected2['HHL'] = selected2['HHL'].map(hhl_dict).copy()
selected2 = selected2.dropna() # all rows with nans dropped 
pd.crosstab(index= selected2['HHL'], columns = selected2['ACCESS'], 
            values= selected2['WGTP'], aggfunc='sum', margins=True, normalize='all')


# Table 3 - HINCP Quantile analysis
selected3 = dataset[['HINCP', 'WGTP']].dropna() # drop nan values
low = selected3.HINCP.quantile(q=0), selected3.HINCP.quantile(q=1/3)
selected3.HINCP.quantile(q=2/3)
selected3.HINCP.quantile(q=1)

# found partial quantile solution