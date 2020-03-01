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

# Table 2, map values after crosstab
selected2 = dataset[['WGTP', 'ACCESS','HHL']]
access_dict = {1.0: 'Yes w/ Subsrc.', 2.0: 'Yes wo/ Subsrc', 3.0 : 'No'}
hhl_dict = {1.0: 'English only', 2.0: 'Spanish', 3.0: 'Other Indo-European languages',
            4.0: 'Asian and Pacific Island languages', 5.0: 'Other'}

selected2['ACCESS'] = selected2['ACCESS'].map(access_dict).copy()
selected2['HHL'] = selected2['HHL'].map(hhl_dict).copy()
selected2 = selected2.dropna() # all rows with nans dropped 
query2 = pd.crosstab(index= selected2['HHL'], columns = selected2['ACCESS'], 
            values= selected2['WGTP'], aggfunc='sum', margins=True, normalize='all')
query2 = query2.applymap(lambda x:"{0:.2f}%".format(x*100))
query2 = query2[['Yes w/ Subsrc.', 'Yes wo/ Subsrc', 'No', 'All']]

# Table 3 - HINCP Quantile analysis
selected3 = dataset[['HINCP', 'WGTP']].dropna() # drop nan values
selected3['HINCP_'] = pd.qcut(selected3.HINCP, 3, labels=["low", "medium", "high"])
query3 = selected3['HINCP'].groupby(selected3.HINCP_).describe()
query3 = query3.drop(columns=['std', '25%','50%','75%','count'])
query3['household_count'] = selected3.groupby(selected3.HINCP_).agg({'WGTP': sum})

# display tables / format tables
print("DATA-51100, Spring 2020" )
print("Lionel Dsilva")
print("Programming Assignment #7 \n")

print("*** Table 1 - Descriptive Statistics of HINCP, grouped by HHT ***")
print(query)
print("\n")
print("*** Table 2 - HHL vs. ACCESS - Frequency Table ***")
print("\t \t \t \t \t sum")
print("\t \t \t \t \t WGTP")
print(query2)
print("\n")
print("*** Table 3 - Quantile Analysis of HINCP - Household Income (past 12 months) ***")
print(query3)