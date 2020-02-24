# import statements
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# get data
dataset = pd.read_csv("C:/Users/ldmag/Downloads/ss13hil.csv", na_values = '')
fig = plt.figure(dpi = 200, figsize=(15,15))

# Pie plot and data, changed nan values to 1
def pie_chart(ax):
    HHL_df = dataset[['HHL']]
    HHL_df = HHL_df.assign(HHL = HHL_df['HHL'].fillna(1))
    hhl_labels = ['English Only', 'Spanish', 'Other Indo-European', 'Asian and Pacific Island language', 'Other']
    hhl = HHL_df.HHL.value_counts()
    ax.axis('equal')
    ax.pie(hhl, startangle=240)
    ax.legend(hhl_labels, loc = 'upper left', prop={'size': 10})
    ax.set_title('Household Languages')


# HINCP with KDE plot superimposed
def histogram(ax):
    HINCP_df = dataset['HINCP']
    HINCP_df.plot(kind='kde', color='k',ls='dashed')
    logspace = np.logspace(1,7,num=100)
    ax.hist(HINCP_df,bins=logspace,facecolor='g',alpha=0.5,histtype='bar',density=True)
    ax.set_title('Distribution of Household Income',fontsize=10)
    ax.set_xlabel('Household Income($)- Log Scaled',fontsize=10)
    ax.set_ylabel('Density',fontsize=10)
    ax.set_xscale("log")
    ax.set_axisbelow(True)
    ax.grid(False)


# VEH, bar chart
def bar_plot(ax):
    Household_Veh = dataset[['VEH', 'WGTP']]
    wgtp = Household_Veh.groupby('VEH')['WGTP'].sum() / 1000
    ax.bar(wgtp.index, wgtp, color='red')
    ax.set_xlabel('# of vehicles')
    ax.set_ylabel('Thousands of Households')
    ax.set_title('Vehicles available in Households')

'''# TAXP and VALP, scatterplot 
value_df = dataset[['TAXP', 'VALP']]
ax3 = fig.add_subplot(224)
taxp = value_df['TAXP'] * 100
value_df = value_df.assign(TAXP = taxp)
scatter = plt.scatter(value_df.VALP, value_df.TAXP, cmap='winter', marker='o')
cbar = plt.colorbar()'''

# Call functions for plots
ax1 = fig.add_subplot(2,2,1)
pie_chart(ax1)
ax2 = fig.add_subplot(2,2,3)
bar_plot(ax2)
ax3 = fig.add_subplot(2,2,2)
histogram(ax3)