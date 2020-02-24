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

# TAXP and VALP, scatterplot [scale TAXP values first and map function to be called in scatter]
def convert_taxp(taxp):
    tax_amt = []
    taxp_dict = get_taxp_mapping_dict()
    for i in taxp:
        if taxp_dict.__contains__(i):
            tax_amt.append(taxp_dict[i])
        else:
            tax_amt.append(np.NaN)
    return tax_amt
    
def get_taxp_mapping_dict():
    taxp_dict = {}
    taxp_dict[1] = np.NaN
    taxp_dict[2] = 1
    taxp_dict[63] = 5500
    counter = 50
    for key in range(3,23):
        taxp_dict[key] = counter
        counter += 50
    for key in range(23,63):
        taxp_dict[key] = counter + 50
        counter += 100
    counter = counter - 50
    for key in range(64,69):
        taxp_dict[key] = counter + 1000
        counter += 1000
    return taxp_dict

def scatterplot(ax):
    mrgp = dataset['MRGP']
    wgtp = dataset['WGTP']
    taxp = dataset['TAXP']
    tax_amt = convert_taxp(taxp)
    valp = dataset['VALP']
    ax.set_xlim([0, 1200000])
    ax.set_ylim([0, 11000])
    map_target = ax.scatter(valp, tax_amt, c=mrgp, s=wgtp, alpha=0.25, cmap='seismic', marker='o')
    ax.set_title('Property Taxes vs Property Values', fontsize=8)
    ax.set_ylabel('Taxes ($)', fontsize=8)
    ax.set_xlabel('Property Value ($)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=5)
    cbar = plt.colorbar(map_target, ax=ax)
    cbar.set_label('First Mortgage Payment (Monthly $)', fontsize='small')
    cbar.ax.tick_params(labelsize=5)

# Call functions for plots and save image
ax1 = fig.add_subplot(2,2,1)
pie_chart(ax1)
ax2 = fig.add_subplot(2,2,3)
bar_plot(ax2)
ax3 = fig.add_subplot(2,2,2)
histogram(ax3)
ax4 = fig.add_subplot(2,2,4)
scatterplot(ax4)

plt.savefig('pums.png', dpi = 200)