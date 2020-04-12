import numpy as np
import pandas as pd
import statsmodels.api as sm

# get data
dataset = pd.read_excel("C:/Users/ldmag/Downloads/HBAT.xls")
target = dataset[['x19']]
selected = dataset[['x6','x7','x9','x11','x12','x16']]
selected = sm.add_constant(selected)

# build model
model = sm.OLS(target,selected).fit()
predictions = model.predict(selected)
model.summary()

# correlation
correlation = selected.corr()

# model after dropping X16 and X9
selected_edit = dataset[['x6','x7','x11','x12']]
selected_edit = sm.add_constant(selected_edit)
model_1 = sm.OLS(target, selected_edit).fit()
predictions_1 = model_1.predict(selected_edit)
model_1.summary()