import pandas as pd
import numpy as np

train = pd.read_csv('C:/Users/ldmag/Desktop/lewisdatathon2020/datathon2020_train.csv')
test = pd.read_csv('C:/Users/ldmag/Desktop/lewisdatathon2020/datathon2020_test.csv')
full_dataset = pd.concat([train, test])

na_percents = full_dataset.isna().sum() / len(full_dataset) * 100
drop_these = na_percents[na_percents > 50.0].index.tolist()
full_dataset = full_dataset.drop(drop_these, axis = 1)
# full_dataset.isna() / len(full_dataset) * 100

# Imputing na
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
full_dataset = pd.DataFrame(imputer.fit_transform(full_dataset), columns = full_dataset.columns)
df = full_dataset.copy()

# set categorical and features
full_dataset['F8'] = full_dataset['F8'].astype('category')
full_dataset['F11'] = full_dataset['F11'].astype('category')
full_dataset['F136'] = full_dataset['F136'].astype('category')
full_data_onehot = pd.get_dummies(full_dataset, columns = ['F8','F11','F136'], prefix = ['F8_Area_Condition', 'F11_Climate', 'F136'])

full_dataset['COVID_TCPM'].astype('str')
full_dataset['COVID_TCPM'] = full_dataset['COVID_TCPM'].apply(lambda x: 0 if x == 'low' else 1)
full_data_onehot.drop(['COVID_TCPM'], axis = 1, inplace = True)
target = full_dataset[['COVID_TCPM']]

# split
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(full_data_onehot, test_size = 2486)
Y_train, Y_test = train_test_split(target, test_size = 2486)

# model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

# checking score
classifier.score(X_train, Y_train)
classifier.score(X_test, Y_test)

# metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

ID = np.array(X_test['ID'].astype('int64'))
submission = pd.DataFrame([ID, Y_pred])
submission = submission.T
submission = submission.rename(columns={0:'ID', 1:'COVID_TCPM'})
submission['COVID_TCPM'] = submission['COVID_TCPM'].apply(lambda x: 'low' if x == 0 else 'high')
submission.to_csv('submission2.csv', index = False)

corr = full_data_onehot.corr()

