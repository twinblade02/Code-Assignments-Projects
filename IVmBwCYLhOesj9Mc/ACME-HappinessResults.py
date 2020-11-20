import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_full = pd.read_csv('ACME-HappinessSurvey2020.csv')

# satisfied or not - classification problem. I have a few ideas. 
# check nans
nans = data_full.isnull() * 100 / len(data_full)
# EDA 
c = data_full.corr() #not enough information from this - a groupby might help but lets see where this goes
fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize=(20,15))
for n, col in enumerate(data_full.columns):
    sns.distplot(data_full[col], ax = axes[n//4, n%4])
    
fig1, axes1 = plt.subplots(nrows= 3, ncols=3, figsize=(20,15))
for v, co in enumerate(data_full.columns):
    sns.boxplot(data_full[co], ax = axes1[v//3, v%3])
    

# scale data - min max is best to match Y's binary values (for proper scaling)
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
clean_data = data_full.copy()
clean_data[['X1','X2','X3','X4','X5','X6']] = mms.fit_transform(clean_data[['X1','X2','X3','X4','X5','X6']])

# feature selection - Select KBest based on ANOVA F-score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
X = clean_data[['X1','X2','X3','X4','X5','X6']]
Y = clean_data['Y']
names = pd.DataFrame(X.columns)
model_fclass = SelectKBest(score_func=f_classif, k='all')
results = model_fclass.fit(X,Y)
results_fclass = pd.DataFrame(results.scores_)
sc = pd.concat([names, results_fclass], axis = 1) # based on this - X1, X3, X5, X6 are significant 

# feature selection on mutual information
model_mutual = SelectKBest(score_func=mutual_info_classif, k='all')
results_mutual = model_mutual.fit(X,Y)
results_mutual_df = pd.DataFrame(results_mutual.scores_)
scrd = pd.concat([names, results_mutual_df], axis = 1) # based on this - X1, X4, X6 are important

###############################################################################################################
# lets try the first feature model but make new, seperate X variables
from sklearn.model_selection import train_test_split
fclass_X = clean_data[['X1','X3','X5','X6']]
mutual_X = clean_data[['X1','X4','X6']]

X_train, X_test, Y_train, Y_test = train_test_split(fclass_X, Y, test_size = 0.2, random_state = 69) # lol random state
X_train_mut, X_test_mut, Y_train_mut, Y_test_mut = train_test_split(mutual_X, Y, test_size = 0.2, random_state = 69)

'''
For a robust classifier, an SVM model is good but to deal with other potential outliers in the unseen set, I think its a safer option to try a random forest model.
An SVM that is not tuned correctly might attempt to overfit - and on the other hand, an improperly pruned RF model might kick results out of the door. Grid searching might work
but overcomplicating something is as bad as not doing anything about it in the first place. Keeping with the concept of Occams Razor, I will keep this model very simple
''' 

# begin RF testing, f-classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

rf_clf = RandomForestClassifier(criterion='gini', max_depth=10, max_features='auto', n_estimators=50, random_state=42)
rf_clf.fit(X_train, Y_train)
Y_pred_fclass = rf_clf.predict(X_test)
cm_fclass = confusion_matrix(Y_test, Y_pred_fclass)
acc_fclass = accuracy_score(Y_test, Y_pred_fclass)

# adding grid search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'criterion': ['gini','entropy'],
    'max_depth': [2, 4, 8, 10, 20, 50],
    'max_features': ['auto','sqrt','log2'],
    'n_estimators': [10, 50, 100, 250, 400, 500]}

#GS = GridSearchCV(estimator = rf_clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
#GS.fit(X_train, Y_train)
#GS.best_params_

plt.figure(figsize=(20,15))
plt.plot([0,1],[0,1], 'r--')
probs = rf_clf.predict_proba(X_test)
probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(Y_test, probs)
roc_auc = auc(fpr,tpr)
label = 'Random Forest Classifier:' + '{0:.2f}'.format(roc_auc)
plt.plot(fpr,tpr,c='g',label=label,linewidth=4)
plt.xlabel('FP Rate', fontsize = 16)
plt.ylabel('TP Rate',fontsize = 16)
plt.title('ROC', fontsize = 16)
plt.legend(loc = 'lower right', fontsize = 15)

# RF test for other feature set (mutual information) EDIT: this model yields 73% - a grid search may yield better performance
rf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=8, max_features='auto', n_estimators=250, random_state=2)
rf.fit(X_train_mut, Y_train_mut)
Y_pred_mut = rf.predict(X_test_mut)
cm_mut = confusion_matrix(Y_test_mut, Y_pred_mut)
acc_mut = accuracy_score(Y_test_mut, Y_pred_mut)

#GS1 = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
#GS1.fit(X_train_mut, Y_train_mut)
#GS1.best_params_

plt.figure(figsize=(20,15))
plt.plot([0,1],[0,1], 'r--')
probs1 = rf.predict_proba(X_test_mut)
probs1 = probs1[:,1]
fpr1, tpr1, thresholds1 = roc_curve(Y_test_mut, probs1)
roc_auc = auc(fpr1,tpr1)
label = 'Random Forest Classifier:' + '{0:.2f}'.format(roc_auc)
plt.plot(fpr1,tpr1,c='g',label=label,linewidth=4)
plt.xlabel('FP Rate', fontsize = 16)
plt.ylabel('TP Rate',fontsize = 16)
plt.title('ROC', fontsize = 16)
plt.legend(loc = 'lower right', fontsize = 15)

'''
RESULTS:
It appears that questions X1, X4, and X6 determine if a customer is satisfied or not. Presence of these variables also improves predictions. As for accuracy scores,
it is possible to get better results with a more balanced class. There are 69 positive and 57 negatives instances in the data. The RF model is a good solution to this 
problem since it is robust to outliers and does not need cleaning (even though it was done here) - testing it on unseen data after retraining could yield better results.

As for the questions - in real world applications, questions like X3 and X5 need not be added at all, since these do not depict key performance indices. Recommended they be removed
and be replaced with some thing like "quality of contents" and "quality of service provided". Post sales questions also contribute to customer satisfaction like returns and replacements.
''' 
