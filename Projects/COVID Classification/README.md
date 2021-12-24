## Script description

This is the Kaggle submission for the Lewis University Data-SAIL competition, wherein the classifier was able to secure a test score of 91% accuracy, placing the team in the Top 5. 

## Model Structure
**Preprocessing**
Data provided was pre-split into a training and test set. This was concatenated to standardize valeus; columns with nan values greater than 50% were removed via a mask. 
A simple imputer was used to fill in the remaining 'NaN' values, using a most frequent value strategy. 
Specific columns were set to categorical values, and were one-hot encoded. 

Further, a target was defined (the Y value);  and the dataset was resplit into a training and test set, this time - with a defined target. 

**The classifier**
A Random Forest classifier was used, using the entropy and 15 estimators parameters.
