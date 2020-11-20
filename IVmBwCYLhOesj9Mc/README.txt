Good day, reader! I hope the script that this text file accompanies satisfies your requirements.

I'm providing this readme file here to have to avoid looking at an annoyingly long and hard to read
script. I'm going to be providing a summary of results I've got below:

I've used a Random Forest model to test against different selections of features.
- Based on ANOVA F-tests, variables X1, X3, X5, and X6 are considered significant.
- Based on mutual information, variables X1, X4 and X6 are considered significant.

Features selected by ANOVA F-tests produced an accuracy of 69% - which isn't too great.
Features selected by mutual information produced an accuracy of 73%, which passes success threshold.

* Noted that there is a small class imbalance but not significant enough to affect results.

Recommend removal of questions represented by variables X2, X3, X5 since they don't provide key performance
metrics that relate to customer happiness. Questions involving post sales may have a better effect
on predictions since an easier return and replacement handling procedure actually involve the
customer and expose them to services help them decide if they are satisfied better than current
survey questions.

No solution is superior - they can be 'better' when tuned correctly.
* Post note: Oddly enough a NN with sigmoid activation (basically logistic regression) yielded
better results but I binned it because a neural net is too complex for a solution like this.
 
