import pandas as pd

# Read data
dataset = pd.read_csv('C:/Users/ldmag/Downloads/cars.csv')

# dataset.head()
X = dataset[['make', 'aspiration']]

# dataset.groupby('make').count()['aspiration'] / len(dataset)

# Probability - aspiration and make
Asp = X['aspiration'].value_counts() / X['aspiration'].count() * 100
Manufacturer = X['make'].value_counts() / X['make'].count() * 100
'''
ProbTable = pd.DataFrame(Asp).append(Manufacturer) 
X.groupby('aspiration')['make'].value_counts() / X.groupby('aspiration')['make'].count() *100
'''

# dataframe for probabilities
make_prob = pd.DataFrame({'make': X['make'].unique(), 'make_P': Manufacturer})

# conditional probability
for asp in X['aspiration'].unique():
    for make in X['make'].unique():
        temp_df = X[X['make'] == make]
        make_count = temp_df.shape[0]
        temp_df2 = temp_df[temp_df['aspiration'] == asp]
        make_asp_count = temp_df2.shape[0]
        if make_asp_count != 0:
            print("Prob(aspiration="+asp+"|make="+make+")"+" = " + str(round(100*make_asp_count/make_count,2)) + "%")
        else:
            print("Prob(aspiration="+asp+"|make="+make+")"+" = " + str(0) + "%")

# print make probability
p = lambda x: print("Prob(make=" + x.make+") =", x.make_P, "%")
make_prob.apply(p,axis=1)
