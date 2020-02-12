import pandas as pd

# Read data
dataset = pd.read_csv('C:/Users/ldmag/Downloads/cars.csv')

# dataset.head()
X = dataset[['make', 'aspiration']]

# dataset.groupby('make').count()['aspiration'] / len(dataset)

# Probability - aspiration and make
Asp = X['aspiration'].value_counts() / X['aspiration'].count() * 100
Man = (X['make'].value_counts() / X['make'].count() * 100).round(2)

# dataframe for probabilities
Man_df = pd.DataFrame({'make': Man.index.unique(), 'make_prob': Man.values})

# Assignment header
print("DATA-51100", "[Spring 2020]")
print("Lionel Dsilva")
print("Programming Assignment 4")
print("\n")

# conditional probability
for make in X['make'].unique():
    for asp in X['aspiration'].unique():
        temp_df = X[X['make'] == make]
        make_count = temp_df.shape[0]
        temp_df2 = temp_df[temp_df['aspiration'] == asp]
        make_asp_count = temp_df2.shape[0]
        if make_asp_count != 0:
            print("Prob(aspiration="+asp+"|make="+make+")"+" = " + str(round(100*make_asp_count/make_count,2)) + "%")
        else:
            print("Prob(aspiration="+asp+"|make="+make+")"+" = " + str(0) + "%")

# print make probability
print("\n")
p_make_prob = lambda x: print("Prob(make="+x.make+") = ", x.make_prob, "%")
Man_df.apply(p_make_prob, axis = 1)

'''
# Alternative solution
X.groupby('aspiration')['make'].value_counts() / X.groupby('aspiration')['make'].count() *100
'''