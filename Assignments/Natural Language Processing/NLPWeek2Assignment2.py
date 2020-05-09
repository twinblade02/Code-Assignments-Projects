from collections import Counter
import nltk
from nltk.corpus import *

# Unigram model
def Unigram(corpus,category):
    tc = len(corpus.words(categories = category))
    counts = Counter(corpus.words(categories = category))
    for w in counts:
        counts[w] = round(counts[w] / tc, 4)
       # print(f"{w}: {counts[w]}")
    print("Most common unigram words")
    print(counts.most_common(15))


# Bigram model
from nltk import bigrams
from collections import defaultdict

def Bigram(corpus,category):
    sentence = corpus.sents(categories = category)[0]
    list(bigrams(sentence, pad_left=True, pad_right=True))
    count = defaultdict(lambda: defaultdict(int))
    for s in corpus.sents(categories = category):
        for w1,w2 in bigrams(s, pad_left=True, pad_right=True):
            count[w1][w2] += 1
    for w1 in count:
        total = sum(count[w1].values())
        for w2 in count[w1]:
            count[w1][w2] = round(count[w1][w2] / total, 3)
            
    sort_dict = sorted(count.items(), key=lambda x: x[1][1], reverse=True)
    # **warning** print the ALL bigrams **warning**
    ''' print(sort_dict) '''
    # return only the first 2
    return sort_dict[:2]

# function calls
Unigram(brown,'humor')
Unigram(brown,'hobbies')
Bigram(brown,'humor')
Bigram(brown,'hobbies')

'''
length = nltk.corpus.brown.words(categories='hobbies')
length1 = nltk.corpus.brown.words(categories='humor')
len(length)
len(length1)
'''

''' Part 2 

I used the brown corpus to look at both the Unigram and Bigram models and split them via categories so that we got a smaller result while still 
retaining a nice corpus with proper sentences. Categories "humor" and "hobbies" were used. The total length of the hobbies category is 82345 and 
the length of the humor category is 21695. Note that humor is signifcantly smaller. 
Calling the function using Unigram(brown,'humor') will yield the most common Unigrams according to their probabilities. Comparing them both, 
we can note that the most common of them are actually stopwords and punctuation and implementing the stopwords function to omit all of these 
will help us show us other words that appear in that corpus. Removing these may also reveal some word patterns we have missed here.
The bigram results are dissimilar, which would make sense given that the categories are very different. Again, we would be able to reduce the number
of results we get by removing both punctuation and stopwords from the corpora. 
'''