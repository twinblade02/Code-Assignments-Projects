import nltk
from nltk.book import *
import string

#Chapter 1
# text 4 - Inaugral address
lst = set(text4)
big_words = [word for word in lst if len(word) == 10]
fdist = FreqDist(big_words)
fdist = sorted(fdist, reverse = True) #Grading: FreqDist sorts word in descending frequency order (most frequent words first).
print(fdist)

# Monty Python and the Holy Grail (text 6) - find and print all uppercase words
split = set(text6)
capital_words = [w for w in split if w[0].isupper()]
print(capital_words)

# function that calculates how often a word occurs in %
def percent(word, text):
    freq = int(text.count(word))
    total = int(len(text))
    return str(round(100 * (freq / total), 2)) + "%"

#Grading:  include code to test function

# Chapter 2
# Program to find words that occur at lest 3 times in corpus
from nltk.corpus import brown
brown_words = set(brown.words())
freq_brownWords = FreqDist(brown.words())
result = [x for x in brown_words if freq_brownWords[x] >=2] #Grading: >= 3
print(result)

#Grading: this displays words that occur 2 times or more

# function that finds 50 most frequent words that are not stopwords
from nltk.corpus import stopwords
def most_frequent():
    mfreq = FreqDist(brown.words())
    w_list = [wo for wo in mfreq]
    for wd in w_list:
        if wd in stopwords.words('english') or not wd.isalpha():
            mfreq.pop(wd)
    return mfreq

#Grading:  function should have a parameter for the text

#Grading: function that computes frequency of word in corpus section
def word_freq(word, category):
    wdist = FreqDist([wo for wo in brown.words(categories = category)])
    return wdist[word]

#Grading:  provide code to test your function
