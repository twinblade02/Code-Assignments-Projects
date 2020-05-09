import nltk
import re
from nltk.book import *

# Ex 1: transform into "NLP is fun!" with assignment statements
words = ['is','NLP','fun','?']
tmp = words[1]
words[1], words[0], words[3] = words[0], tmp, '!'
## tuple version
a,b,c,d = ('is','NLP','fun','?')
a,b = b,a
d = '!'
print(a,b,c,d)



# Ex 2
word_table = [['']*3]*4
word_table[1][2] = 'hello'
'''This occurs because we're essentially creating lists by multiplying empty strings.'''
#Grading:  What occurs?  There is no display of the result.
word_table1 = []
for i in range(4):
    word_table1.append(['']*4)
word_table1[1][2] = 'hello'

''' Pointer to add 'hello' to a list is different as opposed to the previous list. In the new list, we have the pointer essentially
saying - "add the word specified in list at index 1, and within that list, add the word to index 2. Using the range method is more accurate.'''

# Ex 3
def shorten(text,n):
    freq = nltk.FreqDist(text).most_common(n)
    freq = [word for (word,num) in freq]
    print(freq)
    return [w for w in text if w not in freq]
print(' '.join(shorten(text3, 50) [:80]))
#Readability: Code is readable, the output however - does not make sense from a language perspective. However, the human mind may be able to interpolate what comes next to an extent.

# Ex 4
def novel10(text):
    remove = int(0.9* len(text))
    one, two = text[:remove], text[remove:]

    u_words_one = set(one)
    u_words_two = set(two)
    return [w for w in u_words_two if w not in u_words_one]

#function call
novel10(text3)
