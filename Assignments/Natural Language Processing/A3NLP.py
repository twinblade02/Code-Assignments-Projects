# Ex 1
import nltk
import re

text = "The quick brown fox jumps over the lazy dog 123456789 ,./[](){}?#¬`$%^&*"
dec = "33.333"

fil = nltk.re_show(r'[a-zA-Z]+', text)
print(fil)

fil2 = nltk.re_show(r'[A-Z][a-z]*', text)
print(fil2)

fil3 = nltk.re_show(r'p[aeiou]{,2}t', text)
print(fil3)

fil4 = nltk.re_show(r'\d+(\.\d+)?', text)
fil4_1 = nltk.re_show(r'\d+(\.\d+)?', dec)
print(fil4)

fil5 = nltk.re_show(r'([^aeiou][aeiou][^aeiou])*', text)
print(fil5)

fil6 = nltk.re_show(r'/w+|[^\w\s]+', text)
print(fil6)
'''
- Expression 1: [a-zA-Z]+
Matches one or more upper case or lower case alphabets and disregards numerical characters

- Expression 2: [A-Z][a-z]*
Matches to a string beginning with an upper case ASCII character followed by lower case characters

- Expression 3: p[aeiou]{,2}t
Matches to a sting with ASCII character 'p' with a combination of 2 vowels - followed by the character 't'

- Expression 4: \d+(\.\d+)?
Matches digits with a decimal point, optionally includes digits after decimal

- Expression 5: ([^aeiou][aeiou][^aeiou])*
Expression matches and displays lower case vowels accompanied by consonants on either side.
Excludes numerical characters and punctuation. Includes whitespace.

- Expression 6: \w+|[^\w\s]+
Forms tokens out of alphabetic sequences, secondary expressions(punctuation), digits. Excludes whitespace. 
'''

# Ex 2
# A single determiner (assume that a, an, and the are the only determiners)
text2 = "This is a sentence and everything after is an extension of it. These are the numbers 2468"
fil7 = nltk.re_show(r'\b(a|an|the)\b', text2)

# An arithmetic expression using integers, addition, and multiplication, such as 2*3+8
text3 = "Is 2*3+8 equal to 14 or 22?"
fil8 = nltk.re_show(r'\d[\-\+\*]\d[\-\+\*]\d', text3)