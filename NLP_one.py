# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:44:40 2024

@author: HP
"""
#basics ,advanced, python for ds ,python for nlp for test syllabus prepare it
#12 June 2024

sentence="we are learning textmining from sanjivani AI"
#if we want to know position of learning
sentence.index("learning")
##it will show learning at position?
##this is going to show character position from 0 including 

#################################
#we want to know position textmining  word
sentence.split().index("textmining")
#it will split the words in list and count the position
#sentence.split it will do tokanization 

###########################
#we want to print any word in reverse order
sentence.split()[2][::-1]
'''syntax- [start: end end:-1(start)] will start from -1,
-2,-3 till to that perticular word
-1 indicates reverse order'''
#Learning will be printed as gninrael

#################################################
#suppose we want to print first and last word of the sentence
words=sentence.split() #it will do tokanization
#result of tokanization: it will generate list  
words
first_word=words[0]
last_word=words[-1]
print(first_word)
print(last_word)
#concat this 1st and last word
concat_word=first_word+" "+ last_word
concat_word

###################################
#print even words from sentence
sentence="we are learning textmining from sanjivani AI"

[words[i] for i in range(len(words)) if i %2==0]

#print only AI 

sentence
sentence[-3:]
#in sentence last character is -1 (I), -2(A), -3(space)
#in output it will display space also
sentence[-2:]#now here space is removed bcz of -2 index it isreverse index

###################################
#print entire sentence in reverse order
sentence[::-1]

################################################
#print each word in reverse order
words
print(" ".join(word[::-1]for word in words))

##########################################
#tokenization
import nltk
nltk.download('punkt')
from nltk import word_tokenize 
words=word_tokenize("i am reading NLP fundamentals")
print(words)

############################################
#parts of speech(Pos) tagging
nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(words)

########################################
#stop words from NLTK library
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words=stopwords.words('English')
#you can verify 179 stop words in variable explorer
print(stop_words)

#---------------------------------------------------------#
'''13 June 2024'''
#suppose we want to replace word in string
sentence="I visited MY from IND on 14-02-19"
normalized_word=sentence.replace("MY", "Malasia").replace("IND","India")
normalized_word1=normalized_word.replace("14-02-19","15-03-2020")
print(normalized_word1)

#############################
#suppose we want auto correction in the sentence
from autocorrect import Speller

# Declare the function speller defined for English
spell = Speller(lang='en')

# Test the spell checker
corrected_word = spell("Englsih")
print(corrected_word)
 
#######################################
#suppose we want to correct whole sentence
import nltk
nltk.download('punkt')
from nltk import word_tokenize
sentence3="natural language processin deals withh the aart of extract"
#let us first tokenize this sentence
sentence4=word_tokenize(sentence3)
correct_sentence=" ".join([spell(word) for word in sentence4])
print(correct_sentence)

###################################################
#stemming
stemmer=nltk.stem.PorterStemmer()
stemmer.stem("programming")
stemmer.stem("programmed")
stemmer.stem("jumping")
stemmer.stem("jumped")
stemmer.stem("easily")

#Lemmatizer
#lematizer looks ito dictionary words
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize("programed")
lemmatizer.lemmatize("programs")
lemmatizer.lemmatize("batting")
lemmatizer.lemmatize("amazing")

############################################
#chunking (shallow parsing) identifying named entities
import nltk
from nltk.tokenize import word_tokenize
nltk.download("maxent_ne_chunker")
nltk.download('words')
sentence4="we are learning NLP in python by SanjivaniAI"
##first we will tokenize
nltk.download('averaged_perceptron_tagger')
words=word_tokenize(sentence4)
words=nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[a for a in i if len(a)==1]

######################################
#sentence tokenization 
from nltk.tokenize import sent_tokenize
sent=sent_tokenize("I am Aditi bairagi from sanjivani college")
sent

#########################################3
#he went to bank and checked account it was almost 0
#looking this he went to river bank and was crying
from nltk.wsd import lesk
sentence1="keep your savings in the bank"
print(lesk(word_tokenize(sentence1),'bank'))
##output Synset('savings_bank_n.02')
sentence2="It is so risky to drive over the banks of river"
print(lesk(word_tokenize(sentence2),'bank'))


---------------------------------------------------------------------
#18 june 2024

#########################################################33
#regx
import re
chat1="Hello, I am having issue with my order #4123567"

pattern='order[^d]*(\d*)'
matches=re.findall(pattern,chat1)
matches

chat2="I have a problem with my order number 21267112"
pattern='order[^\d]*(\d*)'
matches=re.findall(pattern,chat2)
matches
chat3="I have 21267112 a problem, with my order number "
pattern='order[^\d]*(\d*)'
matches=re.findall(pattern,chat3)
matches
import re
def get_pattern_match(pattern,text):
    matches=re.findall(pattern,text)
    if matches:
        return matches[0]
get_pattern_match('order[^\d]*(\d*)',chat1)

#######################################
chat1=' you ask lot of questions 12345678912, abc@xyz.com'
chat2='here it is:(123)-567-8912,abc@xyz.com'
chat3='yes,phone:12345678912 email:abc@xyz.com'
get_pattern_match('[a-zA-Z0-9_]*@[a-z]*\.[a-zA-Z0-9]*', chat1)
get_pattern_match('[a-zA-Z0-9_]*@[a-z]*\.[a-zA-Z0-9]*', chat2)
get_pattern_match('[a-zA-Z0-9_]*@[a-z]*\.[a-zA-Z0-9]*', chat3)

#######################################################
#patterns for all types of numbers
get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})', chat1)
get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})', chat2)
get_pattern_match('(\d{10})|(\(\d{3}\)-\d{3}-\d{4})', chat3)

#########################################################

text='''Born	Elon Reeve Musk
June 28, 1971 (age 52)
Pretoria, Transvaal, South Africa
Citizenship	
South Africa
Canada
United States
Education	University of Pennsylvania (BA, BS)
Title	
Founder, CEO, and chief engineer of SpaceX
CEO and product architect of Tesla, Inc.
Owner, CTO and Executive Chairman of X (formerly Twitter)
President of the Musk Foundation
Founder of The Boring Company, X Corp., and xAI
Co-founder of Neuralink, OpenAI, Zip2, and X.com (part of PayPal)
Spouses	
Justine Wilson
​
​(m. 2000; div. 2008)​
Talulah Riley
​
​(m. 2010; div. 2012)​
​
​(m. 2013; div. 2016)'''

get_pattern_match(r'age (\d+)',text)
get_pattern_match(r'Born(.*)\n', text).strip()
get_pattern_match(r'Born.*\n(.*)\(age', text).strip()
get_pattern_match(r'\(age.*\n(.*)', text)

#-------------------------------------------------------------------
#19 June 2024
def extract_personal_information(text):
   age=get_pattern_match('age (\d+)',text)
   full_name=get_pattern_match('Born(.*)\n', text)
   birth_date=get_pattern_match('Born.*\n(.*)\(age', text)
   birth_place=get_pattern_match('\(age.*\n(.*)', text)

   return{
        'age':int(age),
        'name':full_name.strip(),
        'birth_date':birth_date.strip(),
        'birth_place':birth_place.strip()}
text='''Ambani in 2007
Born	Mukesh Dhirubhai Ambani
19 April 1957 (age 67)
Aden, Colony of Aden
(present-day Yemen)[1][2]
Nationality	Indian
Alma mater	
St. Xavier's College, Mumbai
Institute of Chemical Technology (B.E.)
Occupation(s)	Chairman and MD, Reliance Industries
Spouse	Nita Ambani ​(m. 1985)​[3]
Children	3
Parents	
Dhirubhai Ambani (father)
Kokilaben Ambani (mother)
Relatives	Anil Ambani (brother)
Tina Ambani (sister-in-law)'''
extract_personal_information(text)
#imports

#############################################3
#pypdf
from PyPDF2 import PdfFileReader
#importing required modules
from PyPDF2 import PdfReader
reader=PdfReader('C:/Users/HP/Desktop/DS/python with NLP/kopargaon-part-1.pdf')
print(len(reader.pages))
page=reader.pages[1]
text=page.extract_text()
print(text)

#--------------------------------------------------------------------
#20 June 2024

import re
sentence5="sharat twitted wittnessing 68th republic day India from Rajpath, \new Delhi, mesmorizing performance by Indian Army!"
re.sub(r'([^\s\w]|_)+', ' ',sentence5).split()

'''
re.sub(r'([^\s\w]|_)+', ' ', some_string)
will replace sequences of non-alphanumeric characters
(including punctuation nut excluding whitespaces)
with a single space. This is commonly used to clean up text 
by removing punctuation and other non-word characters,
making it easier to process for tasks like text analysis
or machine  learning'''

##############################3
#extracting n-grams
#n-gram can be extracted using three techniques
#1.custom defined functions
#2.NLTK
#3.TextBlob
###########################
#extracting n-grams using custom defined function

import re
def n_gram_extractor(input_str, n):
    tokens=re.sub(r'([^\s\w]|_)+', ' ', input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
n_gram_extractor("The cute little boy is playing with kitten",2)
n_gram_extractor("The cute little boy is playing with kitten",3)


#------------------------------------------------------------
#21 June 2024

from nltk import ngrams
#extraction n-grams with nltk
#n-grams if their are two or more words to etract called as n-grams
list(ngrams("The cute little boy is playing with kitten".split(),2))
list(ngrams("The cute little boy is playing with kitten".split(),3))

##################################
from textblob import TextBlob
blob=TextBlob("The cute little boy is playing with kitten")
blob.ngrams(n=2)
blob.ngrams(n=3)

####################################
#tokenization using keras
sentence5="sharat twitted wittnessing 68th republic day India from Rajpath, \new Delhi, mesmorizing performance by Indian Army!"
sentence5
from keras.preprocessing.text import text_to_word_sequence

##############################
#tokenization using TextBlob
from textblob import TextBlob
blob=TextBlob(sentence5)
blob.words

##################################
#tweet tokenizer
from nltk.tokenize import TweetTokenizer
tweet_tokenizer=TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)

#####################################
#multi-word-expression
from nltk.tokenize import MWETokenizer
'''multi- word tokenizer are essential for tasks
where the meaning of the text heavily depends on
the interpretatio of phrases as wholes rather than as sums,
 of individual words. For instance, is sentiment analysis,
 recognizing "not good" as a single negative sentiment unit 
 rather than as "not" and "good" seperately can significantly
 affect the outcome.'''
 
 sentence5
 mwe_tokenizer=MWETokenizer([('republic', 'day')])
 mwe_tokenizer.tokenize(sentence5.split())
 mwe_tokenizer.tokenize(sentence5.replace('!',' ').split())

