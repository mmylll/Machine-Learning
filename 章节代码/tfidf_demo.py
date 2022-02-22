import nltk
import math
import string
#from nltk.corpus import stopwords
#from collections import Counter
#from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]
vectorizer=TfidfVectorizer(min_df=1)
cret = vectorizer.fit_transform(corpus)
print(cret)
fnames = vectorizer.get_feature_names()
print(fnames)
arr = vectorizer.fit_transform(corpus).toarray()
print(arr)
