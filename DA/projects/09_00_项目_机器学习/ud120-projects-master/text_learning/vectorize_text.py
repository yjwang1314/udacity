#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        temp_counter += 1
        # if temp_counter < 200:
        if temp_counter:
            path = os.path.join('..', path[:-1])
            print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            words = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            for replace_w in ["sara", "shackleton", "chris", "germani", "sshacklensf",
                              "cgermannsf"]:
                if replace_w in words:
                    words = words.replace(replace_w, '')

            ### append the text to word_data
            word_data.append(words)
            
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == 'sara':
                from_data.append(0)
            else:
                from_data.append(1)

            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )


from nltk.corpus import stopwords
sw = stopwords.words('english')

### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer  
#将文本中的词语转换为词频矩阵  
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words = 'english')  
#计算个词语出现的次数  
X = vectorizer.fit(word_data)
X = vectorizer.transform(word_data)  
#获取词袋中所有文本关键词  
word = vectorizer.get_feature_names()  
print 'len(word):',len(word) 
#查看词频结果  
# print X.toarray()  


'''
from sklearn.feature_extraction.text import TfidfTransformer
#类调用  
transformer = TfidfTransformer()  
print transformer  
#将词频矩阵X统计成TF-IDF值  
tfidf = transformer.fit_transform(X)  
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重  
print tfidf.toarray()  
'''