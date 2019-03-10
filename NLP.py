# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:28:09 2019

@author: hp
"""

#NATURL LANGUAGE PROCESSING

#IMPORTING THE LIBRARIS
import pandas as pd

#DATASET
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#cleaning text

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('English'))]
    review=' '.join(review)
    corpus.append(review)
    
#BAG OF WORDS MODEL
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
features=cv.fit_transform(corpus).toarray()
labels=dataset.iloc[:,[1]].values

#SPLIT THE DATASET
from sklearn.cross_validation import train_test_split
features_train,features_test,label_train,label_test=train_test_split(features,labels,test_size=0.25,random_state=0)

"""naive bayes classification"""
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(features_train,label_train)

#predicting
pred=classifier.predict(features_test) 

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(label_test,pred)
#55+91, 42+12

"""KERNEL SVM(rbf)"""
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(features_train,label_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(label_test,pred)
#55+91, 42+12

"""KERNEL SVM(sigmoid)"""
from sklearn.svm import SVC
classifier=SVC(kernel='sigmoid',random_state=0)
classifier.fit(features_train,label_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(label_test,pred)

"""knn"""
from sklearn.neighbors import KNeighborsClassifier 
classifier=KNeighborsClassifier()
classifier.fit(features_train,label_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm3=confusion_matrix(label_test,pred)

"""Decision tree classifier"""
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(features_train,label_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm4=confusion_matrix(label_test,pred)

"""Random forest classifier"""
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(features_train,label_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm5=confusion_matrix(label_test,pred)


