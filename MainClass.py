# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:50:06 2019

@author: ori dahari
"""

import os
import re
os.chdir('C:\\Users\ori dahari\Documents\MBA\mini9\Practicum') 

import recordToS3
import SpeachToText
import Twitter_Sentiment_Analysis_loadModel

fileName="test81"
jobName="test81"
recLengthSec=4

recordToS3.main(recLengthSec,fileName)
text=SpeachToText.main(fileName,jobName)
sentenceResult=Twitter_Sentiment_Analysis_loadModel.predict(text)
sentenceResult['sentence']=text
resultList=[sentenceResult]
wordList = re.sub("[^\w]", " ",  text).split()



for i in range(len(wordList)):
    x=Twitter_Sentiment_Analysis_loadModel.predict(wordList[i]) 
    x['word']=wordList[i]
    resultList.append(x)

resultList
