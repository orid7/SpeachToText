# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:50:06 2019

@author: ori dahari
"""

import recordToS3
import SpeachToText

fileName="test2"
jobName="test21"
recLengthSec=4

recordToS3.main(recLengthSec,fileName)
SpeachToText.main(fileName,jobName)



