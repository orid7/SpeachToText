# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:53:28 2018

@author: ori dahari
"""
from __future__ import print_function
import time
import boto3

def main(fileName,jobName):
          
    ###Audio to text
    transcribe = boto3.client('transcribe', region_name="us-west-2")
    job_name = jobName
    job_uri = "https://s3-us-west-2.amazonaws.com/recordtest/{}.wav".format(fileName)
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode='en-US',  
    )
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
    	
        time.sleep(5)
    print(status)
    
     
    ###URL to String
    import urllib.request, json 
    with urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri']) as url:
        data = json.loads(url.read().decode())
    
        EnText=data['results']['transcripts'][0]['transcript']
        print(EnText)

    return EnText
