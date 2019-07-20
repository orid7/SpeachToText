# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 18:40:34 2019

@author: ori dahari
"""

import pyaudio
import wave
import boto3
import os

os.chdir('C:\\Users\ori dahari\Documents\MBA\mini9\Practicum') 
os.getcwd()

def main(recSec,fileName):
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = recSec
    WAVE_OUTPUT_FILENAME = "{}.wav".format(fileName)
     
    audio = pyaudio.PyAudio()
     
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print ("recording...")
    frames = []
     
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print ("finished recording")
     
     
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
     
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    
    
    #upload to S3
  
    
     
    s3 = boto3.client(
    "s3",
    )
    bucket_resource = s3
    
    bucket_resource.upload_file(
    Bucket = 'recordtest',
    Filename=WAVE_OUTPUT_FILENAME,
    Key=WAVE_OUTPUT_FILENAME
    )
