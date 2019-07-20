

# # Twitter Sentiment Analysis

# In[ ]:


#pip install gensim --upgrade
#pip install keras --upgrade
#pip install pandas --upgrade
#pip install tenserflow --upgrade
#pip uninstall gensim    
#sudo apt-get install python3-dev build-essential      
#sudo pip3 install --upgrade gensim
from keras.models import load_model
# In[ ]:
from gensim.models import Word2Vec
# In[ ]:
# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt


# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
# In[ ]:
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# In[ ]:
# nltk

from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec


# Utility

import numpy as np

from collections import Counter

import time
import pickle


# Set log
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:


#nltk.download('stopwords')


# ### Settings

# In[ ]:


# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024


# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"


# ### Read Dataset

# ### Dataset details
# * **target**: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# * **ids**: The id of the tweet ( 2087)
# * **date**: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# * **flag**: The query (lyx). If there is no query, then this value is NO_QUERY.
# * **user**: the user that tweeted (robotickilldozr)
# * **text**: the text of the tweet (Lyx is cool)




# In[ ]:


model=load_model(KERAS_MODEL)
# In[ ]:

w2v_model = Word2Vec.load(WORD2VEC_MODEL)

# In[ ]:
with open(TOKENIZER_MODEL, 'rb') as handle:
    tokenizer = pickle.load(handle)
#tokenizer = pickle.load(TOKENIZER_MODEL,"rb")




# In[ ]:
with open(ENCODER_MODEL, 'rb') as handle:
    encoder = pickle.load(handle)





# ### Pre-Process dataset

# In[ ]:


#stop_words = stopwords.words("english")
#stemmer = SnowballStemmer("english")



# ### Predict

# In[ ]:


def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


# In[ ]:


def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  


# In[ ]:


#predict("I love the music")


# In[ ]:


#predict("I hate the rain")




