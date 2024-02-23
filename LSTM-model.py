#!/usr/bin/env python
# coding: utf-8

# In[1]:


from io import open
from conllu import parse_incr
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import classification_reportf
import seaborn as sns

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[2]:


def load_sentences_tags(sentences_file, tags_file):
        """Loads sentences and tags from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []

        with open(sentences_file, 'r') as file:
            for line in file:
                # replace each token by its index
                tokens = line.split()
                #sentences.append(tokenizer.convert_tokens_to_ids(tokens))
                sentences.append(tokens)
        with open(tags_file, 'r') as file:
            for line in file:
                # replace each tag by its index
                tag_seq = [tag for tag in line.strip().split(' ')]
                tags.append(tag_seq)

        # checks to ensure there is a tag for each token
        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
#             print(sentences[i], tags[i])
            assert len(tags[i]) == len(sentences[i])

        # storing sentences and tags in dict d
        #d['data'] = sentences
        #d['tags'] = tags
        #d['size'] = len(sentences)
        
        return sentences,tags


# ## Load Dataset

# ### For Task-A (Keypharases Indentification)

# In[4]:


X_train,Y_train = load_sentences_tags('./data/task1/train/sentences.txt','./data/task1/train/tags.txt')
X_val,Y_val = load_sentences_tags('./data/task1/val/sentences.txt','./data/task1/val/tags.txt')
X_test,Y_test = load_sentences_tags('./data/task1/test/sentences.txt','./data/task1/test/tags.txt')


# ### For Task-B (keyphases Classification)

# In[ ]:


#X_train,Y_train = load_sentences_tags('./data/task2/train/sentences.txt','./data/task2/train/tags.txt')
#X_val,Y_val = load_sentences_tags('./data/task2/val/sentences.txt','./data/task2/val/tags.txt')
#X_test,Y_test = load_sentences_tags('./data/task2/test/sentences.txt','./data/task2/test/tags.txt')


# In[10]:


num_words = len(set([word.lower() for sentence in X_train for word in sentence]))
num_tags   = len(set([word for sentence in Y_test for word in sentence]))


# In[13]:


print(set([word for sentence in Y_val for word in sentence]))


# In[14]:


seq_tokenizer = Tokenizer()                     
seq_tokenizer.fit_on_texts(X_train)
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(Y_train)


# In[17]:


X_train_encoded = seq_tokenizer.texts_to_sequences(X_train)
X_val_encoded = seq_tokenizer.texts_to_sequences(X_val)
X_test_encoded = seq_tokenizer.texts_to_sequences(X_test)


# In[18]:


Y_train_encoded = label_tokenizer.texts_to_sequences(Y_train)
Y_val_encoded = label_tokenizer.texts_to_sequences(Y_val)
Y_test_encoded = label_tokenizer.texts_to_sequences(Y_test)


# In[19]:


#length of longest sentence
lengths = [len(seq) for seq in X_train_encoded]
print("Length of longest sentence: {}".format(max(lengths)))


# In[20]:


MAX_SEQ_LEN = max(lengths) 

X_train = pad_sequences(X_train_encoded, maxlen=MAX_SEQ_LEN, padding="post", truncating="post", value= 0.0)
Y_train = pad_sequences(Y_train_encoded, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

X_val = pad_sequences(X_val_encoded, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
Y_val = pad_sequences(Y_val_encoded, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

X_test = pad_sequences(X_test_encoded, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
Y_test = pad_sequences(Y_test_encoded, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")


# In[21]:


def load_glove_model(glove_file):

    f = open(glove_file,'r', encoding='utf-8')
    glove_dict = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        glove_dict[word] = embedding
    glove_dict['<unk>'] = np.zeros(200)    
    return glove_dict


# In[22]:


glove_dict = load_glove_model('./glove.6B/glove.6B.200d.txt')


# In[23]:


EMBEDDING_DIM  = 200 
VOCAB_SIZE = num_words + 1
embedding_weights = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

word2id = seq_tokenizer.word_index

# copy vectors from glove to the words present in corpus
for word, index in word2id.items():
    try:
        embedding_weights[index, :] = glove_dict[word]
    except KeyError:
        pass


# In[26]:


# one-hot encoding
dict = {'I': 1, 'O': 2}
Y_train = to_categorical(Y_train) # np.unique(labels)
Y_val = to_categorical(Y_val)
Y_test = to_categorical(Y_test)


# In[27]:


print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)


# In[28]:


NUM_CLASSES = Y_val.shape[2]


# In[ ]:





# ## Model- LSTM

# In[30]:


LSTM_model = Sequential()
LSTM_model.add(Embedding(input_dim     = VOCAB_SIZE,
                             output_dim    = EMBEDDING_DIM,
                             input_length  = MAX_SEQ_LEN,
                             weights       = [embedding_weights],
                             trainable     = True,
                             #mask_zero     = True,
))
LSTM_model.add(Bidirectional(LSTM(256, return_sequences=True,)))
LSTM_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))


# In[31]:


LSTM_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[32]:


LSTM_training = LSTM_model.fit(X_train, Y_train, batch_size=12, epochs=10, validation_data=(X_val, Y_val))


# ## Load and Save Model

# In[33]:


# serialize model to JSON
model_json = LSTM_model.to_json()
with open("LSTM_model_taskA.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
LSTM_model.save_weights("LSTM_model_taskA.h5")
print("Saved model to disk")
 
# load json and create model
json_file = open('LSTM_model_taskB.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
LSTM_model = model_from_json(loaded_model_json)
# load weights into new model
LSTM_model.load_weights("LSTM_model_taskB.h5")
print("Loaded model from disk")


# In[34]:


LSTM_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# In[ ]:





# In[35]:


loss, accuracy = LSTM_model.evaluate(X_test, Y_test, verbose = 1)
print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))


# ### Prediction

# In[59]:


pred = LSTM_model.predict_classes(X_test)


# In[47]:


# remove padding # remove 0 tags
Y_pred = []
for i in range(len(pred)):
    #print(i)
    #print(len(pred[:len(Y_test_encoded[i])])
    Y_pred.append(pred[i][:len(Y_test_encoded[i])])
    #print(Y_pred[i])


# In[48]:


for i in range(len(Y_pred)):
    if len(Y_pred[i]) != len(Y_test_encoded[i]):
        print(i)


# In[53]:


# Flatten
Y_true = [item for sublist in Y_test_encoded for item in sublist]
Y_predicted = [item for sublist in Y_pred for item in sublist]


# ### Classification Report

# In[56]:


target_name = []
target_name.append('pad')


# In[57]:


for i in label_tokenizer.index_word:
    target_name.append(label_tokenizer.index_word[i]) 


# In[58]:


print(classification_report(Y_true, Y_predicted, target_names=target_name))


# In[72]:


get_ipython().system(' pipreqs . ')


# In[65]:





# In[ ]:




