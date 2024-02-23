#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 
import pandas as pd 
import os
import nltk
from nltk.corpus import wordnet
import re
import string
from nltk.corpus import stopwords


# ### Reading corpus

# In[6]:


train_folder = "./data/scienceie2017_train/train2/"
dev_folder = "./data/scienceie2017_dev/dev/"
test_folder = "./data/semeval_articles_test/"


# In[7]:


def read_text_and_ann(textfolder):
    text_data =[]
    annotation=[]
    flist = os.listdir(textfolder)
    for f in flist:
        #print(f)
        if not f.endswith(".ann"):
            #print(f)
            continue
        f_anno = open(os.path.join(textfolder, f), "r")
        #print(f_anno)
        f_text = open(os.path.join(textfolder, f.replace(".ann", ".txt")), "r")
        #print(f_text)
        # there's only one line, as each .ann file is one text paragraph
        for l in f_text:
            text = l
            text_data.append(text)
            #print(text)
        #break
        anno =[]
        for l in f_anno:
            
            #print(l)
            anno_inst = l.strip("\n").split("\t")
            #print(anno_inst)
            #keypharse = anno_inst[2]
            #print(keypharse)
            if len(anno_inst) == 3:
                anno_inst1 = anno_inst[1].split(" ")
                anno_inst1.append(anno_inst[2])
                #print(anno_inst1)
                anno.append(anno_inst1)
                #if len(anno_inst1) == 3:
                    #print(anno_inst1)
                    #keytype, start, end = anno_inst1
                #else:
                   # print(anno_inst1)
                    #keytype, start, _, end = anno_inst1
                #break
        annotation.append(anno)
                
    return text_data, annotation


# In[8]:


train_text, train_annotation =  read_text_and_ann(train_folder)
val_text, val_annotation =  read_text_and_ann(dev_folder)
test_text, test_annotation =  read_text_and_ann(test_folder)


# ### Intialize  text

# In[15]:


text = train_text[29]


# In[14]:


train_annotation[89]


# In[ ]:





# In[ ]:





# In[ ]:





# ### Cleaning the data and Removing StopWords

# In[16]:


def clean_data(text):
    
    text = text.lower()
    
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub("\'\w+", '', text)
    text = re.sub("[0-9_]+", "", text)
    # Remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    
    text = re.sub('\s{2,}', " ", text)
    text = text.split()
    return text


# In[17]:


clean_text = clean_data(text)


# In[18]:


print(clean_text)


# In[19]:


# Remove stop words
stop_words = set(stopwords.words('english'))
preprocessed_text = [token for token in clean_text if token not in stop_words]
preprocessed_text = " ".join(preprocessed_text)    


# In[20]:


print(preprocessed_text)


# ## Find Hypernyms Pairs present in the text

# In[21]:


def get_hypernyms(word):
    synsets = wordnet.synsets(word)
    hypernyms = []
    for synset in synsets:
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                hypernyms.append(lemma.name())
    return list(set(hypernyms))


# In[22]:


def find_hypernym_pairs(sentence):
    tokens = nltk.word_tokenize(sentence)
    pairs = []
    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)):
            if tokens[j] in get_hypernyms(tokens[i]) or tokens[i] in get_hypernyms(tokens[j]):
                if (tokens[i], tokens[j]) not in pairs:
                    pairs.append((tokens[i], tokens[j]))
    return pairs


# In[23]:


hypernym_pairs = find_hypernym_pairs(preprocessed_text)
print(hypernym_pairs)


# In[ ]:





# ## Find Synonyms pairs in the text

# In[27]:


# Define a function to find synonyms for a given word
def get_synonyms(word):
    synsets = wordnet.synsets(word)
    synonyms = []
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))


# In[28]:


def find_synonym_pairs(sentence):
    tokens = nltk.word_tokenize(sentence)
    pairs = []
    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)):
            if tokens[j] in get_synonyms(tokens[i]) or tokens[i] in get_synonyms(tokens[j]):
                if (tokens[i], tokens[j]) not in pairs:
                    pairs.append((tokens[i], tokens[j]))
    return pairs


# In[29]:


synonym_pairs = find_synonym_pairs(preprocessed_text)
print(synonym_pairs)


# In[ ]:




