#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import logging
import os
import shutil
import random
import numpy as np
import os
import sys
import argparse
import random
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
from pytorch_pretrained_bert import BertForTokenClassification, BertConfig
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from pytorch_pretrained_bert import BertTokenizer



# In[18]:


#Parameters
data_dir = './data/task1/' # for Task-A(keyphare Identification)
#data_dir = './data/task2/' # for Task-B(keyphare Classification)
bert_model_dir = './model/' # for bert
#bert_model_dir = './model_sci/' # for scienticif bert
model_dir = 'experiments/base_model'
batch_size = 4
max_len = 512
learning_rate = 3e-5
epoch_num = 10
clip_grad =10


# In[3]:


class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_len = max_len
        self.token_pad_idx = 0

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        tag2idx = self.tag2idx
        idx2tag = self.idx2tag
        self.tag_pad_idx = self.tag2idx['O']

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)

    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences_tags(self, sentences_file, tags_file, d):
        sentences = []
        tags = []

        with open(sentences_file, 'r') as file:
            for line in file:
                # replace each token by its index
                tokens = line.split()
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))
        
        with open(tags_file, 'r') as file:
            for line in file:
                # replace each tag by its index
                tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
                tags.append(tag_seq)

        # checks to ensure there is a tag for each token
        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
#             print(sentences[i], tags[i])
            assert len(tags[i]) == len(sentences[i])

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['tags'] = tags
        d['size'] = len(sentences)

    def load_data(self, data_type):
        data = {}
        
        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            tags_path = os.path.join(self.data_dir, data_type, 'tags.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data):
        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        # one pass over data
        for i in range(data['size']//self.batch_size):
            # fetch sentences and tags
            sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
            tags = [data['tags'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_tags = self.tag_pad_idx * np.ones((batch_len, max_len))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                    batch_tags[j][:cur_len] = tags[j]
                else:
                    batch_data[j] = sentences[j][:max_len]
                    batch_tags[j] = tags[j][:max_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)
    
            yield batch_data, batch_tags


# In[4]:


class RunningAverage():
    """A simple class that maintains the running average of a quantity"""
    
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


# ### Loading Data

# In[5]:


# Initialize the DataLoader
print("Loading the datasets...")
data_loader = DataLoader(data_dir, bert_model_dir, token_pad_idx=0)
    


# In[6]:


# Load training data and test data
train_data = data_loader.load_data('train')
val_data = data_loader.load_data('val')
test_data = data_loader.load_data('test')


# In[7]:


# Specify the training and validation dataset sizes
train_size = train_data['size']
val_size = val_data['size']
test_size = test_data['size']


# In[8]:


train_size , val_size , test_size


# In[9]:


train_steps = train_size // batch_size
val_steps = val_size // batch_size


# ### Model

# In[10]:


# Prepare model
model = BertForTokenClassification.from_pretrained(bert_model_dir, num_labels=len(data_loader.tag2idx))
#model.to(device)


# #### Not fine-tunnig the whole model. only the classifier part

# In[11]:


# Prepare optimizer
param_optimizer = list(model.classifier.named_parameters()) 
optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))


# ### Training the model

# In[12]:


def train(model, data_iterator, optimizer, scheduler):
    """Train the model on `steps` batches"""
    # set model to training mode
    scheduler.step()
    model.train()
    

    # a running average object for loss
    loss_avg = RunningAverage()
    
    # Use tqdm for progress bar
    t = trange(train_steps)
    for i in t:
        
        # fetch the next training batch
        batch_data, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)
        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip_grad)
        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))


# In[13]:


def evaluate(model, data_iterator, mark='Eval'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    

    idx2tag = idx2tag

    true_tags = []
    pred_tags = []

    # a running average object for loss
    loss_avg = RunningAverage()

    for _ in range(eval_steps):
        
        # fetch the next evaluation batch
        batch_data, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)

        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
        
        loss_avg.update(loss.item())
        
        batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks)  # shape: (batch_size, max_len, num_labels)
        
        batch_output = batch_output.detach().cpu().numpy()
        batch_tags = batch_tags.to('cpu').numpy()

        pred_tags.extend([idx2tag.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])
        true_tags.extend([idx2tag.get(idx) for indices in batch_tags for idx in indices])
    assert len(pred_tags) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    f1 = f1_score(true_tags, pred_tags)
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics: ".format(mark) + metrics_str)
    report = classification_report(true_tags, pred_tags)
    print(report)
    
    return metrics


# #### save checkpoints

# In[14]:


def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


# #### Load Checkpoints

# In[15]:


def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


# In[16]:


def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, model_dir, restore_file=None):
    
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file +'.pth.tar')
        load_checkpoint(restore_path, model, optimizer)
        
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, epoch_num))

        # Compute number of batches in one epoch
        train_steps = train_size // batch_size
        val_steps = val_size // batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data)
        # Train for one epoch on training set
        train(model, train_data_iterator, optimizer, scheduler)

        # data iterator for evaluation
        train_data_iterator = data_loader.data_iterator(train_data)
        val_data_iterator = data_loader.data_iterator(val_data)

        # Evaluate for one epoch on training set and validation set
        eval_steps = train_steps
        train_metrics = evaluate(model, train_data_iterator, mark='Train')
        eval_steps = val_steps
        val_metrics = evaluate(model, val_data_iterator, mark='Val')
        
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer.optimizer if args.fp16 else optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer_to_save.state_dict()},
                               is_best=improve_f1>0,
                               checkpoint=model_dir)
        


# In[17]:


# Train and evaluate the model
logging.info("Starting training for {} epoch(s)".format(epoch_num))
train_and_evaluate(model, train_data, val_data, optimizer, scheduler, model_dir, restore_file = None)


# In[ ]:





# ### Evaluate 

# In[ ]:


eval_steps = test_size // batch_size
test_data_iterator = data_loader.data_iterator(test_data)


# In[ ]:


config_path = os.path.join(bert_model_dir, 'bert_config.json')
config = BertConfig.from_json_file(config_path)model = BertForTokenClassification(config, num_labels=len(tag2idx))


# In[ ]:


load_checkpoint(os.path.join(model_dir, 'best.pth.tar'), model)


# In[ ]:


print("Starting evaluation...")
test_metrics = evaluate(model, test_data_iterator, mark='Test')



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




