#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset


# In[2]:


import numpy as np
import pandas as pd
import csv
import re


# In[3]:


from google.colab import drive
drive.mount('/content/gdrive/')


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


# In[5]:


import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stop= set(stopwords.words('english'))


# In[6]:


device = torch.device('cuda')


# In[7]:


nltk.download('punkt')


# In[8]:


get_ipython().system('pip install gensim')


# In[9]:


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import pickle
import gensim


# In[12]:


model=KeyedVectors.load_word2vec_format("/content/gdrive/MyDrive/SentimentAnalysis/GoogleNews-vectors-negative300.bin",binary=True)
# model=api.load("/content/gdrive/MyDrive/Colab Notebooks/SentimentAnalysis/GoogleNews-vectors-negative300.bin")


# In[14]:


train_df = pd.read_csv("/content/gdrive/MyDrive/SentimentAnalysis/train.csv",engine='python',encoding='latin')
test_df = pd.read_csv("/content/gdrive/MyDrive/SentimentAnalysis/test.csv",engine='python',encoding='latin')


# In[15]:


print(train_df.head())
print(test_df.head())
print(train_df.shape)
print(test_df.shape)


# In[16]:


train_df=train_df.drop(columns=['Opinion Towards', 'Sentiment'])
train_df=train_df.dropna()
train_df


# In[17]:


test_df=test_df.drop(columns=['Opinion Towards', 'Sentiment'])
test_df=test_df.dropna()
test_df.head()


# In[18]:


test_df = test_df[test_df["Target"] == "Donald Trump"]
val_df = train_df[train_df["Target"] == "Hillary Clinton"]
train_df = train_df[train_df["Target"] != "Hillary Clinton"]


# In[19]:


print(train_df.shape)
print(test_df.shape)
print(val_df.shape)


# In[20]:


print(train_df.head())
print(test_df.head())


# In[21]:


print(train_df['Stance'].value_counts())
print(test_df['Stance'].value_counts())
print(val_df['Stance'].value_counts())


# In[22]:


remove_against = 1002- 588


# In[23]:


against_df= train_df[train_df["Stance"]=="AGAINST"]
none_df=train_df[train_df["Stance"] == "NONE"]
favor_df =train_df[train_df["Stance"]=="FAVOR"]


# In[24]:


against_drop_indices=np.random.choice(against_df.index, remove_against, replace=False)
against_undersampled=against_df.drop(against_drop_indices)


# In[25]:


against_undersampled


# In[26]:


balanced_train_df= pd.concat([favor_df, against_undersampled, none_df])


# In[27]:


balanced_train_df["Stance"].value_counts()


# In[28]:


balanced_train_df["Stance"] = balanced_train_df["Stance"].map({
    "AGAINST" : [1, 0, 0],
    "FAVOR" : [0, 1, 0],
    "NONE" : [0, 0, 1]
})


# In[29]:


test_df["Stance"] = test_df["Stance"].map({
    "AGAINST" : [1, 0, 0],
    "FAVOR" : [0, 1, 0],
    "NONE" : [0, 0, 1]
})


# In[30]:


val_df["Stance"] = val_df["Stance"].map({
    "AGAINST" : [1, 0, 0],
    "FAVOR" : [0, 1, 0],
    "NONE" : [0, 0, 1]
})


# In[31]:


val_df


# In[32]:


balanced_train_df.shape


# In[33]:


train_set= list(balanced_train_df.to_records(index=False))
test_set = list(test_df.to_records(index=False))
val_set = list(val_df.to_records(index=False))


# In[34]:


def remove_links_mentions(word):
    link_re_pattern = "https?:\/\/t.co/[\w]+"
    mention_re_pattern = "@\w+"
    word = re.sub(link_re_pattern, "", word)
    word = re.sub(mention_re_pattern, "", word)
    word= re.sub(r'[?|!|\'|"|#]',r'',word)
    word= re.sub(r',[.|,|)|(|\|/]',r' ',word)
    word=word.replace('_',' ')
    return word.lower()


# In[35]:


# !pip install -U sentence-transformers


# In[36]:


cleaned_train_tweet=[]
cleaned_train_target=[]
for tweet, target,stance in train_set:
    cleaned_train_tweet.append(remove_links_mentions(tweet))

for tweet, target,stance in train_set:
    if target in cleaned_train_target:
        continue
    else:
        cleaned_train_target.append(remove_links_mentions(target))        


# In[37]:


cleaned_test_tweet=[]
cleaned_test_target=[]
for tweet, target,stance in test_set:
    cleaned_test_tweet.append(remove_links_mentions(tweet))

for tweet, target,stance in test_set:
    if (target) in cleaned_test_target:
        continue
    else:
        cleaned_test_target.append(remove_links_mentions(target)) 


# In[38]:


cleaned_val_tweet=[]
cleaned_val_target=[]
for tweet, target,stance in val_set:
    cleaned_val_tweet.append(remove_links_mentions(tweet))

for tweet, target,stance in val_set:
    if (target) in cleaned_val_target:
        continue
    else:
        cleaned_val_target.append(remove_links_mentions(target))


# In[39]:


cleaned_train_target=list(cleaned_train_target)
print(len(cleaned_train_target))
print(len(cleaned_train_tweet))
print(cleaned_train_tweet)


# In[40]:


cleaned_test_target=list(cleaned_test_target)
print(len(cleaned_test_target))
print(len(cleaned_test_tweet))
print(cleaned_test_tweet)


# In[41]:


cleaned_val_target=list(cleaned_val_target)
print(len(cleaned_val_target))
print(len(cleaned_val_tweet))
print(cleaned_val_tweet)


# In[42]:


print(balanced_train_df['Stance'])


# In[43]:


# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# embeddings = model.encode(cleaned_train_tweet)


# In[44]:


# embeddings_target = model.encode(cleaned_train_target)


# In[45]:


# embeddings_test_tweet = model.encode(cleaned_test_tweet)


# In[46]:


# embeddings_test_target = model.encode(cleaned_test_target)


# In[47]:


# print(embeddings_test_target.shape)


# In[48]:


# print(embeddings.shape) 


# In[49]:


import nltk
import time
from nltk.corpus import stopwords
from nltk import word_tokenize
import joblib
nltk.download('stopwords')


# In[50]:


stop_words = set(stopwords.words('english'))
i = 0
cleaned_tweet_token = []
for data in cleaned_train_tweet:
    new_text = []
    data = word_tokenize(data)
    #print(data)
    for word in data:
        if word not in stop_words:
            new_text.append(word)  
    #tokens = word_tokenize(new_text)
    cleaned_tweet_token.insert(i,new_text)
    i = i+1    


# In[51]:


stop_words = set(stopwords.words('english'))
i = 0
cleaned_target_token = []
for data in cleaned_train_target:
    new_text = []
    data = word_tokenize(data)
    #print(data)
    for word in data:
        if word not in stop_words:
            new_text.append(word)  
    #tokens = word_tokenize(new_text)
    cleaned_target_token.insert(i,new_text)
    i = i+1   


# In[52]:


print(cleaned_target_token)


# In[53]:


stop_words = set(stopwords.words('english'))
i = 0
cleaned_tweet_token_test = []
for data in cleaned_test_tweet:
    new_text = []
    data = word_tokenize(data)
    #print(data)
    for word in data:
        if word not in stop_words:
            new_text.append(word)  
    #tokens = word_tokenize(new_text)
    cleaned_tweet_token_test.insert(i,new_text)
    i = i+1    


# In[54]:


stop_words = set(stopwords.words('english'))
i = 0
cleaned_tweet_token_val = []
for data in cleaned_val_tweet:
    new_val = []
    data = word_tokenize(data)
    #print(data)
    for word in data:
        if word not in stop_words:
            new_val.append(word)  
    #tokens = word_tokenize(new_text)
    cleaned_tweet_token_val.insert(i,new_text)
    i = i+1    


# In[55]:


print(cleaned_tweet_token_val)


# In[56]:


stop_words = set(stopwords.words('english'))
i = 0
cleaned_target_token_test = []
for data in cleaned_test_target:
    new_text = []
    data = word_tokenize(data)
    #print(data)
    for word in data:
        if word not in stop_words:
            new_text.append(word)  
    #tokens = word_tokenize(new_text)
    cleaned_target_token_test.insert(i,new_text)
    i = i+1    


# In[57]:


stop_words = set(stopwords.words('english'))
i = 0
cleaned_target_token_val = []
for data in cleaned_val_target:
    new_val = []
    data = word_tokenize(data)
    #print(data)
    for word in data:
        if word not in stop_words:
            new_val.append(word)  
    #tokens = word_tokenize(new_text)
    cleaned_target_token_val.insert(i,new_text)
    i = i+1  


# In[58]:


print(cleaned_target_token_test)


# In[59]:


# !pip install --upgrade gensim


# In[60]:


print(model.vector_size)
train_tweet =[]
count=0
for words in cleaned_tweet_token:
    vector_list = [model[word] for word in words if word in model.vocab]
    train_tweet.append(vector_list)

from itertools import zip_longest
train_tweet = np.array(list(zip_longest(*train_tweet, fillvalue=np.zeros(300)))).T
train_tweet = train_tweet.transpose([1, 2, 0])
# train_tweet = np.array(train_tweet)
print(train_tweet.shape)


# In[61]:


train_target =[]
count=0
for words in cleaned_target_token:
    vector_list = [model[word] for word in words if word in model.vocab]
    train_target.append(vector_list)

from itertools import zip_longest
train_target = np.array(list(zip_longest(*train_target, fillvalue=np.zeros(300)))).T
train_target = train_target.transpose([1, 2, 0])
# train_tweet = np.array(train_tweet)
print(train_target.shape)


# In[62]:


test_tweet =[]
count=0
for words in cleaned_tweet_token_test:
    vector_list = [model[word] for word in words if word in model.vocab]
    test_tweet.append(vector_list)

from itertools import zip_longest
test_tweet = np.array(list(zip_longest(*test_tweet, fillvalue=np.zeros(300)))).T
test_tweet = test_tweet.transpose([1, 2, 0])
# train_tweet = np.array(train_tweet)
print(test_tweet.shape)


# In[63]:


val_tweet =[]
count=0
for words in cleaned_tweet_token_val:
    vector_list = [model[word] for word in words if word in model.vocab]
    val_tweet.append(vector_list)

from itertools import zip_longest
val_tweet = np.array(list(zip_longest(*val_tweet, fillvalue=np.zeros(300)))).T
val_tweet = val_tweet.transpose([1, 2, 0])
# train_tweet = np.array(train_tweet)
print(val_tweet.shape)


# In[64]:


test_target =[]
count=0
for words in cleaned_target_token_test:
    vector_list = [model[word] for word in words if word in model.vocab]
    test_target.append(vector_list)

from itertools import zip_longest
test_target = np.array(list(zip_longest(*test_target, fillvalue=np.zeros(300)))).T
test_target = test_target.transpose([1, 2, 0])
# train_tweet = np.array(train_tweet)
print(test_target.shape)


# In[65]:


val_target =[]
count=0
for words in cleaned_target_token_val:
    vector_list = [model[word] for word in words if word in model.vocab]
    val_target.append(vector_list)

from itertools import zip_longest
val_target = np.array(list(zip_longest(*val_target, fillvalue=np.zeros(300)))).T
val_target = val_target.transpose([1, 2, 0])
# train_tweet = np.array(train_tweet)
print(val_target.shape)


# In[66]:


# train_set[0][0]
# list_of_sent_tweet=[]
# for i in range(len(train_set)):
#         list_of_sent_tweet.append(train_set[i][0])
# print(list_of_sent_tweet)  


# In[67]:


# print(train_set[0][0])
# list_of_sent_target=[]
# for i in range(len(train_set)):
#         list_of_sent_target.append(train_set[i][1])
# print(list_of_sent_target)


# In[68]:


# w2v_model_tweet=gensim.models.Word2Vec(list_of_sent_tweet,min_count=1,size=50)
# w2v_model_target=gensim.models.Word2Vec(list_of_sent_target,min_count=1,size=50)


# In[69]:


# words_tweet1=list(w2v_model_tweet.wv.vocab)
# words_tweet=[]
# for i in words_tweet1:
#     if i not in stop:
#         words_tweet.append(i)
# print(len(words_tweet))


# In[70]:


# words_target1=list(w2v_model_target.wv.vocab)
# words_target=[]
# for i in words_target1:
#     if i not in stop:
#         words_target.append(i)
# print(len(words_target))


# In[71]:


# word2index ={token : idx for idx, token in enumerate(total_words)}
# print(type(word2index))


# In[72]:


test_stance=pd.DataFrame.from_records(test_df["Stance"].to_numpy()).to_numpy()
train_stance=pd.DataFrame.from_records(balanced_train_df["Stance"].to_numpy()).to_numpy()
val_stance=pd.DataFrame.from_records(val_df["Stance"].to_numpy()).to_numpy()
print(train_stance.shape)


# In[73]:


batch_size = 100


# In[74]:


print(train_tweet.shape)
print(train_target.shape)
print(train_stance.shape)


# In[75]:


train_ds = TensorDataset(torch.from_numpy(train_tweet), torch.from_numpy(train_target), torch.from_numpy(train_stance))
test_ds = TensorDataset(torch.from_numpy(test_tweet), torch.from_numpy(test_target), torch.from_numpy(test_stance))
val_ds = TensorDataset(torch.from_numpy(val_tweet), torch.from_numpy(val_target), torch.from_numpy(val_stance))


# In[76]:


train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size, drop_last=True)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, drop_last=True)
print(len(train_dl.dataset))


# In[77]:


class LSTM_target(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim) :
        super().__init__()

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state 
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        # embs = self.embedding(x)

        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hncn = self.lstm(x, hidden)
        hn, cn = hncn
        # cn.shape = (2, batch_size, hidden_dim)
        # print(cn)
        return cn

    def init_hidden(self):
        return (torch.zeros(2, batch_size, hidden_dim), torch.zeros(2, batch_size, hidden_dim))


# In[78]:


class LSTM_twitter(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim) :
        super().__init__()

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.W = nn.Linear(2*hidden_dim, 3, bias=False)
        
    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state 
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        # embs = self.embedding(x)
        #cn=lstm_target.forward() 
        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        out, hncn = self.lstm(x, hidden)
        hn, cn = hncn
        hn = hn.reshape(batch_size, 2*hidden_dim)  # convert hn from shape [2, h] to [2*h, ]
        c = self.W(hn)  # hn @ welf.W or hn.cat() @ self.W 
        #return F.softmax(F.tanh(c))
        return torch.tanh(c)

    def init_hidden(self):
        return torch.zeros(2, batch_size, hidden_dim)


# In[79]:


# model = BiLSTM_SentimentAnalysis(len(word2index), 64, 32, 0.2)
# model = model.to(device)
embedding_dim, hidden_dim =  300, 32
lstm_target = LSTM_target( embedding_dim, hidden_dim)
lstm_twitter = LSTM_twitter(embedding_dim, hidden_dim)


# In[80]:


lstm_target.to(device)
lstm_twitter.to(device)


# In[80]:





# In[81]:


criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(lstm_target.parameters(),lr = 0.1)
optimizer2 = torch.optim.Adam(lstm_twitter.parameters(), lr = 0.1)


# In[82]:


from sklearn.metrics import f1_score
epochs = 5000
losses = []
for e in range(epochs):

    loss1 = 0
    num = 0
    for batch_idx, batch in enumerate(train_dl):
        
        # print(h0.is_cuda,c0.is_cuda)
        
        tweet, target, stance = batch
       # print(len(batch), batch[0], batch[1], batch[2])
        # stance should be one hot encoded i.e. it stance.shape = (batch_size, 3)
        # input = batch[0].to(device)
        # target = batch[1].to(device)
        # target=torch.tensor(target.astype(np.int32))
        # tweet=torch.tensor(tweet.astype(np.int32))
        # stance=torch.tensor(stance.astype(np.int32))
        target = target.float().to(device)
        tweet = tweet.float().to(device)
        stance = stance.float().to(device)
        # print(target.is_cuda)
        with torch.set_grad_enabled(True):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # print(target.shape)
            # print(target.shape, h0.shape, c0.shape)
            h0, c0 =  lstm_target.init_hidden()
            h0 = h0.to(device)
            c0 = c0.to(device)
            ct = lstm_target(target, (h0, c0))

            ht = lstm_twitter.init_hidden()
            ht = ht.to(device)

            c = lstm_twitter(tweet, (ht, ct)) # doubt
            loss = criterion(c, stance.argmax(dim = 1))
            loss.backward()
            optimizer2.step()
            optimizer1.step()
            loss1+=loss.item()
            num += 1

    if e % 50 == 0:
        print("Epochs : {}, Loss : {}".format(e, loss1/num))

        losses.append(loss.item())

        total_length = 0
        total_right = 0
        for batch_idx, batch in enumerate(val_dl):

            tweet, target, stance = batch
            target = target.float().to(device)
            tweet = tweet.float().to(device)
            stance = stance.float().to(device)

            c = lstm_twitter(tweet, (h0, ct)) # doubt
            total_length +=100
            total_right += f1_score(c.argmax(dim = 1).cpu().int().numpy(), stance.argmax(dim = 1).cpu().int().numpy(), average = 'weighted')
        

        print("The f1_score for the validation data is ", 100*(total_right/total_length))

    

    


# In[86]:


from sklearn.metrics import f1_score
batch_acc = []
total_length = 0
total_right = 0
new = []
for batch_idx, batch in enumerate(test_dl):

        h0, c0 =  lstm_target.init_hidden()
        h0 = h0.to(device)
        c0 = c0.to(device)
        tweet, target, stance = batch
        target = target.float().to(device)
        tweet = tweet.float().to(device)
        stance = stance.float().to(device)
        cnt = 0
        with torch.set_grad_enabled(False):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # print(target.shape, h0.shape, c0.shape)
            ct = lstm_target(target, (h0, c0))

            ht = lstm_twitter.init_hidden()
            ht = ht.to(device)

            c = lstm_twitter(tweet, (h0, ct)) # doubt
            total_length +=100
            total_right += f1_score(c.argmax(dim = 1).cpu().int().numpy(), stance.argmax(dim = 1).cpu().int().numpy(), average = "weighted")
            print(100*f1_score(c.argmax(dim = 1).cpu().int().numpy(), stance.argmax(dim = 1).cpu().int().numpy(), average = "weighted"))
            # loss = criterion(c, stance.float())
print("The f1_score for the test data is ", 100*(total_right/total_length))

