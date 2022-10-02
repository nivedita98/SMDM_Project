#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset


# In[2]:


# !pip install -Iv transformers==3.4.0


# In[3]:


import numpy as np
import pandas as pd
import csv
import re
from transformers import AutoTokenizer, AutoModel
from transformers import AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput


# In[4]:


from google.colab import drive
drive.mount('/content/gdrive/')


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


# In[6]:


import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stop= set(stopwords.words('english'))


# In[7]:


device = torch.device('cuda')


# In[8]:


nltk.download('punkt')


# In[9]:


get_ipython().system('pip install gensim')


# In[10]:


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import pickle
import gensim


# In[11]:


train_c = pd.read_csv("/content/gdrive/MyDrive/SentimentAnalysis/train.csv",engine='python',encoding='latin')
train_df = pd.read_csv("/content/gdrive/MyDrive/SentimentAnalysis/SemEval2016.csv",engine='python',encoding='latin')
test_df = pd.read_csv("/content/gdrive/MyDrive/SentimentAnalysis/test.csv",engine='python',encoding='latin')


# In[12]:


print(train_df.head())
print(test_df.head())
print(train_df.shape)
print(test_df.shape)


# In[13]:


train_df=train_df.drop(columns=['Opinion towards','Worker ID', 'Instance ID'])
train_df=train_df.dropna()
train_df
train_c=train_c.drop(columns=['Opinion Towards', 'Sentiment' ])
train_c=train_c.dropna()
train_c
train_c = train_c[train_c["Target"] == "Climate Change is a Real Concern"]
train_df=pd.concat([train_df,train_c],axis=0)


# In[14]:


test_df=test_df.drop(columns=['Opinion Towards', 'Sentiment'])
test_df=test_df.dropna()
test_df.head()


# In[15]:


val_df = train_df[train_df["Target"] == "Hillary Clinton"]
test_df = test_df[test_df["Target"] == "Donald Trump"]
train_df = train_df[train_df["Target"] != "Hillary Clinton"]
train_df = train_df[train_df["Target"] != "Donald Trump"]


# In[16]:


print(train_df.shape)
print(test_df.shape)
print(val_df.shape)


# In[17]:


print(train_df.head())
train_df=train_df[train_df['Stance']!='NEUTRAL']
val_df=val_df[val_df['Stance']!='NEUTRAL']
print(test_df.head())
print(train_df.shape)
print(train_df["Stance"].value_counts())


# In[18]:


print(train_df['Stance'].value_counts())
print(test_df['Stance'].value_counts())
print(val_df['Stance'].value_counts())


# In[19]:


remove_against = 16244 - 7364


# In[20]:


against_df= train_df[train_df["Stance"]=="AGAINST"]
none_df=train_df[train_df["Stance"] == "NONE"]
favor_df =train_df[train_df["Stance"]=="FAVOR"]


# In[21]:


against_drop_indices=np.random.choice(against_df.index, remove_against, replace=False)
against_undersampled=against_df.drop(against_drop_indices)


# In[22]:


against_undersampled


# In[23]:


balanced_train_df= pd.concat([favor_df, against_undersampled, none_df])


# In[24]:


balanced_train_df["Stance"].value_counts()


# In[25]:


balanced_train_df["Stance"] = balanced_train_df["Stance"].map({
    "AGAINST" : [1, 0, 0],
    "FAVOR" : [0, 1, 0],
    "NONE" : [0, 0, 1]
})


# In[26]:


test_df["Stance"] = test_df["Stance"].map({
    "AGAINST" : [1, 0, 0],
    "FAVOR" : [0, 1, 0],
    "NONE" : [0, 0, 1]
})


# In[27]:


val_df["Stance"] = val_df["Stance"].map({
    "AGAINST" : [1, 0, 0],
    "FAVOR" : [0, 1, 0],
    "NONE" : [0, 0, 1]
})


# In[28]:


val_df


# In[29]:


balanced_train_df.shape


# In[30]:


train_set= list(balanced_train_df.to_records(index=False))
test_set = list(test_df.to_records(index=False))
val_set = list(val_df.to_records(index=False))


# In[31]:


def remove_links_mentions(word):
    link_re_pattern = "https?:\/\/t.co/[\w]+"
    mention_re_pattern = "@\w+"
    word = re.sub(link_re_pattern, "", word)
    word = re.sub(mention_re_pattern, "", word)
    word= re.sub(r'[?|!|\'|"|#]',r'',word)
    word= re.sub(r',[.|,|)|(|\|/]',r' ',word)
    word=word.replace('_',' ')
    return word.lower()


# In[32]:


cleaned_train_tweet=[]
cleaned_train_target=[]
for tweet, target,stance in train_set:
    cleaned_train_tweet.append(remove_links_mentions(tweet))

for tweet, target,stance in train_set:
    if target in cleaned_train_target:
        continue
    else:
        cleaned_train_target.append(remove_links_mentions(target))  


# In[33]:


cleaned_test_tweet=[]
cleaned_test_target=[]
for tweet, target,stance in test_set:
    cleaned_test_tweet.append(remove_links_mentions(tweet))

for tweet, target,stance in test_set:
    if (target) in cleaned_test_target:
        continue
    else:
        cleaned_test_target.append(remove_links_mentions(target)) 


# In[34]:


cleaned_val_tweet=[]
cleaned_val_target=[]
for tweet, target,stance in val_set:
    cleaned_val_tweet.append(remove_links_mentions(tweet))

for tweet, target,stance in val_set:
    if (target) in cleaned_val_target:
        continue
    else:
        cleaned_val_target.append(remove_links_mentions(target))


# In[35]:


cleaned_train_target=list(cleaned_train_target)
print(len(cleaned_train_target))
print(len(cleaned_train_tweet))
print(cleaned_train_tweet)


# In[36]:


cleaned_test_target=list(cleaned_test_target)
print(len(cleaned_test_target))
print(len(cleaned_test_tweet))
print(cleaned_test_tweet)


# In[37]:


cleaned_val_target=list(cleaned_val_target)
print(len(cleaned_val_target))
print(len(cleaned_val_tweet))
print(cleaned_val_tweet)


# In[38]:


print(balanced_train_df['Stance'])


# In[39]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


# In[40]:


train_tokens = tokenizer(cleaned_train_tweet,cleaned_train_target,
                   padding = True,
                   truncation = True,
                   max_length = 100,
                   return_tensors = 'pt',
                   return_token_type_ids = False)

val_tokens = tokenizer(cleaned_val_tweet,cleaned_val_target,
                   padding = True,
                   truncation = True,
                   max_length = 100,
                   return_tensors = 'pt',
                   return_token_type_ids = False)


# In[41]:


test_stance=pd.DataFrame.from_records(test_df["Stance"].to_numpy()).to_numpy()
train_stance=pd.DataFrame.from_records(balanced_train_df["Stance"].to_numpy()).to_numpy()
val_stance=pd.DataFrame.from_records(val_df["Stance"].to_numpy()).to_numpy()
print(train_stance.shape)


# In[42]:


train_tokens


# In[43]:


from torch.utils.data import DataLoader,TensorDataset, RandomSampler, SequentialSampler, Dataset

class Twitter_Dataset(Dataset):
    def __init__(self, tokens, labels):
        self.input_ids = tokens['input_ids']
        self.attentions = tokens['attention_mask']
        self.label = torch.tensor(labels)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        tokens = self.input_ids[idx]
        attention = self.attentions[idx]
        sample = {"Tokens": tokens,"Masks":attention, "Class": label}
        return sample


# In[44]:


Train = Twitter_Dataset(train_tokens,train_stance)
Validation = Twitter_Dataset(val_tokens,val_stance)
train_loader = DataLoader(Train,batch_size = 10, shuffle = True)
val_loader = DataLoader(Validation, batch_size = 10, shuffle = False)


# In[45]:


class CustomModel(nn.Module):
    def __init__(self,checkpoint,num_labels): 
        super(CustomModel,self).__init__() 
        self.num_labels = num_labels 

        #Load Model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained(checkpoint,config=AutoConfig.from_pretrained(checkpoint, output_attentions=True,output_hidden_states=True))
        self.dropout = nn.Dropout(0.1) 
        self.classifier1 = nn.Linear(768,300)
        self.classifier2 = nn.Linear(300,num_labels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)# load and initialize weights

    def forward(self, input_ids=None, attention_mask=None,labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        #Add custom layers
        sequence_output = self.dropout(outputs[0]) #outputs[0]=last hidden state

        class1 = self.classifier1(sequence_output[:,0,:].view(-1,768))
        a1 = self.relu(class1)
        class2 = self.classifier2(a1)
        logits = self.softmax(class2)
        # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)


# In[46]:


model = CustomModel("cardiffnlp/twitter-roberta-base-sentiment",3)


# In[47]:


model.to(device)


# In[48]:


device = torch.device('cuda')


# In[49]:


for param in model.model.parameters():
    param.requires_grad = True
model.to(device)


# In[1]:


from transformers import AdamW

optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
device = "cuda"
for epoch in range(20):
    epoch_loss=0
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['Tokens'].to(device)
        attention_mask = batch['Masks'].to(device)
        label = batch['Class'].to(device)
        outputs = model(input_ids,attention_mask,label.argmax(dim = 1))
        loss = outputs.loss
        batch_loss=loss.item()
        optim.step()
        epoch_loss+=batch_loss
    normalized_epoch_loss = epoch_loss/(len(train_loader))
    print("Epoch {} ; Epoch loss: {} ".format(epoch+1,normalized_epoch_loss))


# In[51]:


label.shape


# In[52]:


from torch.utils.data import DataLoader,TensorDataset, RandomSampler, SequentialSampler, Dataset

class Twitter_Test_Dataset(Dataset):
    def __init__(self, tokens):
        self.input_ids = tokens['input_ids']
        self.attentions = tokens['attention_mask']

    def __len__(self):
        return len(self.attentions)

    def __getitem__(self, idx):
        tokens = self.input_ids[idx]
        attention = self.attentions[idx]
        sample = {"Tokens": tokens,"Masks":attention}
        return sample


# In[53]:


Test_tokens = tokenizer(cleaned_test_tweet,cleaned_test_target,
                   padding = True,
                   truncation = True,
                   max_length = 100,
                   return_tensors = 'pt',
                   return_token_type_ids = False)


# In[54]:


test = Twitter_Test_Dataset(Test_tokens)
test_loader = DataLoader(test,batch_size = 10, shuffle = False)


# In[55]:


model.eval()
preds = []
with torch.no_grad():
    for batch in test_loader:
        pred = model(batch['Tokens'].to(device),batch['Masks'].to(device))
        preds.extend(pred.logits.tolist())


# In[56]:


from sklearn.metrics import f1_score

print(100*f1_score(np.array(preds).argmax(1), test_stance.argmax(1), average ='weighted'))


# In[56]:





# In[56]:




