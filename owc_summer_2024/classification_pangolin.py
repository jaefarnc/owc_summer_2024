#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch

print(torch.cuda.is_available())

base = "pangolin"


# In[11]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator


# In[12]:


from huggingface_hub import login
login(token = "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")


# In[13]:


#pip install datasets


# In[14]:


import torch
from transformers import pipeline

accelerator = Accelerator()

model = "facebook/bart-large-mnli"
# Load the zero-shot class
device = 0
classifier = pipeline("zero-shot-classification", model=model, device = accelerator.device)


# In[15]:


import pandas as pd
from datasets import Dataset
df = pd.read_csv(f"{base}/{base}_tweets.csv")
print(len(df))


# In[ ]:


df['tweet_content']=df['tweet_content'].astype('str').fillna("")

dataset = Dataset.from_pandas(df)
#labels = ["video gamer", "not video gamer"]
#labels = ["anime","gaming","wildlife welfare","suspicious wildlife criminal"]
labels = ['suspicious wildlife criminal', 'not suspicious wildlife criminal']
def classify_batch(batch):
    sequences = batch['tweet_content']
    results = classifier(sequences, candidate_labels=labels)
    scores = [result['scores'][result['labels'].index(labels[0])] for result in results]
    return {"classification":scores}

batch_size = 64
results = dataset.map(classify_batch, batched=True, batch_size=batch_size)  # Use df directly
result_df = results.to_pandas()



# In[ ]:


#filtered_df = result_df[result_df['classification']>=0.95]
#filtered_df = filtered_df[filtered_df['classification']<0.7]
#print(len(filtered_df))


# In[ ]:


# df = filtered_df.copy()
# df['tweet_content']=df['tweet_content'].astype('str').fillna("")

# dataset = Dataset.from_pandas(df)
# labels = ["suspicious wildlife criminal", "not suspicious wildlife criminal"]

# def classify_batch(batch):
#     sequences = batch['tweet_content']
#     results = classifier(sequences, candidate_labels=labels)
#     scores = [result['scores'][result['labels'].index(labels[0])] for result in results]
#     return {"classification":scores}

# batch_size = 64
# results = dataset.map(classify_batch, batched=True, batch_size=batch_size)  # Use df directly
# result_df = results.to_pandas()

# filtered_df = result_df[result_df['classification']>=0.9]


# In[ ]:


# filtered_df.to_csv(f'{base}/{base}_filtered_tweets.csv')
#filtered_df.to_csv('temp.csv)


# In[ ]:


result_df.to_csv(f'{base}/{base}_tweets_classifier0.csv')


# In[ ]:




