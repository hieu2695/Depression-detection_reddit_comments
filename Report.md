<h1 style = 'text-align: center;'>DATS 6312 Project</h1>
<h1 style = 'text-align: center;'>Depression Identification</h1>
<h3 style = 'text-align: center;'>Team 5: Tran Hieu Le, Voratham Tiabrat, Trinh Vu</h3>

<h1>Table of Contents<span class="tocSkip"></span></h1>

<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span>

<li><span><a href="#Objective-and-Scope" data-toc-modified-id="Objective-and-Scope-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Objective and Scope</a></span></li>
    
<li><span><a href="#Dataset" data-toc-modified-id="Dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Dataset</a></span></li>
    
    

<li><span><a href="#Data-Preprocessing" data-toc-modified-id="Data-Preprocessing-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Preprocessing</a></span><ul class="toc-item"><li><span><a href="#Loading-libraries" data-toc-modified-id="Loading-libraries-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Loading libraries</a></span></li><li><span><a href="#Loading-data" data-toc-modified-id="Loading-data-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Loading data</a></span></li><li><span><a href="#Text-preprocessing" data-toc-modified-id="Text-preprocessing-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Text preprocessing</a></span></li><li><span><a href="#Overview-of-the-data" data-toc-modified-id="Overview-of-the-data-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Overview of the data</a></span></li></ul></li>


<li><span><a href="#Word-embedding" data-toc-modified-id="Word-embedding-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Word embedding</a></span><ul class="toc-item"><li><span><a href="#Creating-a-vocabulary" data-toc-modified-id="Creating-a-vocabulary-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Creating a vocabulary</a></span></li><ul class="toc-item"><li><span><a href="#Getting-the-size-of-the-vocabulary" data-toc-modified-id="Getting-the-size-of-the-vocabulary-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Getting the size of the vocabulary</a></span></li></ul><li><span><a href="#Removing-words-that-are-not-in-the-vocabulary" data-toc-modified-id="Removing-words-that-are-not-in-the-vocabulary-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Removing words that are not in the vocabulary</a></span></li><li><span><a href="#Splitting-the-training,-validation-and-testing-data" data-toc-modified-id="Splitting-the-training,-validation-and-testing-data-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Splitting the training, validation and testing data</a></span></li><li><span><a href="#Creating-a-list-to-store-comments" data-toc-modified-id="Creating-a-list-to-store-comments-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Creating a list to store comments</a></span></li><li><span><a href="#Keras-embedding-layer" data-toc-modified-id="Keras-embedding-layer-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Keras embedding layer</a></span></li><ul class="toc-item"><li><span><a href="#Vectorizing-the-training,-validation-and-testing-comments" data-toc-modified-id="Vectorizing-the-training,-validation-and-testing-comments-5.1.1"><span class="toc-item-num">5.5.1&nbsp;&nbsp;</span>Vectorizing the training, validation and testing comments</a></span></li></ul><li><span><a href="#Pre-trained-Word2Vec-layer" data-toc-modified-id="Pre-trained-Word2Vec-layer-5.6"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>Pre-trained Word2Vec layer</a></span><ul class="toc-item"><li><span><a href="#Train-the-Word2Vec-model" data-toc-modified-id="Train-the-Word2Vec-model-5.6.1"><span class="toc-item-num">5.6.1&nbsp;&nbsp;</span>Train the Word2Vec model</a></span></li><li><span><a href="#Saving-the-Word2Vec-embedding-in-a-.txt-file" data-toc-modified-id="Saving-the-Word2Vec-embedding-in-a-.txt-file-5.6.2"><span class="toc-item-num">5.6.2&nbsp;&nbsp;</span>Saving the Word2Vec embedding in a .txt file</a></span></li><li><span><a href="#Loading-the-Word2Vec-embedding" data-toc-modified-id="Loading-the-Word2Vec-embedding-5.6.3"><span class="toc-item-num">5.6.3&nbsp;&nbsp;</span>Loading the Word2Vec embedding</a></span></li><li><span><a href="#Mapping-the-embedding-vectors" data-toc-modified-id="Mapping-the-embedding-vectors-5.6.4"><span class="toc-item-num">5.6.4&nbsp;&nbsp;</span>Mapping the embedding vectors</a></span></li></ul></li></ul></li>

<li><span><a href="#Building-Neural-Network-Models" data-toc-modified-id="Building-Neural-Network-Models-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Building Neural Network Models</a></span><ul class="toc-item"><li><span><a href="#Setting-callbacks" data-toc-modified-id="Setting-callbacks-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Setting callbacks</a></span></li><li><span><a href="#CNN-and-Keras-embedding-layer" data-toc-modified-id="CNN-and-Keras-embedding-layer-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>CNN and Keras embedding layer</a></span><ul class="toc-item"><li><span><a href="#Building-architecture-(CNN-and-Keras)" data-toc-modified-id="Building-architecture-(CNN-and-Keras)-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>Building architecture</a></span></li><li><span><a href="#Compiling-and-training-the-model-(CNN-and-Keras)" data-toc-modified-id="Compiling-and-training-the-model-(CNN-and-Keras)-6.2.2"><span class="toc-item-num">6.2.2&nbsp;&nbsp;</span>Compiling and training the model</a></span></li><li><span><a href="#Plotting-the-learning-curve-(CNN-and-Keras)" data-toc-modified-id="Plotting-the-learning-curve-(CNN-and-Keras)-6.2.3"><span class="toc-item-num">6.2.3&nbsp;&nbsp;</span>Plotting the learning curve</a></span></li><li><span><a href="#Model-evaluation-(CNN-and-Keras)" data-toc-modified-id="Model-evaluation-(CNN-and-Keras)-6.2.4"><span class="toc-item-num">6.2.4&nbsp;&nbsp;</span>Model evaluation</a></span></li></ul></li><li><span><a href="#RNN-and-Keras-embedding-layer" data-toc-modified-id="RNN-and-Keras-embedding-layer-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>RNN and Keras embedding layer</a></span><ul class="toc-item"><li><span><a href="#Building-architecture-(RNN-and-Keras)" data-toc-modified-id="Building-architecture-(RNN-and-Keras)-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>Building architecture</a></span></li><li><span><a href="#Compiling-and-training-the-model-(RNN-and-Keras)" data-toc-modified-id="Compiling-and-training-the-model-(RNN-and-Keras)-6.3.2"><span class="toc-item-num">6.3.2&nbsp;&nbsp;</span>Compiling and training the model</a></span></li><li><span><a href="#Plotting-the-learning-curve-(RNN-and-Keras)" data-toc-modified-id="Plotting-the-learning-curve-(RNN-and-Keras)-6.3.3"><span class="toc-item-num">6.3.3&nbsp;&nbsp;</span>Plotting the learning curve</a></span></li><li><span><a href="#Model-evaluation-(RNN-and-Keras)" data-toc-modified-id="Model-evaluation-(RNN-and-Keras)-6.3.4"><span class="toc-item-num">6.3.4&nbsp;&nbsp;</span>Model evaluation</a></span></li></ul></li><li><span><a href="#CNN-and-Word2Vec" data-toc-modified-id="CNN-and-Word2Vec-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>CNN and Word2Vec</a></span><ul class="toc-item"><li><span><a href="#Building-architecture-(CNN-and-Word2Vec)" data-toc-modified-id="Building-architecture-(CNN-and-Word2Vec)-6.4.1"><span class="toc-item-num">6.4.1&nbsp;&nbsp;</span>Building architecture</a></span></li><li><span><a href="#Compiling-and-training-the-model-(CNN-and-Word2Vec)" data-toc-modified-id="Compiling-and-training-the-model-(CNN-and-Word2Vec)-6.4.2"><span class="toc-item-num">6.4.2&nbsp;&nbsp;</span>Compiling and training the model</a></span></li><li><span><a href="#Plotting-the-learning-curve-(CNN-and-Word2Vec)" data-toc-modified-id="Plotting-the-learning-curve-(CNN-and-Word2Vec)-6.4.3"><span class="toc-item-num">6.4.3&nbsp;&nbsp;</span>Plotting the learning curve</a></span></li><li><span><a href="#Model-evaluation-(CNN-and-Word2Vec)" data-toc-modified-id="Model-evaluation-(CNN-and-Word2Vec)-6.4.4"><span class="toc-item-num">6.4.4&nbsp;&nbsp;</span>Model evaluation</a></span></li></ul></li><li><span><a href="#RNN-and-Word2Vec" data-toc-modified-id="RNN-and-Word2Vec-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>RNN and Word2Vec</a></span><ul class="toc-item"><li><span><a href="#Building-architecture-(RNN-and-Word2Vec)" data-toc-modified-id="Building-architecture-(RNN-and-Word2Vec)-6.5.1"><span class="toc-item-num">6.5.1&nbsp;&nbsp;</span>Building architecture</a></span></li><li><span><a href="#Compiling-and-training-the-model-(RNN-and-Word2Vec)" data-toc-modified-id="Compiling-and-training-the-model-(RNN-and-Word2Vec)-6.5.2"><span class="toc-item-num">6.5.2&nbsp;&nbsp;</span>Compiling and training the model</a></span></li><li><span><a href="#Plotting-the-learning-curve-(RNN-and-Word2Vec)" data-toc-modified-id="Plotting-the-learning-curve-(RNN-and-Word2Vec)-6.5.3"><span class="toc-item-num">6.5.3&nbsp;&nbsp;</span>Plotting the learning curve</a></span></li><li><span><a href="#Model-evaluation-(RNN-and-Word2Vec)" data-toc-modified-id="Model-evaluation-(RNN-and-Word2Vec)-6.5.4"><span class="toc-item-num">6.5.4&nbsp;&nbsp;</span>Model evaluation</a></span></li></ul></li></ul></li>


<li><span><a href="#Result-and-Discussion" data-toc-modified-id="Result-and-Discussion-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Result and Discussion</a></span></li>
    
<li><span><a href="#Challenges" data-toc-modified-id="Challenges-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Challenges</a></span></li>

 
<li><span><a href="#Future-Work" data-toc-modified-id="Future-Work-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Future Work</a></span></li> 
    
<li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Conclusion</a></span></li> 
    
    
<li><span><a href="#References" data-toc-modified-id="References-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>References</a></span></li>
</ul>
</div>

# Introduction

<div style="text-align: justify">
Major depressive disorder or depression is a mental illness that negatively impacts an individual’s mood and activities leads to a persistent feeling of sadness and loss of interest. It is estimated that depression accounts for 4.3% of the global burden of disease and 1 million suicides every year. Despite these detrimental effects, the symptoms of depression can only be diagnosed after two weeks. Therefore, a method to detect early depression is a vital improvement in addressing this mental disease.</div>

<div style="text-align: justify">
<br/>
In recent years, social media platforms such as Facebook, Twitter and Reddit have become an important part in mental health support. It is stated that social media platforms have been increasingly used by individuals to connect with others, share experiences, ease stress, anxiety, and depression, boost self-worth, provide comfort and joy, and prevent loneliness. Many forums related to mental health support have been created and attracted a lot of users. Analyzing the sentiment in comments and posts on such platforms can provide insights about an individual's feeling and health status, which helps to increase the likelihood to identify mental illness including depression.</div>

<div style="text-align: justify">
<br/> 
There have been many studies that attempt to identify the depression using social media platforms such as “Utilizing Neural Networks and Linguistic Metadata for Early Detection of Depression Indications in Text Sequences” (Marcel, Sven & Christoph, 2018) and “Identifying Depression on Social Media” (Kali, 2019). While the paper of Marcel, Sven & Christoph discusses the efficiency of different word embedding techniques such as word2vec, GloVe and fastText, Kali focuses on the comparison of different models. Both studies show the outstanding performance of convolutional neural network (CNN) in predicting depressed subject. Kali's character-based CNN model achieves a high prediction accuracy at 92.5% which is much better than the 85% accuracy of his base-line SVM.</div>


# Objective and Scope

<div style="text-align: justify">
This project aims to identify depressed subjects utilizing comments scrapped from Reddit. With respect to the excellence of CNN in previous research, we will build a CNN architecture to predict comments labeled as depression. Moreover, we also expect to see how a more popular neural network model in text classification will perform comparing to the CNN.</div>

<div style="text-align: justify">
<br/>
We will build two neural network models to detect depression. The base-line model is CNN which follows a simple architecture: an embedding layer, a Conv1D layer, a GlobalAveragePooling1D layer (an alternative to the whole Flatten - Fully Connected - Dropout paradigm) and the output layer. The second model is the recurrent neural network (RNN) which is well-known for text classification and sequence classification tasks. The RNN architecture also begins with an embedding layer followed by a Long Short Term Memory (LSTM) layer. The Drop-out layer is included between the LSTM layer and the output layer to reduce overfitting.</div>
    
<div style="text-align: justify">
<br/>
For model evaluation, we will use accuracy and the test loss as primary metrics. To improve the prediction accuracy, we will implement a word embedding model at the embedding layer of the neural network architectures. We will use the keras embedding layer created by Tokenizer in the Keras API and a pre-trained Word2Vec model for word embedding.</div>

# Dataset

<div style="text-align: justify">
The data was collected following the reasoning of JT Wolohan in his paper “Detecting Linguistic Traces of Depression in Topic-Restricted Text : Attending to Self-Stigmatized Depression with NLP” (2018). The data was scraped from two subreddits: /r/depression and /r/AskReddit using Python Reddit API Wrapper (PRAW).</div>

<div style="text-align: justify">
<br/>
Comments in /r/depression are labeled as depression (1) and comments in /r/AskReddit are labeled as non-depression (0). The dataset after removing missing data due to deleted comments contains 5,474 comments in total: 2,719 comments labeled as depression and 2,755 comments labeled as non-depression, which makes the dataset extremely balance for analyzing and modeling. The dataset is divided into training set and testing set with an 70% - 30% ratio. From the training data, 30% of the comments is split for the validation.</div>

# Data Preprocessing

## Loading libraries
We will load some libraries which are useful for this project.


```python
import warnings  # ignore warning
warnings.filterwarnings("ignore")
```


```python
# matplot
import matplotlib.pyplot as plt
%matplotlib inline 

# Set matplotlib sizes
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=15)
plt.rc('figure', titlesize=20)
```


```python
# sklearn metrics
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
```


```python
# Load some libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import array
import re
import os

from nltk.tokenize import RegexpTokenizer
from string import punctuation
from nltk.tokenize import sent_tokenize, word_tokenize 

# keras packages
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

```

    Using TensorFlow backend.



```python
# Make directory to save the models
directory = os.path.dirname('../model/')
if not os.path.exists(directory):
    os.makedirs(directory)
```


```python
# Make directory to save the figures
directory = os.path.dirname('../figure/')
if not os.path.exists(directory):
    os.makedirs(directory)
```

## Loading data


```python
# load the scrapped and labeled data
df = pd.read_csv("../data/data.csv")

df.head() # print first 5 rows of df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Our most-broken and least-understood rules is ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Perhaps what's worse than depression is not be...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>A cop stopped me from killing myself last nigh...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Who else is just keeping themselves alive for ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>It doesn't get better. Fuck off with that shit...</td>
    </tr>
  </tbody>
</table>
</div>



## Text preprocessing

Before building the models, we need to preprocess the texts in the Reddit comments. All comments will be converted to lowercase letters. Irrelevant texts such as subreddits, warnings, html tags, numbers, extra punctuations and whitespaces will also be removed. Sentences will not be trimmed for length and stop words will be kept since they affect the sentiment and semantics of the comments.


```python
def clean_text(document):
    
    """
    The clean_text function preprocesses the texts in a document/comment
    
    Parameters
    ----------
    document: the raw text
    
    Returns
    ----------
    tokens: a list of preprocessed tokens
    
    """
    
    
    document = ' '.join([word.lower() for word in word_tokenize(document)]) # lowercase texts
    tokens = word_tokenize(document) # tokenize the document
    
    for i in range(0,len(tokens)):
        # remove whitespaces
        tokens[i] = tokens[i].strip()
        # remove html links
        tokens[i] = re.sub(r'\S*http\S*', '', tokens[i]) # remove links with http
        tokens[i] = re.sub(r'\S*\.org\S*', '', tokens[i]) # remove links with .org
        tokens[i] = re.sub(r'\S*\.com\S*', '', tokens[i]) # remove links with .com
        
        # remove subreddit titles (e.g /r/food)
        tokens[i] = re.sub(r'S*\/r\/\S*', '' ,tokens[i]) 
        
        # remove non-alphabet characters
        tokens[i] = re.sub("[^a-zA-Z]+", "", tokens[i])
        
        tokens[i] = tokens[i].strip() # remove whitespaces 
        
        # remove all blanks from the list
    while("" in tokens): 
        tokens.remove("") 
     
    return tokens
    
```


```python
# call clean_text on df
# for each row in df
for i in range(0,len(df)):
    # use clean_text on the document/text stored in the content column
    clean = clean_text(df.loc[i,"content"])
    # joining the tokens together by whitespaces
    df.loc[i,"clean_content"] = ' '.join([token for token in clean])
```


```python
df = df.dropna() # remove null data due to some deleted comments
df = df[df["clean_content"] != ''] # remove blank comments
df.to_csv('../data/preprocessed_data.csv', index=False) # save the preprocessed data 
```

## Overview of the data


```python
# select label and clean_content columns
df = df[["label","clean_content"]] 
df.reset_index(drop=True,inplace=True) # reset the df index
```


```python
# print the number of subjects that are diagonosed with depression
print("The number of subjects with depression is:", df[df["label"]==1].shape[0])
# print the number of subjects that are not diagonosed with depression
print("The number of subjects without depression is:", df[df["label"]==0].shape[0])
```

    The number of subjects with depression is: 2719
    The number of subjects without depression is: 2755


The ratio of depression and non-depression is approximately 50:50. This shows that our dataset is symmetric and balance for modeling.

# Word embedding

## Creating a vocabulary

Before implementing word embedding models, we need to create a vocabulary of unique words in the Reddit comments. Words that appear only 1 time will be removed to tidy the data since they have no impact on the prediction.


```python
from collections import Counter

words = [] # create a list to store words in the data
for i in range(0,len(df)):
    tokens = df.loc[i,"clean_content"].split() # split words by whitespaces since we've already preprocessed text
    for token in tokens:
        words.append(token) # append word to the list words
```


```python
vocab = Counter(words) # list the words and count their occurence
tokens = [k for k,c in vocab.items() if c >= 2] # remove words that appear only 1 time
vocab = tokens # get the vocabulary of unique words
```

### Getting the size of the vocabulary


```python
# The vocabulary size is the total number of words in our vocabulary, plus one for unknown words
vocab_size = len(set(vocab)) + 1
vocab_size
```




    11689



## Removing words that are not in the vocabulary


```python
for i in range(0,len(df)): # remove the words that are not in the vocab
    tokens = df.loc[i,"clean_content"].split() # split words by whitespaces
    # selecting the words in the vocab and re-join them by whitespaces
    df.loc[i,"clean_content"] = ' '.join([token for token in tokens if token in vocab])
```

## Splitting the training, validation and testing data


```python
from sklearn.model_selection import train_test_split

# Divide the data into training (70%) and testing (30%)
df_train_valid, df_test = train_test_split(df, train_size=0.7, random_state=42, stratify=df["label"])

# Divide the trainning data into training (70%) and validation (30%)
df_train, df_valid = train_test_split(df_train_valid, train_size=0.7, random_state=42, stratify=df_train_valid["label"])
```


```python
df_train.reset_index(drop=True,inplace=True) # reset index
df_valid.reset_index(drop=True,inplace=True)
df_test.reset_index(drop=True,inplace=True)
```


```python
# print training data dimensions
df_train.shape
```




    (2681, 2)




```python
# print validation data dimensions
df_valid.shape
```




    (1150, 2)




```python
# print testing data dimensions
df_test.shape
```




    (1643, 2)




```python
train_docs = [] # a list to store comments from the training set
valid_docs = [] # a list to store comments from the validation set
test_docs = [] # a list to store comments from the testing set

for i in range(0,len(df_train)):
    text = df_train.loc[i,"clean_content"] # selecting each comment in each row 
    train_docs.append(text)  # append each comment to a list of documents 
    
for i in range(0,len(df_valid)):
    text = df_valid.loc[i,"clean_content"]
    valid_docs.append(text)
    
for i in range(0,len(df_test)):
    text = df_test.loc[i,"clean_content"]
    test_docs.append(text)

```

## Creating a list to store comments


```python
docs = train_docs + valid_docs + test_docs
```


```python
# get the max-length of the documents
max_length = max([len(document.split()) for document in docs])
# print max_length
max_length
```




    3415



## Keras embedding layer

### Vectorizing the training, validation and testing comments

- We will use the Tokenizer class in the Keras API to convert the texts in each comment into a numeric vector. At first, we use the Tokenizer to create the word indices, and then convert the training, validation and testing data to sequence of word indexes. We choose the documents with the maximum length as the default length and pad the sequences to create a sequence of same length to be passed to the neural networks. 
- It is noted that 0s will be padded to sequences that have smaller length than the default length. We set padding = 'pre' to pad these 0s to the beginning of the sequences. Since the LSTM layer takes the final output/hidden state to make prediction, a bunch of 0s at the end of the sequence would affect the predictive ability of the network.


```python
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(docs)
# getting the vocabulary of words after using Tokenizer()
# it will have the same unique words and the same length as the above vocab but with different orders of words
tokenizer_vocab = tokenizer.word_index
```


```python
def defineXY(df, docs):
    """
    defineXY convert the texts into numeric vectors using the Tokenizer class in the Keras API
    
    Parameters
    ----------
    df: the dataframe
    
    Returns
    ----------
    X_label: an array of vectorized comments
    Y_label: an array of target labels
    
    """
    
    # converting comments into numeric vectors using layer embedding
    encoded_docs = tokenizer.texts_to_sequences(docs) 
    
    # save each vectorized comment into X_label array
    # max_length set to ensure that all vectorized comments will have the same length
    # since the recurrent network predicts the next elements using the outputs of previous ones
    # we need to set padding as pre so that the RNN model won't predict zeros at the end of previous vectors
    # and give wrong predictions
    X_label = pad_sequences(encoded_docs, maxlen=max_length, padding='pre') 
    
    # saving the label for each comment into an array
    df_label = df["label"]
    y_label = array([df_label[i] for i in range(0,len(df_label))]) # save target label into an array
    
    return X_label, y_label
```


```python
# call defineXY on the training, validation and testing data
Xtrain, ytrain = defineXY(df_train, train_docs)
Xvalid, yvalid = defineXY(df_valid, valid_docs)
Xtest, ytest = defineXY(df_test, test_docs)
```

## Pre-trained Word2Vec layer

We will build a pre-trained Word2Vec model as the embedding layer.

### Train the Word2Vec model


```python
sentences = [] # create a list to store sentences of tokens
for item in docs: # for each sentence in docs
    tokens = item.split() # split the sentence into tokens
    sentences.append(tokens) # append the list of tokens in a sentence to sentences
```


```python
from gensim.models import Word2Vec
# train the sentences with Word2Vec model
# the dimension is set at 100 which is the default dimension output we will use in this project
# we only count words that appear at least 2 times
model_word2vec = Word2Vec(sentences, size=100, window=5, workers=8, min_count=2)
```


```python
print(model_word2vec) # check the Word2Vec model details
```

    Word2Vec(vocab=11688, size=100, alpha=0.025)



```python
# the length of the vocabulary when using Tokenizer in Keras API
print('The number of words in vocabulary using Keras embedding layer is:', len(tokenizer_vocab))
```

    The number of words in vocabulary using Keras embedding layer is: 11688



```python
# checking the length of the vocab when using Word2Vec
print('The number of words in vocabulary using Word2Vec is:', len(model_word2vec.wv.vocab))
```

    The number of words in vocabulary using Word2Vec is: 11688


The length of the vocabulary after using the word2vec model and Keras embedding layer are the same. There is no mistakes in our word embedding process.

### Saving the Word2Vec embedding in a .txt file
We will save the Word2Vec embedding model in a file. It will save time for the project since we do not have to re-train the Word2Vec model when we need to make changes.


```python
from os import listdir
# save model in ASCII (word2vec) format
filename = '../word2vec_embedding/embedding_word2vec.txt'
model_word2vec.wv.save_word2vec_format(filename, binary=False)
```

### Loading the Word2Vec embedding


```python
def load_embedding(filename):
    
    """
    load_embedding loads the saved Word2Vec embedding model 
    
    Parameters
    ----------
    filename: the name of the embedding file
    
    Returns
    ----------
    raw_embedding: a dictionary of words and their vectors as arrays
    
    """
    
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding
```


```python
raw_embedding = load_embedding('../word2vec_embedding/embedding_word2vec.txt')
```

### Mapping the embedding vectors

We need to map the vectors and words from the raw_embedding to the vectors and words from the tokenizer_vocab in a correct order.


```python
# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab_size):
    """
    get_weight_maxtrix re-arrange the raw_embedding dictionary in the correct order given by the tokenizer_vocab
    
    Parameters
    ----------
    embedding: the embedding dictionary
    vocab_size: the length of vocabulary plus 1 for unknown words
    
    Returns
    ----------
    weight_matrix: the array of the vectorized words in the correct order
    
    """
    
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 100))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in tokenizer_vocab.items(): # for word and their index i in the tokenizer_vocab
        if i > vocab_size:
            continue
        vector = embedding.get(word) # get the word in the embedding file
        if vector is not None: # word not found will be returned zero
            weight_matrix[i] = vector # store the vector of the word in the weight_matrix at the position i
    return weight_matrix
```


```python
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, vocab_size)
```

# Building Neural Network Models

In this part, we will build two kinds of neural network models: Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN).

## Setting callbacks


```python
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ModelCheckpoint callback
model_checkpoint_cb_CNN_keras = ModelCheckpoint(   # for CNN using Keras API Tokenizer
    filepath = "../model/CNN_keras.h5",
    save_best_only=True)     

model_checkpoint_cb_RNN_keras = ModelCheckpoint(  # for RNN using Keras API Tokenizer
    filepath = "../model/RNN_keras.h5",   
    save_best_only=True)

model_checkpoint_cb_CNN_word2vec = ModelCheckpoint(   # for CNN using word2vec
    filepath = "../model/CNN_word2vec.h5",
    save_best_only=True)     

model_checkpoint_cb_RNN_word2vec = ModelCheckpoint(  # for RNN using word2vec
    filepath = "../model/RNN_word2vec.h5",   
    save_best_only=True)


# EarlyStopping callback
early_stopping_cb = EarlyStopping(
    patience=5,  # stopping after 5 epochs without improvement
    restore_best_weights=True)

# ReduceLROnPlateau callback
reduce_lr_on_plateau_cb = ReduceLROnPlateau(
    verbose = 1,
    factor=0.1,  # reducing the learning rate by 10 times 
    patience=2)  # after 2 epochs without improvement in validation loss
```

## CNN and Keras embedding layer

### Building architecture (CNN and Keras)
We will build a simple architecture for CNN using a Conv1D layer. The GlobalAveragePooling1D layer is used as an alternative for the Flatten - Fully Connected (FC) - Dropout paradigm. In the future work, we can add more convolutional layers to obtain better results. 


```python
# create a squential
model = Sequential()
# add the embedding layer
model.add(Embedding(vocab_size, 100, input_length=max_length)) 
# add the convolutional layer
model.add(Conv1D(filters=32, kernel_size=8, padding="same", activation='relu'))
# add the GAP layer
model.add(GlobalAveragePooling1D())
# add a fully connected layer with the activation function as relu
model.add(Dense(10, activation='relu'))
# add the output layer
# since the this a binary prediciton (0 or 1)
# sigmoid is the activation function and the dimensionality of the output space is 1
model.add(Dense(1, activation='sigmoid'))

# print the model summary
model.summary()

```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 3415, 100)         1168900   
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 3415, 32)          25632     
    _________________________________________________________________
    global_average_pooling1d_1 ( (None, 32)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                330       
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 1,194,873
    Trainable params: 1,194,873
    Non-trainable params: 0
    _________________________________________________________________


### Compiling and training the model (CNN and Keras)


```python
# compile network
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=10 ** -3), metrics=['accuracy'])
# fit network
history_CNN_keras = model.fit(Xtrain, ytrain, epochs=100, validation_data=(Xvalid, yvalid), 
            callbacks=[model_checkpoint_cb_CNN_keras, early_stopping_cb, reduce_lr_on_plateau_cb])

```

    Train on 2681 samples, validate on 1150 samples
    Epoch 1/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.6801 - accuracy: 0.6319 - val_loss: 0.6348 - val_accuracy: 0.7148
    Epoch 2/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.5651 - accuracy: 0.7438 - val_loss: 0.5077 - val_accuracy: 0.7809
    Epoch 3/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.4594 - accuracy: 0.8023 - val_loss: 0.4382 - val_accuracy: 0.8409
    Epoch 4/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3770 - accuracy: 0.8482 - val_loss: 0.3623 - val_accuracy: 0.8661
    Epoch 5/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2934 - accuracy: 0.8836 - val_loss: 0.2926 - val_accuracy: 0.8913
    Epoch 6/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2236 - accuracy: 0.9206 - val_loss: 0.2555 - val_accuracy: 0.9078
    Epoch 7/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.1781 - accuracy: 0.9407 - val_loss: 0.2262 - val_accuracy: 0.9174
    Epoch 8/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.1428 - accuracy: 0.9567 - val_loss: 0.2070 - val_accuracy: 0.9296
    Epoch 9/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.1141 - accuracy: 0.9646 - val_loss: 0.1924 - val_accuracy: 0.9365
    Epoch 10/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.0921 - accuracy: 0.9724 - val_loss: 0.1910 - val_accuracy: 0.9374
    Epoch 11/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.0749 - accuracy: 0.9784 - val_loss: 0.1854 - val_accuracy: 0.9383
    Epoch 12/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.0602 - accuracy: 0.9862 - val_loss: 0.1898 - val_accuracy: 0.9365
    Epoch 13/100
    2681/2681 [==============================] - 15s 5ms/step - loss: 0.0492 - accuracy: 0.9881 - val_loss: 0.1881 - val_accuracy: 0.9374
    
    Epoch 00013: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 14/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.0412 - accuracy: 0.9914 - val_loss: 0.1880 - val_accuracy: 0.9374
    Epoch 15/100
    2681/2681 [==============================] - 15s 5ms/step - loss: 0.0404 - accuracy: 0.9918 - val_loss: 0.1883 - val_accuracy: 0.9374
    
    Epoch 00015: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
    Epoch 16/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.0397 - accuracy: 0.9922 - val_loss: 0.1883 - val_accuracy: 0.9365


### Plotting the learning curve (CNN and Keras)


```python
pd.DataFrame(history_CNN_keras.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.savefig('../figure/learning_curve_CNN_keras.pdf')
plt.show()
```


![png](output_81_0.png)


### Model evaluation (CNN and Keras)

### 1. Loss and Accuracy


```python
# Load the model
model = keras.models.load_model("../model/CNN_keras.h5")

# evaluating the model
loss, accuracy = model.evaluate(Xtest, ytest, verbose = 0)

# print loss and accuracy
print("loss:", loss)
print("accuracy:", accuracy)
```

    loss: 0.1906582722944852
    accuracy: 0.9360924959182739


### 2. Confusion matrix


```python
# predict probabilities for test set
yhat_probs = model.predict(Xtest, verbose=0) # 2d arrary
# predict crisp classes for test set
yhat_classes = model.predict_classes(Xtest, verbose=0) # 2d array

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
```


```python
print("Confusion Matrix")
pd.DataFrame(
    confusion_matrix(ytest, yhat_classes, labels=[0,1]), 
    index=['True : {:}'.format(x) for x in [0,1]], 
    columns=['Pred : {:}'.format(x) for x in [0,1]])
```

    Confusion Matrix





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pred : 0</th>
      <th>Pred : 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True : 0</th>
      <td>772</td>
      <td>55</td>
    </tr>
    <tr>
      <th>True : 1</th>
      <td>50</td>
      <td>766</td>
    </tr>
  </tbody>
</table>
</div>



### 3. ROC curve


```python
# getting false positive rate, true positive rate 
fpr, tpr, threshold = metrics.roc_curve(ytest, yhat_probs)
# roc auc score
auc = roc_auc_score(ytest, yhat_probs)

# plot ROC curve
plt.figure(figsize=(8,5))
plt.tight_layout()
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('../figure/ROC_curve_CNN_keras.pdf')
plt.show()
```


![png](output_89_0.png)


This high AUC score of 0.977 shows that the model is outstanding at discrimination.

### 4. Precision, Recall and F1 score


```python
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes)
print('F1 score: %f' % f1)
```

    Precision: 0.933009
    Recall: 0.938725
    F1 score: 0.935858


## RNN and Keras embedding layer

### Building architecture (RNN and Keras)
We will build a simple architecture for RNN with a Long Short Term Memory (LSTM) layer.


```python
# define model
# create a squential
model = Sequential()
# add the embedding layer
model.add(Embedding(vocab_size, 100, input_length=max_length))
# add the LSTM layer
model.add(LSTM(100))
# add drop out to prevent overfitting
model.add(Dropout(0.2))
# add the output layer
model.add(Dense(1, activation='sigmoid'))

# print model summary
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 3415, 100)         1168900   
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               80400     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 1,249,401
    Trainable params: 1,249,401
    Non-trainable params: 0
    _________________________________________________________________


### Compiling and training the model (RNN and Keras)


```python
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=10 ** -3), metrics=['accuracy'])
history_RNN_keras = model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), epochs=100, 
                        callbacks=[model_checkpoint_cb_RNN_keras, early_stopping_cb, reduce_lr_on_plateau_cb])
```

    Train on 2681 samples, validate on 1150 samples
    Epoch 1/100
    2681/2681 [==============================] - 167s 62ms/step - loss: 0.4857 - accuracy: 0.7766 - val_loss: 0.3199 - val_accuracy: 0.8922
    Epoch 2/100
    2681/2681 [==============================] - 171s 64ms/step - loss: 0.1970 - accuracy: 0.9336 - val_loss: 0.2488 - val_accuracy: 0.9122
    Epoch 3/100
    2681/2681 [==============================] - 165s 62ms/step - loss: 0.1127 - accuracy: 0.9683 - val_loss: 0.2393 - val_accuracy: 0.9104
    Epoch 4/100
    2681/2681 [==============================] - 158s 59ms/step - loss: 0.1239 - accuracy: 0.9590 - val_loss: 0.2457 - val_accuracy: 0.9261
    Epoch 5/100
    2681/2681 [==============================] - 158s 59ms/step - loss: 0.0478 - accuracy: 0.9862 - val_loss: 0.2605 - val_accuracy: 0.9183
    
    Epoch 00005: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 6/100
    2681/2681 [==============================] - 158s 59ms/step - loss: 0.0197 - accuracy: 0.9963 - val_loss: 0.2606 - val_accuracy: 0.9287
    Epoch 7/100
    2681/2681 [==============================] - 158s 59ms/step - loss: 0.0160 - accuracy: 0.9970 - val_loss: 0.2634 - val_accuracy: 0.9296
    
    Epoch 00007: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
    Epoch 8/100
    2681/2681 [==============================] - 158s 59ms/step - loss: 0.0145 - accuracy: 0.9974 - val_loss: 0.2639 - val_accuracy: 0.9296


### Plotting the learning curve (RNN and Keras)


```python
pd.DataFrame(history_RNN_keras.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.savefig('../figure/learning_curve_RNN_keras.pdf')
plt.show()
```


![png](output_99_0.png)


### Model evaluation (RNN and Keras)

### 1. Loss and Accuracy


```python
# Load the model
model = keras.models.load_model("../model/RNN_keras.h5")

# evaluating the model
loss, accuracy = model.evaluate(Xtest, ytest, verbose = 0)

# print loss and accuracy
print("loss:", loss)
print("accuracy:", accuracy)
```

    loss: 0.26942492168730015
    accuracy: 0.9013998508453369


### 2. Confusion matrix


```python
# predict probabilities for test set
yhat_probs = model.predict(Xtest, verbose=0) # 2d arrary
# predict crisp classes for test set
yhat_classes = model.predict_classes(Xtest, verbose=0) # 2d array

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

print("Confusion Matrix")
pd.DataFrame(
    confusion_matrix(ytest, yhat_classes, labels=[0,1]), 
    index=['True : {:}'.format(x) for x in [0,1]], 
    columns=['Pred : {:}'.format(x) for x in [0,1]])
```

    Confusion Matrix





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pred : 0</th>
      <th>Pred : 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True : 0</th>
      <td>702</td>
      <td>125</td>
    </tr>
    <tr>
      <th>True : 1</th>
      <td>37</td>
      <td>779</td>
    </tr>
  </tbody>
</table>
</div>



### 3. ROC curve


```python
# getting false positive rate, true positive rate 
fpr, tpr, threshold = metrics.roc_curve(ytest, yhat_probs)
# roc auc score
auc = roc_auc_score(ytest, yhat_probs)

# plot ROC curve
plt.figure(figsize=(8,5))
plt.tight_layout()
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('../figure/ROC_curve_RNN_keras.pdf')
plt.show()
```


![png](output_106_0.png)


With a high AUC score at 0.9642, the CNN model with embedding layer provides an outstanding discrimination.

### 4. Precision, Recall and F1 score


```python
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes)
print('F1 score: %f' % f1)
```

    Precision: 0.861726
    Recall: 0.954657
    F1 score: 0.905814


## CNN and Word2Vec

### Building architecture (CNN and Word2Vec)
For the use of the Word2Vec embedding, we add this pre-trained embedding layer at the beginning of the architecture. To ensure that the neural network does not try to adapt the pre-learned vectors as part of training the network, we need to freeze this pre-trained Word2Vec layer.


```python
# create the pre-trained embedding layer by using the embedding_vectors as the weights
# trainable is set as False to freeze the pre-trained layer
embedding_layer = Embedding(vocab_size, 100, embeddings_initializer=keras.initializers.Constant(embedding_vectors), input_length=max_length, trainable=False)
```


```python
# create a squential
model = Sequential()
# add the embedding layer
model.add(embedding_layer) 
# add the convolutional layer
model.add(Conv1D(filters=128, kernel_size=5, padding="same", activation='relu'))
# add the GAP layer
model.add(GlobalAveragePooling1D())
# add a fully connected layer with the activation function as relu
model.add(Dense(10, activation='relu'))
# add the output layer
# since the this a binary prediciton (0 or 1)
# sigmoid is the activation function and the dimensionality of the output space is 1
model.add(Dense(1, activation='sigmoid'))

# print the model summary
model.summary()

```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 3415, 100)         1168900   
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 3415, 128)         64128     
    _________________________________________________________________
    global_average_pooling1d_2 ( (None, 128)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 10)                1290      
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 1,234,329
    Trainable params: 65,429
    Non-trainable params: 1,168,900
    _________________________________________________________________


### Compiling and training the model (CNN and Word2Vec)


```python
# compile network
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=10 ** -3), metrics=['accuracy'])
# fit network
history_CNN_word2vec = model.fit(Xtrain, ytrain, epochs=100, validation_data=(Xvalid, yvalid),
            callbacks=[model_checkpoint_cb_CNN_word2vec, early_stopping_cb, reduce_lr_on_plateau_cb])
```

    Train on 2681 samples, validate on 1150 samples
    Epoch 1/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.6277 - accuracy: 0.6512 - val_loss: 0.5468 - val_accuracy: 0.8183
    Epoch 2/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.4768 - accuracy: 0.8101 - val_loss: 0.4392 - val_accuracy: 0.8235
    Epoch 3/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.4158 - accuracy: 0.8318 - val_loss: 0.4089 - val_accuracy: 0.8365
    Epoch 4/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3931 - accuracy: 0.8448 - val_loss: 0.3926 - val_accuracy: 0.8487
    Epoch 5/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3784 - accuracy: 0.8579 - val_loss: 0.3796 - val_accuracy: 0.8530
    Epoch 6/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3679 - accuracy: 0.8594 - val_loss: 0.3770 - val_accuracy: 0.8635
    Epoch 7/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3613 - accuracy: 0.8661 - val_loss: 0.3580 - val_accuracy: 0.8635
    Epoch 8/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3481 - accuracy: 0.8680 - val_loss: 0.3593 - val_accuracy: 0.8704
    Epoch 9/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3446 - accuracy: 0.8680 - val_loss: 0.3666 - val_accuracy: 0.8565
    
    Epoch 00009: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 10/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3350 - accuracy: 0.8780 - val_loss: 0.3399 - val_accuracy: 0.8713
    Epoch 11/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3276 - accuracy: 0.8799 - val_loss: 0.3387 - val_accuracy: 0.8713
    Epoch 12/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3267 - accuracy: 0.8750 - val_loss: 0.3371 - val_accuracy: 0.8713
    Epoch 13/100
    2681/2681 [==============================] - 15s 5ms/step - loss: 0.3256 - accuracy: 0.8758 - val_loss: 0.3361 - val_accuracy: 0.8704
    Epoch 14/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3248 - accuracy: 0.8765 - val_loss: 0.3352 - val_accuracy: 0.8713
    Epoch 15/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3244 - accuracy: 0.8758 - val_loss: 0.3347 - val_accuracy: 0.8739
    Epoch 16/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3233 - accuracy: 0.8769 - val_loss: 0.3332 - val_accuracy: 0.8730
    Epoch 17/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3223 - accuracy: 0.8773 - val_loss: 0.3323 - val_accuracy: 0.8713
    Epoch 18/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3216 - accuracy: 0.8765 - val_loss: 0.3317 - val_accuracy: 0.8713
    Epoch 19/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3208 - accuracy: 0.8803 - val_loss: 0.3307 - val_accuracy: 0.8748
    Epoch 20/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3203 - accuracy: 0.8799 - val_loss: 0.3302 - val_accuracy: 0.8722
    Epoch 21/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3195 - accuracy: 0.8750 - val_loss: 0.3290 - val_accuracy: 0.8748
    Epoch 22/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3186 - accuracy: 0.8788 - val_loss: 0.3282 - val_accuracy: 0.8739
    Epoch 23/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3173 - accuracy: 0.8803 - val_loss: 0.3274 - val_accuracy: 0.8748
    Epoch 24/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3170 - accuracy: 0.8791 - val_loss: 0.3264 - val_accuracy: 0.8765
    Epoch 25/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3158 - accuracy: 0.8814 - val_loss: 0.3256 - val_accuracy: 0.8765
    Epoch 26/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3157 - accuracy: 0.8791 - val_loss: 0.3251 - val_accuracy: 0.8757
    Epoch 27/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3144 - accuracy: 0.8806 - val_loss: 0.3243 - val_accuracy: 0.8722
    Epoch 28/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3137 - accuracy: 0.8833 - val_loss: 0.3234 - val_accuracy: 0.8748
    Epoch 29/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3137 - accuracy: 0.8810 - val_loss: 0.3229 - val_accuracy: 0.8774
    Epoch 30/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3124 - accuracy: 0.8814 - val_loss: 0.3218 - val_accuracy: 0.8748
    Epoch 31/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3118 - accuracy: 0.8814 - val_loss: 0.3211 - val_accuracy: 0.8748
    Epoch 32/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3110 - accuracy: 0.8836 - val_loss: 0.3203 - val_accuracy: 0.8757
    Epoch 33/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3105 - accuracy: 0.8821 - val_loss: 0.3200 - val_accuracy: 0.8774
    Epoch 34/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3099 - accuracy: 0.8818 - val_loss: 0.3187 - val_accuracy: 0.8765
    Epoch 35/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3091 - accuracy: 0.8833 - val_loss: 0.3179 - val_accuracy: 0.8765
    Epoch 36/100
    2681/2681 [==============================] - 15s 5ms/step - loss: 0.3081 - accuracy: 0.8847 - val_loss: 0.3173 - val_accuracy: 0.8774
    Epoch 37/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3078 - accuracy: 0.8855 - val_loss: 0.3167 - val_accuracy: 0.8765
    Epoch 38/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3073 - accuracy: 0.8847 - val_loss: 0.3164 - val_accuracy: 0.8774
    Epoch 39/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3063 - accuracy: 0.8844 - val_loss: 0.3155 - val_accuracy: 0.8774
    Epoch 40/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3060 - accuracy: 0.8840 - val_loss: 0.3149 - val_accuracy: 0.8774
    Epoch 41/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3054 - accuracy: 0.8851 - val_loss: 0.3143 - val_accuracy: 0.8791
    Epoch 42/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3045 - accuracy: 0.8847 - val_loss: 0.3142 - val_accuracy: 0.8783
    Epoch 43/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3047 - accuracy: 0.8847 - val_loss: 0.3136 - val_accuracy: 0.8783
    Epoch 44/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.3036 - accuracy: 0.8859 - val_loss: 0.3126 - val_accuracy: 0.8783
    Epoch 45/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3032 - accuracy: 0.8847 - val_loss: 0.3122 - val_accuracy: 0.8783
    Epoch 46/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3024 - accuracy: 0.8840 - val_loss: 0.3114 - val_accuracy: 0.8800
    Epoch 47/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3022 - accuracy: 0.8847 - val_loss: 0.3109 - val_accuracy: 0.8809
    Epoch 48/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3021 - accuracy: 0.8877 - val_loss: 0.3116 - val_accuracy: 0.8757
    Epoch 49/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3019 - accuracy: 0.8855 - val_loss: 0.3101 - val_accuracy: 0.8774
    Epoch 50/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3014 - accuracy: 0.8836 - val_loss: 0.3093 - val_accuracy: 0.8809
    Epoch 51/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3002 - accuracy: 0.8859 - val_loss: 0.3089 - val_accuracy: 0.8800
    Epoch 52/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.3001 - accuracy: 0.8851 - val_loss: 0.3084 - val_accuracy: 0.8809
    Epoch 53/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2999 - accuracy: 0.8851 - val_loss: 0.3080 - val_accuracy: 0.8800
    Epoch 54/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2988 - accuracy: 0.8859 - val_loss: 0.3073 - val_accuracy: 0.8800
    Epoch 55/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2981 - accuracy: 0.8862 - val_loss: 0.3073 - val_accuracy: 0.8783
    Epoch 56/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2981 - accuracy: 0.8870 - val_loss: 0.3064 - val_accuracy: 0.8783
    Epoch 57/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2973 - accuracy: 0.8877 - val_loss: 0.3059 - val_accuracy: 0.8809
    Epoch 58/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2968 - accuracy: 0.8870 - val_loss: 0.3059 - val_accuracy: 0.8791
    Epoch 59/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2965 - accuracy: 0.8870 - val_loss: 0.3049 - val_accuracy: 0.8791
    Epoch 60/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2959 - accuracy: 0.8862 - val_loss: 0.3045 - val_accuracy: 0.8783
    Epoch 61/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2956 - accuracy: 0.8855 - val_loss: 0.3040 - val_accuracy: 0.8809
    Epoch 62/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2949 - accuracy: 0.8870 - val_loss: 0.3036 - val_accuracy: 0.8791
    Epoch 63/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2946 - accuracy: 0.8855 - val_loss: 0.3034 - val_accuracy: 0.8800
    Epoch 64/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2943 - accuracy: 0.8888 - val_loss: 0.3037 - val_accuracy: 0.8783
    Epoch 65/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2942 - accuracy: 0.8870 - val_loss: 0.3024 - val_accuracy: 0.8791
    Epoch 66/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2935 - accuracy: 0.8870 - val_loss: 0.3020 - val_accuracy: 0.8800
    Epoch 67/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2932 - accuracy: 0.8874 - val_loss: 0.3023 - val_accuracy: 0.8757
    Epoch 68/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2934 - accuracy: 0.8874 - val_loss: 0.3024 - val_accuracy: 0.8748
    
    Epoch 00068: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
    Epoch 69/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2927 - accuracy: 0.8866 - val_loss: 0.3019 - val_accuracy: 0.8757
    Epoch 70/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2922 - accuracy: 0.8859 - val_loss: 0.3014 - val_accuracy: 0.8765
    Epoch 71/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2919 - accuracy: 0.8866 - val_loss: 0.3012 - val_accuracy: 0.8774
    Epoch 72/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2917 - accuracy: 0.8866 - val_loss: 0.3010 - val_accuracy: 0.8783
    Epoch 73/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2916 - accuracy: 0.8870 - val_loss: 0.3009 - val_accuracy: 0.8800
    Epoch 74/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2915 - accuracy: 0.8866 - val_loss: 0.3009 - val_accuracy: 0.8800
    Epoch 75/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2915 - accuracy: 0.8870 - val_loss: 0.3008 - val_accuracy: 0.8800
    Epoch 76/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2914 - accuracy: 0.8874 - val_loss: 0.3008 - val_accuracy: 0.8800
    Epoch 77/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2914 - accuracy: 0.8881 - val_loss: 0.3007 - val_accuracy: 0.8800
    Epoch 78/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2913 - accuracy: 0.8881 - val_loss: 0.3007 - val_accuracy: 0.8800
    Epoch 79/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2913 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    
    Epoch 00079: ReduceLROnPlateau reducing learning rate to 1.0000000656873453e-06.
    Epoch 80/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2912 - accuracy: 0.8888 - val_loss: 0.3006 - val_accuracy: 0.8800
    Epoch 81/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2912 - accuracy: 0.8888 - val_loss: 0.3006 - val_accuracy: 0.8800
    Epoch 82/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2912 - accuracy: 0.8888 - val_loss: 0.3006 - val_accuracy: 0.8800
    
    Epoch 00082: ReduceLROnPlateau reducing learning rate to 1.0000001111620805e-07.
    Epoch 83/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    Epoch 84/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    
    Epoch 00084: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-08.
    Epoch 85/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    Epoch 86/100
    2681/2681 [==============================] - 15s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    
    Epoch 00086: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-09.
    Epoch 87/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    Epoch 88/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    
    Epoch 00088: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-10.
    Epoch 89/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    Epoch 90/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    
    Epoch 00090: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-11.
    Epoch 91/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    Epoch 92/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800
    
    Epoch 00092: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-12.
    Epoch 93/100
    2681/2681 [==============================] - 16s 6ms/step - loss: 0.2912 - accuracy: 0.8885 - val_loss: 0.3006 - val_accuracy: 0.8800


### Plotting the learning curve (CNN and Word2Vec)


```python
pd.DataFrame(history_CNN_word2vec.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.savefig('../figure/learning_curve_CNN_word2vec.pdf')
plt.show()
```


![png](output_117_0.png)


### Model evaluation (CNN and Word2Vec)

### 1. Loss and Accuracy


```python
# Load the model
model = keras.models.load_model("../model/CNN_word2vec.h5")

# evaluating the model
loss, accuracy = model.evaluate(Xtest, ytest, verbose = 0)

# print loss and accuracy
print("loss:", loss)
print("accuracy:", accuracy)
```

    loss: 0.31271492067227913
    accuracy: 0.8800973892211914


### 2. Confusion matrix


```python
# predict probabilities for test set
yhat_probs = model.predict(Xtest, verbose=0) # 2d arrary
# predict crisp classes for test set
yhat_classes = model.predict_classes(Xtest, verbose=0) # 2d array

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

print("Confusion Matrix")
pd.DataFrame(
    confusion_matrix(ytest, yhat_classes, labels=[0,1]), 
    index=['True : {:}'.format(x) for x in [0,1]], 
    columns=['Pred : {:}'.format(x) for x in [0,1]])
```

    Confusion Matrix





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pred : 0</th>
      <th>Pred : 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True : 0</th>
      <td>721</td>
      <td>106</td>
    </tr>
    <tr>
      <th>True : 1</th>
      <td>91</td>
      <td>725</td>
    </tr>
  </tbody>
</table>
</div>



### 3. ROC curve


```python
# getting false positive rate, true positive rate 
fpr, tpr, threshold = metrics.roc_curve(ytest, yhat_probs)
# roc auc score
auc = roc_auc_score(ytest, yhat_probs)

# plot ROC curve
plt.figure(figsize=(8,5))
plt.tight_layout()
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('../figure/ROC_curve_CNN_word2vec.pdf')
plt.show()
```


![png](output_124_0.png)


With a high AUC score at 0.9432, the RNN model with embedding layer has an outstanding discrimination.

### 4. Precision, Recall and F1 score


```python
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes)
print('F1 score: %f' % f1)
```

    Precision: 0.872443
    Recall: 0.888480
    F1 score: 0.880389


## RNN and Word2Vec

### Building architecture (RNN and Word2Vec)


```python
# define model
# create a squential
model = Sequential()
# add the embedding layer
model.add(embedding_layer) 
# add the LSTM layer
model.add(LSTM(100))
model.add(Dropout(0.2))
# add the output layer
model.add(Dense(1, activation='sigmoid'))

# print model summary
model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 3415, 100)         1168900   
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 100)               80400     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 1,249,401
    Trainable params: 80,501
    Non-trainable params: 1,168,900
    _________________________________________________________________


### Compiling and training the model (RNN and Word2Vec)


```python
# compile network
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=10 ** -3), metrics=['accuracy'])
# fit network
history_RNN_word2vec = model.fit(Xtrain, ytrain, epochs=100, validation_data=(Xvalid, yvalid),
            callbacks=[model_checkpoint_cb_RNN_word2vec, early_stopping_cb, reduce_lr_on_plateau_cb])
```

    Train on 2681 samples, validate on 1150 samples
    Epoch 1/100
    2681/2681 [==============================] - 138s 51ms/step - loss: 0.5495 - accuracy: 0.7374 - val_loss: 0.4204 - val_accuracy: 0.8452
    Epoch 2/100
    2681/2681 [==============================] - 138s 51ms/step - loss: 0.3774 - accuracy: 0.8571 - val_loss: 0.3598 - val_accuracy: 0.8635
    Epoch 3/100
    2681/2681 [==============================] - 135s 50ms/step - loss: 0.3520 - accuracy: 0.8612 - val_loss: 0.3881 - val_accuracy: 0.8470
    Epoch 4/100
    2681/2681 [==============================] - 137s 51ms/step - loss: 0.3580 - accuracy: 0.8665 - val_loss: 0.3357 - val_accuracy: 0.8835
    Epoch 5/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.3453 - accuracy: 0.8620 - val_loss: 0.3628 - val_accuracy: 0.8661
    Epoch 6/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.4289 - accuracy: 0.8310 - val_loss: 0.3265 - val_accuracy: 0.8791
    Epoch 7/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.3621 - accuracy: 0.8635 - val_loss: 0.3480 - val_accuracy: 0.8757
    Epoch 8/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.3229 - accuracy: 0.8799 - val_loss: 0.3452 - val_accuracy: 0.8748
    
    Epoch 00008: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
    Epoch 9/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.2939 - accuracy: 0.8911 - val_loss: 0.3062 - val_accuracy: 0.8974
    Epoch 10/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.2737 - accuracy: 0.9023 - val_loss: 0.3140 - val_accuracy: 0.8870
    Epoch 11/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.2598 - accuracy: 0.9030 - val_loss: 0.3035 - val_accuracy: 0.8930
    Epoch 12/100
    2681/2681 [==============================] - 140s 52ms/step - loss: 0.2562 - accuracy: 0.9071 - val_loss: 0.2982 - val_accuracy: 0.8991
    Epoch 13/100
    2681/2681 [==============================] - 141s 52ms/step - loss: 0.2518 - accuracy: 0.9120 - val_loss: 0.2998 - val_accuracy: 0.8965
    Epoch 14/100
    2681/2681 [==============================] - 139s 52ms/step - loss: 0.2488 - accuracy: 0.9116 - val_loss: 0.2978 - val_accuracy: 0.8974
    Epoch 15/100
    2681/2681 [==============================] - 140s 52ms/step - loss: 0.2457 - accuracy: 0.9123 - val_loss: 0.2964 - val_accuracy: 0.8983
    Epoch 16/100
    2681/2681 [==============================] - 143s 53ms/step - loss: 0.2441 - accuracy: 0.9127 - val_loss: 0.2871 - val_accuracy: 0.8991
    Epoch 17/100
    2681/2681 [==============================] - 137s 51ms/step - loss: 0.2609 - accuracy: 0.9094 - val_loss: 0.2842 - val_accuracy: 0.9017
    Epoch 18/100
    2681/2681 [==============================] - 134s 50ms/step - loss: 0.2415 - accuracy: 0.9176 - val_loss: 0.2937 - val_accuracy: 0.8913
    Epoch 19/100
    2681/2681 [==============================] - 144s 54ms/step - loss: 0.2344 - accuracy: 0.9202 - val_loss: 0.2845 - val_accuracy: 0.8991
    
    Epoch 00019: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
    Epoch 20/100
    2681/2681 [==============================] - 143s 53ms/step - loss: 0.2306 - accuracy: 0.9217 - val_loss: 0.2865 - val_accuracy: 0.8974
    Epoch 21/100
    2681/2681 [==============================] - 141s 53ms/step - loss: 0.2335 - accuracy: 0.9194 - val_loss: 0.2900 - val_accuracy: 0.8948
    
    Epoch 00021: ReduceLROnPlateau reducing learning rate to 1.0000000656873453e-06.
    Epoch 22/100
    2681/2681 [==============================] - 142s 53ms/step - loss: 0.2343 - accuracy: 0.9209 - val_loss: 0.2900 - val_accuracy: 0.8948


### Plotting the learning curve (RNN and Word2Vec)


```python
pd.DataFrame(history_RNN_word2vec.history).plot(figsize=(8, 5))

# Save and show the figure
plt.tight_layout()
plt.title('Learning Curve')
plt.xlabel('epochs')
plt.savefig('../figure/learning_curve_RNN_word2vec.pdf')
plt.show()
```


![png](output_134_0.png)


### Model evaluation (RNN and Word2Vec)

### 1. Loss and Accuracy


```python
# Load the model
model = keras.models.load_model("../model/RNN_word2vec.h5")

# evaluating the model
loss, accuracy = model.evaluate(Xtest, ytest, verbose = 0)

# print loss and accuracy
print("loss:", loss)
print("accuracy:", accuracy)
```

    loss: 0.314475532827476
    accuracy: 0.8934875130653381


### 2. Confusion matrix


```python
# predict probabilities for test set
yhat_probs = model.predict(Xtest, verbose=0) # 2d arrary
# predict crisp classes for test set
yhat_classes = model.predict_classes(Xtest, verbose=0) # 2d array

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

print("Confusion Matrix")
pd.DataFrame(
    confusion_matrix(ytest, yhat_classes, labels=[0,1]), 
    index=['True : {:}'.format(x) for x in [0,1]], 
    columns=['Pred : {:}'.format(x) for x in [0,1]])
```

    Confusion Matrix





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pred : 0</th>
      <th>Pred : 1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>True : 0</th>
      <td>714</td>
      <td>113</td>
    </tr>
    <tr>
      <th>True : 1</th>
      <td>62</td>
      <td>754</td>
    </tr>
  </tbody>
</table>
</div>



### 3. ROC curve


```python
# getting false positive rate, true positive rate 
fpr, tpr, threshold = metrics.roc_curve(ytest, yhat_probs)
# roc auc score
auc = roc_auc_score(ytest, yhat_probs)

# plot ROC curve
plt.figure(figsize=(8,5))
plt.tight_layout()
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('../figure/ROC_curve_RNN_word2vec.pdf')
plt.show()
```


![png](output_141_0.png)


The AUC score of 0.9396 suggests that the model has an outstanding discrimination.

### 4. Precision, Recall and F1 score


```python
# precision tp / (tp + fp)
precision = precision_score(ytest, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(ytest, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(ytest, yhat_classes)
print('F1 score: %f' % f1)
```

    Precision: 0.869666
    Recall: 0.924020
    F1 score: 0.896019


# Result and Discussion

<div style="text-align: justify">
Overall, all the models have very good performance in classifying detection. The accuracies when applying the models to the testing data range from 88% to 93% which indicates their excellence predictive powers. The AUC scores are all above 0.9, which suggests that our models have an outstanding class separation capacity. The reason for the good performance of the models would be the symmetry of the dataset. The ratio of non-depressed comments and depressed comments are approximately 50:50. Furthermore, we do not have to face much troubles during the text preprocessing process since the texts do not contain a lot of special characters. In future work, we need to collect a bigger and more representative dataset so there would be more difficulties in preprocessing the raw texts.</div>


<div style="text-align: justify">
<br/>
The architecture with the best performance is the CNN with keras embedding layer. The accuracy is 93.61% and the test loss is 0.19. The model that has worst performance is the CNN with pre-trained Word2Vec. Its accuracy is 88.01% and the test loss is 0.31. The two RNN models show small difference in their performances. The accuracy and test loss of the RNN model with keras embedding layer are 90.14% and 0.27 respectively, while the case of pre-trained Word2Vec has little worse accuracy and loss at 89.35% and 0.31 respectively. It is worthwide to note that the computational speed of CNN is much faster than this of RNN.</div>

<div style="text-align: justify">
<br/>
The results show that using a pre-trained Word2Vec seems not be as good as using the keras embedding layer. The majority of comments and posts in social platforms such as Facebook, Twitter and Reddit are often written in informal ways and do not follow the correct grammar structures. There are many sentences containing short informal texts and symbols to express feelings. Therefore, the Word2Vec seems not to be an optimal option to train our dataset since the model prefers well-structured and formal texts from platforms such as wikipedia and electronic articles. However, there is an advantage of using Word2Vec is that the neural network models are likely to have no overfitting issue when increasing the number of epochs. In contrast, using keras embedding layer shows a clear overfitting sign as the accuracy of training data quickly gets close to 100% but the accuracy and the loss of validation data has no clear improvement after some epochs.</div>

<div style="text-align: justify">
<br/>
Another interesting point to discuss is the comparison of CNN and RNN. At first, we will briefly explain how these two models work to detect depression. A CNN model learns to recognize patterns and special expressions from the texts and use them as the filter or feature to classify sentences and comments. In contrast, RNN is designed to recognize patterns across time. The model extracts the sequential information and uses a long range semantic dependency for classification. In other words, CNN is good at classification using specific, local features, phrases and words, while RNN is better at learning the comprehension of global/long-range semantics to solve the tasks. Since our dataset comes from a social media platform, many comments are informally written and contain no sequential structure. Some users also use short texts to describe feelings rather than writing a long story to express their conditions. This results in a better performance of CNN model because the classification is primarily based on the extraction of sentiment terms. RNN would be better when the data contains textual documents and requires the detection of sequential information.</div>

# Challenges

<div style="text-align: justify">
The most difficult issue in analyzing data from social media platforms is the use of special characters and emojis. Since our dataset is selected carefully, the majority of comments are written as texts. However, to extend the project to a wider application, we need to address this challenge. This task can be tackled by creating an encoder or an embedding model to convert the special characters, emojis and symbols into numbers or vectors. </div>

# Future Work

There are many studies that can be done to improve this project in the future:
- The first improvement is to address data using special characters, symbols and emojis during the text preprocessing.
- More data which contains both textual writings and informal comments from a variety of social media sources would be collected to make our dataset more diverse and representative.
- To deal with a dataset requiring both sequential information and local features extraction, we can ensemble the CNN and RNN architectures to improve the classification accuracy. A neural network model containing convolutional layers and LSTM layers would be constructed to detect depression. 
- Different word embedding models such as Word2Vec, GloVe, fastText can be implemented to the pre-trained embedding layer to figure out the most optimal word embedding technique.


# Conclusion

In this project, our group presents a method to address early detection of depression using Reddit comments. Two neural network models including CNN and RNN are built to detect subjects with depression. For more insights, we use the keras layer embedding and the pre-trained Word2Vec model to create the input embedding layer of the networks. The results show that keras embedding layer has better performance than the Word2Vec. The reason may be the inefficiency of Word2Vec in accounting informal and ungrammatical data.


According to the model evaluation, CNN model with keras embedding layer has the best performance. The reason that CNN has better classification accuracy than RNN may be that the dataset does not have many textual comments so the extraction of sequential information is not necessary. RNN model is expected to have more outstanding performance when dealing with formal and academic writings such as personal experience stories and mental health reports.

 
For future improvement, more data containing both textual data and informal writings will be collected. An ensemble of CNN and RNN would be constructed to effectively recognize both local and sequential features. Different word embedding models such as GloVe and fastText would be pre-trained to implement to the embedding layer in order to extend the scope of the project.

# References

1. Jacob Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”. In: CoRR abs/1810.04805. 2018.

2. JT Wolohan. “Detecting Linguistic Traces of Depression in Topic-Restricted Text : Attending to Self-Stigmatized Depression with NLP”. In: 2018.

3. Kali Cornn. "Identifying Depression on Social Media". Department of Statistics, Stanford University. 2019.

4. Marcel Trotzek, Sven Koitka, and Christoph M. Friedrich. “Utilizing Neural Networks and Linguistic Metadata for Early Detection of Depression Indications in Text Sequences”. In: CoRR abs/1804.07000. 2018.

5. Soroush Vosoughi, Prashanth Vijayaraghavan, and Deb Roy. “Tweet2Vec: Learning Tweet Embeddings Using Character-level CNN-LSTM Encoder-Decoder”. In: SIGIR. 2016.




```python

```
