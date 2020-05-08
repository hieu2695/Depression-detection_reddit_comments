# Depression-detection_reddit_comments

## Introduction

Major depressive disorder or depression is a mental illness that negatively impacts an individual’s mood and activities leads to a persistent feeling of sadness and loss of interest. It is estimated that depression accounts for 4.3% of the global burden of disease and 1 million suicides every year. Despite these detrimental effects, the symptoms of depression can only be diagnosed after two weeks. Therefore, a method to detect early depression is a vital improvement in addressing this mental disease.

In recent years, social media platforms such as Facebook, Twitter and Reddit have become an important part in mental health support. It is claimed that social media platforms have been increasingly used by individuals to connect with others, share experiences and feelings. Analyzing the sentiment in comments and posts on such platforms would provide insights about an individual's feeling and health status, which helps to identify mental illness including depression.

## Objective 

This project aims to identify depressed subjects utilizing comments scrapped from Reddit. With respect to the excellence of CNN in previous research, we will build a CNN architecture to predict comments labeled as depression. We also expect to see how a more popular neural network model in text classification will perform comparing to the CNN. There will be two neural network architectures: the base-line model is CNN, and the second model is the recurrent neural network (RNN) which is well-known for text classification and sequence classification tasks. Moreover, word embeddings will be implemented in our models. We will vectorize the comments and create an embedding layer using the Tokenizer class in the Keras API, and pre-train a Word2Vec model.

## Dataset

The data was collected following the reasoning of JT Wolohan in his paper “Detecting Linguistic Traces of Depression in Topic-Restricted Text: Attending to Self-Stigmatized Depression with NLP” (2018). The data was scraped from two subreddits: /r/depression and /r/AskReddit using Python Reddit API Wrapper (PRAW). Comments in /r/depression are labeled as depression and comments in /r/AskReddit are labeled as non-depression. For future work, more data would be collected to make the dataset more diverse and representative.
