import csv 
import re
import nltk 
from nltk.corpus import stopwords
from pathlib import Path
import pathlib
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
import pickle

# Inverted index review text + summary= tokens -> (reviewerID, asin, count)
# Inverted index title + categories = tokens -> (asin)

reviews_df = pd.read_csv("merged-reviews-final.csv")
metadata_df = pd.read_csv("merged-metadata-final.csv")
metadata_df = metadata_df.drop_duplicates(subset = ['asin'])

def tokenize_data(text): 
    s = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = re.findall("[a-zA-Z]+", text)
    filtered = [ps.stem(w).lower() for w in words if not w in s]
    return filtered

def create_reviews_inverted_index():
    inverted_index = {}
    reviews_df['description'] = reviews_df['asin'].map(metadata_df.set_index('asin')['description'])
    reviews_df['summary_and_review'] = reviews_df['reviewText'] + " " + \
                reviews_df['summary'] +  " " + reviews_df['description']
    for index, row in reviews_df.iterrows():
        if(bool(row['verified'])):
            tokenized = tokenize_data(str(row['summary_and_review']))
            tokenized_set = set(tokenized)
            reviewerID = row['reviewerID']
            asin = row['asin']
            for token in tokenized_set:
                count = tokenized.count(token)
                if token in inverted_index:
                    inverted_index[token].append((reviewerID, asin, count))
                else:
                    inverted_index[token] = [(reviewerID, asin, count)]

    return inverted_index

def create_title_inverted_index():
    inverted_index = {}
    metadata_df['title_and_cat'] = metadata_df['title'].astype(str) + metadata_df['category'].astype(str)
    for index, row in metadata_df.iterrows():
        tokenized = tokenize_data(str(row['title_and_cat']))
        tokenized_set = set(tokenized)
        asin = row['asin']
        for token in tokenized_set:
            if token in inverted_index:
                inverted_index[token].append(asin)
            else:
                inverted_index[token] = [asin]
    return inverted_index

def export_inverted_index():
    index_titles = create_title_inverted_index()
    index_reviews = create_reviews_inverted_index()

    with open('title_index.pickle', 'wb') as handle:
        pickle.dump(index_titles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('review_index.pickle', 'wb') as handle:
        pickle.dump(index_reviews, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
export_inverted_index()
