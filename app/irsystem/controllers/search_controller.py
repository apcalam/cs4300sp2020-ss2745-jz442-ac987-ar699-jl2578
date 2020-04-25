from nltk.stem import PorterStemmer
import pickle
import os.path as path
from nltk.tokenize import word_tokenize
from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
import csv
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import pathlib
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')

project_name = "Gifter.ai"
net_id = "Shreya Subramanian: ss2745, Joy Zhang: jz442, Aparna Calambur: ac987, Ashrita Raman: ar699, Jannie Li: jl2578"


def tokenize_query(text):
    s = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = re.findall("[a-zA-Z]+", text)
    filtered = [ps.stem(w).lower() for w in words if not w in s]
    return filtered


TITLE_WEIGHT = 5
REVIEW_WEIGHT = 1
NUM_RESULTS = 10
GIFT_WORDS = tokenize_query("gift present")

# Load inverted indices
cur_path = pathlib.Path(__file__).parent.absolute(
).parent.absolute().parent.absolute()
path_review_index = path.abspath(
    path.join(cur_path, "datafiles/review_index.pickle"))
path_title_index = path.abspath(
    path.join(cur_path, "datafiles/title_index.pickle"))

with open(path_review_index, 'rb') as handle:
    review_index = pickle.load(handle)

with open(path_title_index, 'rb') as handle:
    title_index = pickle.load(handle)

# Load reviews df and metadata df
cur_path = pathlib.Path(__file__).parent.absolute(
).parent.absolute().parent.absolute()
path1 = path.abspath(
    path.join(cur_path, "datafiles/merged-metadata-final.csv"))
metadata_df = pd.read_csv(path1)
path2 = path.abspath(
    path.join(cur_path, "datafiles/merged-reviews-final.csv"))
reviews_df = pd.read_csv(path2)


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    price = request.args.get('price')
    if not price:
        price = 50
    if not query:
        query = "gift present"
    output_message = "Based on your inputs, here are some gift ideas!"
    asin_list = boolean_search(query)
    data = create_product_list(asin_list, float(price))
    # productid, title, review_summary1, review_summary2, review1, review2, image, price
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, asins=asin_list)


def create_product_list(asin_list, price):
    product_list = []
    for asin in asin_list:
        row = metadata_df.loc[metadata_df.asin == asin]
        reviews = reviews_df.loc[reviews_df.asin == asin]
        if(not row.empty):
            title = str(row['title'].iloc[0])
            review1 = str(reviews['reviewText'].iloc[0])
            review2 = str(reviews['reviewText'].iloc[1])
            summary1 = str(reviews['summary'].iloc[0])
            summary2 = str(reviews['summary'].iloc[1])
            image = str(row['image'].iloc[0])
            out_price = '$' + str(row['price'].iloc[0])

            if (float(row['price'].iloc[0]) <= price):
                product_tuple = (asin, title, summary1,
                                 summary2, review1, review2, image, out_price)
                product_list.append(product_tuple)
    # list of tuples
    return product_list


def add_review_score(reviewerID, asin, count, review_score):
    if asin in review_score:
        if reviewerID in review_score[asin]:
            review_score[asin][reviewerID] += REVIEW_WEIGHT * count
        else:
            review_score[asin][reviewerID] = REVIEW_WEIGHT * count
    else:
        review_score[asin] = {reviewerID: REVIEW_WEIGHT * count}


def add_product_score(asin, score, product_score, review_score, title_score):
    if(asin in product_score):
        product_score[asin] += score
    else:
        product_score[asin] = score
        review_score[asin] = {}
        title_score[asin] = 0


def boolean_search(query):
    # to find the most relevant product
    product_score = {}
    # to find the most relevant reviews for a product
    review_score = {}  # {asin -> {reviewid ->  score}}
    # score how many query words the title of the product has
    title_score = {}

    query_tok = tokenize_query(query)
    for token in query_tok:

        if(token in review_index):
            for (reviewerID, asin, count) in review_index[token]:
                add_review_score(reviewerID, asin, count,
                                 review_score)

        if(token in title_index):
            title_asins = list(set(title_index[token]))
            for asin in title_asins:
                if(asin in title_score):
                    title_score[asin] += TITLE_WEIGHT
                else:
                    title_score[asin] = TITLE_WEIGHT

    for asin in title_score.keys():
        product_score[asin] = title_score[asin]
    for asin in review_score.keys():
        reviews = review_score[asin]
        total_review_score = 0
        for review in reviews.keys():
            total_review_score += reviews[review]
        total_review_score /= (len(reviews) + 1)

        if asin in product_score:
            product_score[asin] += total_review_score
        else:
            product_score[asin] = total_review_score

    product_score_sorted = {k: v for k, v in sorted(
        product_score.items(), key=lambda item: item[1], reverse=True)}

    return list(product_score_sorted.keys())[: NUM_RESULTS]
