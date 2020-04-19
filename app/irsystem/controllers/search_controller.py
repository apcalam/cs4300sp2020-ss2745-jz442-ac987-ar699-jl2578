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
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
import os.path as path


# python -m nltk.downloader stopwords

project_name = "Gifter.ai"
net_id = "Shreya Subramanian: ss2745, Joy Zhang: jz442, Aparna Calambur: ac987, Ashrita Raman: ar699, Jannie Li: jl2578"


@irsystem.route('/', methods=['GET'])
def search():
    

    cur_path = pathlib.Path(__file__).parent.absolute().parent.absolute().parent.absolute()

    #print(cur_path)
    path_1 =  path.abspath(path.join(cur_path,"datafiles/reviews-m03.csv"))
    print(path_1)
    reviews_dct = create_review_list(path_1)
    query = "cutterpede mom dinosaur moist damp lmao lamp pole"
    print(boolean_search(reviews_dct, query))

    query = request.args.get('search')
    price = float(request.args.get('price'))
    if not query:
        data = []
        output_message = ''
    else:
        output_message = "Relevant products"
        data = create_product_list(boolean_search(reviews_dct, query), price)
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)

def tokenize(text): 
    s = set(stopwords.words('english'))
    words = re.findall("[a-zA-Z]+", text)
    lower = map(lambda x: x.lower(), words)
    filtered = [w for w in lower if not w in s] 
    return filtered

def create_product_list(asin_list, price):
    cur_path = pathlib.Path(__file__).parent.absolute().parent.absolute().parent.absolute()
    path1 =  path.abspath(path.join(cur_path,"datafiles/metadata-m03.csv"))
    metadata_df = pd.read_csv(path1)
    product_list = []
    for asin in asin_list:
        row = metadata_df.loc[metadata_df.asin == asin]
        title = str(row['title'].iloc[0])
        out_price = '$' + str(row['price'].iloc[0])
        if (float(row['price'].iloc[0]) <= price):
            product_list.append(title + "   --   " + str(out_price))
    return product_list

def create_review_list(csv_name):
    result = dict()
    i = 0
    with open(csv_name) as csvfile:
        reviewsCSV = csv.reader(csvfile, delimiter=',')
        for row in reviewsCSV:
            if i == 0:
                i += 1
                continue
            review_text = row[2] + " " + row[3]
            tokenized = tokenize(review_text)
            if (row[4] == "False"):
                continue
            if (row[0] in result):
                result[row[0]] = result[row[0]] + tokenized
            else:
                result[row[0]] = tokenized
    return result

#TODO N'T doesn't work
def get_word_count(input_rev_dct):
    result = dict()
    for product in input_rev_dct:
        words = input_rev_dct[product]
        visited = list()
    for word in words:
        if word in visited:
            continue
        visited.append(word)
        if (word in result):
            result[word] += 1
        else:
            result[word] = 1
    return result  

#TODO N'T doesn't work
def find_good_types(input_rev_dct): 
    word_count = get_word_count(input_rev_dct)
    result = list()
    for word in word_count:
        value = word_count[word]
        if (word_count[word]) > 1:
            result.append(word)
    result.sort()
    return result

#TODO N'T doesn't work
def filter_good_types(reviews_dct, good_types):
    for product in reviews_dct:
        review = reviews_dct[product]
        good = [w for w in review if w in good_types] 
        reviews_dct[product] = good
    return reviews_dct


# good_types = find_good_types(reviews_dct) 
# reviews_dct = filter_good_types(reviews_dct, good_types)

#TODO use root of word
def boolean_search(reviews_dct, query):
    query_tok = tokenize(query)
    result = list()
    for product in reviews_dct:
        review = reviews_dct[product]
        review_s = set(review)
        query_s = set(query_tok)
        words = review_s.intersection(query_s)
        if (len(words) > 0):
            result.append(product)
    return result[:4]