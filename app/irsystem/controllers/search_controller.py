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
from nltk.tokenize import word_tokenize
import os.path as path
from nltk.stem import PorterStemmer 
import pickle

project_name = "Gifter.ai"
net_id = "Shreya Subramanian: ss2745, Joy Zhang: jz442, Aparna Calambur: ac987, Ashrita Raman: ar699, Jannie Li: jl2578"

# Load inverted indices
cur_path = pathlib.Path(__file__).parent.absolute().parent.absolute().parent.absolute()
path_review_index =  path.abspath(path.join(cur_path,"datafiles/review_index.pickle"))
path_title_index =  path.abspath(path.join(cur_path,"datafiles/title_index.pickle"))

with open(path_review_index, 'rb') as handle:
    review_index = pickle.load(handle)

with open(path_title_index, 'rb') as handle:
    title_index = pickle.load(handle)

@irsystem.route('/', methods=['GET'])
def search():    
    query = request.args.get('search')
    price = request.args.get('price')
    if not price:
        price = 50
    if not query:
        data = []
        output_message = ''
        asin_list = []
    else:
        output_message = "Relevant products"
        asin_list = boolean_search(query)
        data = create_product_list(asin_list, float(price))
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data, asins=asin_list)

def tokenize_query(text): 
    s = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = re.findall("[a-zA-Z]+", text)
    filtered = [ps.stem(w).lower() for w in words if not w in s]
    return filtered

def create_product_list(asin_list, price):
    cur_path = pathlib.Path(__file__).parent.absolute().parent.absolute().parent.absolute()
    path1 =  path.abspath(path.join(cur_path,"datafiles/merged-metadata-final.csv"))
    metadata_df = pd.read_csv(path1)
    product_list = []
    for asin in asin_list:
        row = metadata_df.loc[metadata_df.asin == asin]
        if(not row.empty):
            title = str(row['title'].iloc[0])
            out_price = '$' + str(row['price'].iloc[0])
            if (float(row['price'].iloc[0]) <= price):
                product_list.append(title + "   --   " + str(out_price))
    return product_list

#TODO use root of word
def boolean_search(query):
    query_tok = tokenize_query(query)
    result = list()
    for token in query_tok:
        token_list = []
        if(token in review_index):
            review_asins = list(set([token_list.append(asin) for (_,asin,_) in review_index[token]]))
        else:
            review_asins = []
        if(token in title_index):
            title_asins = list(set(title_index[token]))
        else:
            title_asins = []
        review_asins = list(set(review_asins).difference(set(title_asins)))
        result += title_asins
        result += review_asins
    return result[:5]