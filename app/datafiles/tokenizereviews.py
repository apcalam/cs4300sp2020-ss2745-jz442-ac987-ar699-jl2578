import csv 
import re
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# python -m nltk.downloader stopwords

def tokenize(text): 
    s = set(stopwords.words('english'))
    words = re.findall("[a-zA-Z]+", text)
    lower = map(lambda x: x.lower(), words)
    filtered = [w for w in lower if not w in s] 
    return filtered

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

reviews_dct = create_review_list("reviews-m03.csv")
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
    return result



query = "cutterpede mom dinosaur moist damp lmao lamp pole"

print(boolean_search(reviews_dct, query))