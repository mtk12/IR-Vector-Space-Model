import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import re
from collections import defaultdict
from nltk.stem import PorterStemmer

#stemming initialization
ps = PorterStemmer() 

def preprocess(file_content,stop_words):
    #cleaning documents
        file_content = re.sub('  ', ' ', file_content)
        file_content = re.sub(r"won't", "will not", file_content)
        file_content = re.sub(r"can\'t", "can not", file_content)
        file_content = re.sub(r"n\'t", " not", file_content)
        file_content = re.sub(r"\'re", " are", file_content)
        file_content = re.sub(r"\'s", " is", file_content)
        file_content = re.sub(r"\'d", " would", file_content)
        file_content = re.sub(r"\'ll", " will", file_content)
        file_content = re.sub(r"\'t", " not", file_content)
        file_content = re.sub(r"\'ve", " have", file_content)
        file_content = re.sub(r"\'m", " am", file_content)
        file_content = re.sub(r'[^\w\s]',' ', file_content)
        file_content = file_content.lower()
        file_content = [words if words not in stop_words else '' for words in file_content.split(' ')]
        doc = []
        doc = list(filter(None, file_content)) 
        stemmed = []
        
        #stemming
        for i in doc:
            stemmed.append(ps.stem(i))
            
        return stemmed
        
#creates inverted index
def inverted_index(stop_words):
    
    tokens = []

    for i in range(0,56):
        doc_no = i
        with open ("Trump Speechs\Trump Speechs\speech_" + str(doc_no) + ".txt",'r') as file:
            next(file)
            file_content = file.read().replace('\n',' ')
        
        file_token = preprocess(file_content,stop_words)
            
        #creating posting list
        for x in file_token:
            tokens.append(x)
            
#        #removing duplicates
        tokens = list(set(tokens))
        tokens = sorted(tokens)
    return tokens


#creates inverted index
def document_tokenization(stop_words):
    
    tokens = {}

    for i in range(0,56):
        doc_no = i
        with open ("Trump Speechs\Trump Speechs\speech_" + str(doc_no) + ".txt",'r') as file:
            next(file)
            file_content = file.read().replace('\n',' ')
        
        file_token = preprocess(file_content,stop_words)
        
        file_token = list(set(file_token))
        file_token = sorted(file_token)
        
        key = i
        tokens[key] = file_token

    return tokens