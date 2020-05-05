import re
import math
import spacy
spacy_nlp = spacy.load('en_core_web_sm')
    
'''
Loading Stopwords
'''
stop_words = []
with open ("Stopword-List.txt",'r') as file:
    s=file.read().replace('\n',' ')
stop_words = s.split()

'''
Preprocessing Function takes file string or query string and 
remove stopwords, special chracters, and lemmatize it
and return tokens
'''
def preprocess(file_content):
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
        file_content = re.sub("[^A-Za-z0-9]+"," ",file_content)
        file_content = re.sub('  ', ' ', file_content)
        file_content = file_content.lower()
        file_content = [words if words not in stop_words else '' for words in file_content.split(' ')]
        file_content = list(filter(None, file_content)) 
        file_content = ' '.join(file_content)
        doci = spacy_nlp(file_content)
        tokens = [token.lemma_ for token in doci]
        
        return tokens

'''
Tokenize documents and return all unique tokens present in the documents
'''
def tokenization():
    
    tokens = []

    for i in range(0,56):
        doc_no = i
        with open ("Trump Speechs\Trump Speechs\speech_" + str(doc_no) + ".txt",'r') as file:
            next(file)
            file_content = file.read().replace('\n',' ')
        
        file_token = preprocess(file_content)
            
        #creating posting list
        for x in file_token:
            tokens.append(x)
            
    #removing duplicates
    tokens = list(set(tokens))
    tokens = sorted(tokens)
    
    return tokens

'''
Retrieving all tokens in each document and storing in dictionary
'''
def document_tokenization():
    
    tokens = {}

    for i in range(0,56):
        doc_no = i
        with open ("Trump Speechs\Trump Speechs\speech_" + str(doc_no) + ".txt",'r') as file:
            next(file)
            file_content = file.read().replace('\n',' ')
        
        file_token = preprocess(file_content)
        
        file_token = sorted(file_token)
        
        key = i
        tokens[key] = file_token

    return tokens

'''
Calculating term frequency of the tokens in each documents
'''
def term_frequency(all_tokens,docu_tokens):
    
    tf = {}
    for i in range(0,56):
        tf[i] = dict.fromkeys(all_tokens,0)
        for j in docu_tokens[i]:
            tf[i][j] += 1
    
    return tf

'''
Calculating Inverse Document frequency
'''
def inverse_doument_frequency(tf,all_tokens):
    
    df = {} 
    for i in all_tokens:
        df[i] = 0
        for j in range(0,56):
            if( tf[j][i] > 0 ):
                df[i] += 1
    
    idf = {}
    for i in all_tokens:
        idf[i] = math.log(df[i]/56)
        
    return idf
    
'''
Calculating TFIDF
'''    
def tfidf(tf,idf,all_tokens):
    
    tfidf = {}
    
    for i in range(0,56):
        tfidf[i] = {}
        for j in all_tokens:
            tfidf[i][j] = tf[i][j] * idf[j]
    
    return tfidf

'''
Preprocess query and Making Query vector
'''
def query_processing(query,token,idf):
    
    q = preprocess(query)
    qv = dict.fromkeys(token,0)

    for i in q:
        if (i in token):
            qv[i] += 1
        else:
            print(i + " does not exists in dictionary!")

     
    for i in qv:
        qv[i] = qv[i]*idf[i]
    
    return qv
    