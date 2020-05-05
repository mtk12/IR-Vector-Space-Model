import numpy as np
import vsm_model
import time
from flask import Flask, render_template,request
import operator 
import json

app = Flask(__name__)

'''
Creating Tokens,Document Tokens, Term Frequency, IDF, and TFIDF
'''

#all_tokens = vsm_model.tokenization()
#docu_tokens = vsm_model.document_tokenization()
#tf = vsm_model.term_frequency(all_tokens,docu_tokens)
#idf = vsm_model.inverse_doument_frequency(tf,all_tokens)
#tfidf = vsm_model.tfidf(tf,idf,all_tokens)

'''
Saving Tokens,Document Tokens, Term Frequency, IDF, and TFIDF
'''

#with open('all_tokens.json', 'w') as fp:
#    json.dump(all_tokens, fp)
#
#with open('docu_tokens.json', 'w') as fp:
#    json.dump(docu_tokens, fp)
#
#with open('term_frequency.json', 'w') as fp:
#    json.dump(tf, fp)
#
#with open('idf.json', 'w') as fp:
#    json.dump(idf, fp)
#    
#with open('tfidf.json', 'w') as fp:
#    json.dump(tfidf, fp)


'''
Loading Tokens,Document Tokens, Term Frequency, IDF, and TFIDF
'''

with open('all_tokens.json') as f:
  for token in f:
        all_tokens = json.loads(token)

with open('docu_tokens.json') as f:
  docu_tokens = json.load(f)

with open('term_frequency.json') as f:
  tf = json.load(f)

with open('idf.json') as f:
  idf = json.load(f)
  
with open('tfidf.json') as f:
  tfidf = json.load(f)


'''
Cosine Similarity calculation
Funtion takes two arguments Query and Document vectors and Returns cosine similarity between them
'''
def cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

#Returning Relevant document retrieved
def documents_ret(a):
    
    docu = {}
    for i in range(0,56):
        doc_no = i
        with open ("Trump Speechs\Trump Speechs\speech_" + str(doc_no) + ".txt",'r') as file:
            next(file)
            s=file.read().replace('\n',' ')
            
        key = 'speech_' + str(doc_no)
        
        docu.setdefault(key,[])
        docu[key].append(s)
        
    documents = {}
    if(a):
        keys = list(a.keys())
        values = list(a.values())
        for i in range(len(keys)):
            speech = "speech_" + str(keys[i])
            documents.setdefault(speech,[])
            documents[speech].append(docu.get(speech))
            documents[speech].append(values[i])
    else:
        documents = {}
    
    return documents
        
        
'''
Default/Home Page is loaded
'''
@app.route('/')
def dictionary():
    return render_template('home.html')

'''
Query Processing Function take query and display result
'''
@app.route("/query", methods=['POST'])
def upload():
    #query processing start time
    start = time.time()
    query = request.form['query']
    alpha = request.form['alpha']
    print(alpha)
    if alpha == '':
        alpha = 0.0005
    q = vsm_model.query_processing(query,all_tokens,idf)
    res={}
    temp = 0
    vec1 = list(q.values())
    
    for x in range(0,56):
        vec2 = list(tfidf[str(x)].values())
        sim = cosine_similarity(vec1,vec2)
        if sim > float(alpha):
            temp = sim
            res[x] = temp
    
    res = dict(sorted(res.items(), key=operator.itemgetter(1),reverse=True))
    documents = documents_ret(res)
    
    print(res)
    end = time.time()
    times = end - start
    
    return render_template('dictionary.html',dictionary = documents, num_docs= len(documents), time = str(times) + " " + "seconds",quer=query)

if __name__ == '__main__':
    app.run()
