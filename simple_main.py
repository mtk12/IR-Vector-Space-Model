import numpy as np
import vsm_model
import json
import operator
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

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

print("Enter Query and 'quit' for exit")
query = input('Enter query:')
alpha = input('Enter alpha value:')

while (True):
    if(query == "quit"):
        break
    q = vsm_model.query_processing(query,all_tokens,idf)
    res={}
    temp=0
    vec1=list(q.values())
    
    for x in range(0,56):
        vec2 = list(tfidf[str(x)].values())
        if cosine_sim(vec1,vec2)>float(alpha):
            temp=cosine_sim(vec1,vec2)
            res[x]=temp
    
    res = dict(sorted(res.items(), key=operator.itemgetter(1),reverse=True))

    print("Retrieved Document")
    keys = list(res.keys())
    values = list(res.values())
    for i in range(len(keys)):
        print("Document:" + str(keys[i]) + "\t|Score " + str(values[i]),end = '\n')
    
    print("\n==================================")
    print("\nEnter Query and 'quit' for exit")
    query = input('Enter query:')
    if(query == "quit"):
        break
    alpha = input('Enter alpha value:')

