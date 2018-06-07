import numpy as np

def pad(data,stats):
 
    for ind,d in enumerate(data):
            for context in d:
                for i in range(stats[ind]-len(context)):
                        context.append(0)
        
def get_len(data):
    context_l=[]
    dec_ip_l=[]
    dec_op_l=[]
    
    for i in data[0]:
        context_l.append(len(i))
    for i in data[1]:
        dec_ip_l.append(len(i))
    for i in data[2]:
        dec_op_l.append(len(i))
    
    return context_l,dec_ip_l,dec_op_l

def lower(word):
    
    if len(word)>1:
        if word[0].isupper() and '_' not in word and '.' not in word:
            return word[0].lower()+word[1:]
        else:
            return word
    else:
        return word


def replace_token_no(data,vocab):
    
    for ind,d in enumerate(data):
        for i1,context in enumerate(d):
            data[ind][i1]=[w if w==0 else vocab[lower(w)] for w in context]
            
