import numpy as np

def pad(data,stats):
 
    for ind,d in enumerate(data):
        if ind==0:
            fill=[0 for i in range(stats[2])]
            for context in d:
                for k in context:
                    for l in range(stats[2]-len(k)):
                        k.append(0)
                for i in range(stats[ind]-len(context)):
                        context.append(fill)    
        
        else:
            for context in d:
                for i in range(stats[ind]-len(context)):
                        context.append(0)
        
def get_len(data):
    context_l=[]
    dec_ip_l=[]
    dec_op_l=[]
    sent_l=[]
    for i in data[0]:
        context_l.append(len(i))
        l1=[]
        for j in i:
            l1.append(len(j))
        sent_l.append(l1)
    for i in data[1]:
        dec_ip_l.append(len(i))
    for i in data[2]:
        dec_op_l.append(len(i))
    
    return context_l,dec_ip_l,dec_op_l,sent_l

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
        if ind==0:
            for i1,context in enumerate(d):
                for i2,c in enumerate(context):
                    data[ind][i1][i2]=[w if w==0 else vocab[lower(w)] for w in c]
        else:
            
            for i1,context in enumerate(d):
                data[ind][i1]=[w if w==0 else vocab[lower(w)] for w in context]
            
