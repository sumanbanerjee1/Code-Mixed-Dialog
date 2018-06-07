import re
import os
import argparse
import json
import pandas as pd
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "dialog_dstc_2") 
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=source_dir)

    args = parser.parse_args()
    return args

eos = '<EOS>'
beg = '<beg>'
eok = '<EOK>'
def count_kb(content):
    total_api_calls = len(re.findall('\tapi_call',content))
    failed_api_calls = len(re.findall('api_call no result',content))
    return total_api_calls - failed_api_calls

def locate_kb(content):
    
    kb_start_found = False
    start_index = []
    end_index = []
    
    for turn_no, current_turn in enumerate(content):
        if "R_post_code" in current_turn and not kb_start_found:
            kb_start_found = True
            start_index.append(turn_no)
        if kb_start_found:
            if "<SILENCE>" in current_turn:
                end_index.append(turn_no) 
                kb_start_found = False

    start_index.append(len(content)) #full dialog ease of programming
    return start_index,end_index

def _tokenize(sentence):
    return sentence.split() + [eos] 

def _tokenize_kb(sentence): 
    return sentence.split() + [eok]

def process_kb(given_kb):
    processed_kb = []
    for i in given_kb:
        processed_kb = processed_kb + [re.sub('\d+','',i,1).strip().split(' ')]
    if processed_kb:
        return processed_kb 
    return None

def get_all_dialogs(filename):
    fname=open(filename,'r')
    s = ''
    for i in fname.readlines():
        s = s + i
    all_=s.split('\n\n')
    fname.close()
    return all_[0:-1]

def get_vocab(train_fname,test_fname,dev_fname):
    train=get_all_dialogs(train_fname)
    test=get_all_dialogs(test_fname)
    dev=get_all_dialogs(dev_fname)
    all_dialogs=train+test+dev
    
    words=set([])
    for i in all_dialogs:
        dialog=i.split('\n')
        for utterance_response in dialog:
            utterances=re.sub('\d+','',utterance_response,1).strip().split('\t')
            words.update(utterances[0].split(' '))
            if len(utterances)>1:
                words.update(utterances[1].split(' '))
    w=sorted(words)
    for ind,i in enumerate(w):
        if len(i)>1 and i[len(i)-1]==',':
            w.remove(i)
        if len(i)>1 and '_' not in i and i[0].isupper() and '.' not in i:
            w[ind]=i[0].lower()+i[1:]
    return sorted(set(w))

def get_data(fname): #pass a file object only
    all_dialogues = get_all_dialogs(fname)

    pre_kb = []
    post_kb = []
    kb = []
    utterance = []
    response = []
    count=0
    for dialog_num , single_dialogue in enumerate(all_dialogues):
        history = [[beg]]
        content = single_dialogue.split('\n')
        len_of_dialogue = len(content)
        kb_start_index, kb_end_index = locate_kb(content) #single arrays if no kb found
        kb_occurences = len(kb_start_index) - 1
        
        for i in range(0,kb_start_index[0]):
            utterance_response = content[i].split('\t')
            utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
            
            if len(utterance_response) < 2: #handles api call no result
                history = history + [re.sub('\d+','',content[i],1).strip().split(' ')]
                #print(dialog_num)
                continue
            pre_kb.append(history)
            current_utterance = utterance_response[0].split(' ')
            current_response = utterance_response[1].split(' ')
            kb.append([])
            post_kb.append([])
            utterance.append(current_utterance[0:])
            response.append(current_response[0:])
            history = history + [current_utterance] + [current_response]

        current_pre = history    #entire  pre-kb conversation
        
        for m in range(0,kb_occurences):
        
        #kb processing
        
            current_kb = process_kb(content[kb_start_index[m]:kb_end_index[m]])
            
            
            if kb_occurences > 1: #adds the api call in the history for the second time.
                utterance_response = content[kb_start_index[m]-1].split('\t')
                utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
                current_utterance = utterance_response[0].split(' ')
                if len(utterance_response)>1:
                    current_response = utterance_response[1].split(' ')
                current_pre = current_pre + [current_utterance] + [current_response]
            
            history = []

            for i in range(kb_end_index[m],kb_start_index[m+1]):
                utterance_response = content[i].split('\t')
                utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
                pre_kb.append(current_pre) #pre remains fixed over timesteps
                current_utterance = utterance_response[0].split(' ')
                if len(utterance_response)>1:
                    current_response = utterance_response[1].strip().split(' ')
                kb.append(current_kb)
                post_kb.append(history)
                utterance.append(current_utterance[0:])
                response.append(current_response[0:])
                
                history = history + [current_utterance] + [current_response]

    data = [pre_kb,kb,post_kb,utterance,response]
    return data

def append_context(data):
    
    context=[]
    for i in range(len(data[0])):
        c=[]
        
        c.extend(data[0][i])
        if len(data[1][i])>0:
            c.extend(data[1][i])
        if len(data[2][i])>0:
            c.extend(data[2][i])
        if len(data[3][i])>0:
            c.extend([data[3][i]])
        
        context.append(c)
    return [context,data[4]]


def data_stats(data): 
    
    for ind,d in enumerate(data):
        if ind==0: #pre
            c_len=[]
            for context in d:
                c_len.append(len(context))
        if ind==1: #KB
            utt_len=[]
            for context in d:
                utt_len.append(len(context))
        if ind==2: #post
            resp_len=[]
            for context in d:
                resp_len.append(len(context))

    utterances_len=utt_len+resp_len

    return [max(c_len),max(utterances_len),max(utterances_len)]



def vec(w,words):
    if w in words.index:
        return words.loc[w].as_matrix()
    else:
        return 0

def append_GO(data):
    for i,d in enumerate(data[1]):
        data[1][i]=['<GO>']+d
    
def get_dec_outputs(data):
    
    dec_op=[]
    for i in data[1]:
        temp=i+['<EOS>']
        temp=temp[1:]
        dec_op.append(temp)
    data.append(dec_op)

def prepro(args):
    source_dir = args.source_dir 
    target_dir = args.target_dir
    source_fname = source_dir+ '/dialog-dstc2-'
    target_fname = target_dir+ '/phred-dialog-dstc2-'
    
    train_input = source_fname+ 'trn.txt'
    test_input = source_fname+ 'tst.txt'
    dev_input = source_fname+ 'dev.txt'
     
    train_output = get_data(train_input)
    test_output = get_data(test_input)
    dev_output = get_data(dev_input)

    train_flattened=append_context(train_output)
    test_flattened=append_context(test_output)
    dev_flattened=append_context(dev_output)
    
    vocab=get_vocab(train_input,test_input,dev_input)
    vocab.append('<beg>')
    vocab.append('<GO>')
    vocab.append('<EOS>')
    vocab_dict={k: v+1 for v, k in enumerate(vocab)}
    vocab_dict['<PAD>']=0

    append_GO(train_flattened)
    append_GO(test_flattened)
    append_GO(dev_flattened)

    get_dec_outputs(train_flattened)
    get_dec_outputs(test_flattened)
    get_dec_outputs(dev_flattened)

    train_stats=data_stats(train_flattened)
    test_stats=data_stats(test_flattened)
    dev_stats=data_stats(dev_flattened)

    total_stats=[max(test_stats[0],max(train_stats[0],dev_stats[0])),
                  max(test_stats[1],max(train_stats[1],dev_stats[1])),
                 max(test_stats[2],max(train_stats[2],dev_stats[2]))]

    with open(target_fname+'vocab.json','w+') as fp:
        json.dump(vocab_dict,fp)
    
    with open(target_fname+'train.json','w+') as fp1:
        json.dump(train_flattened,fp1)

    with open(target_fname+'test.json','w+') as fp2:
        json.dump(test_flattened,fp2)
    
    with open(target_fname+'dev.json','w+') as fp3:
        json.dump(dev_flattened,fp3)
    
    with open(target_fname+'stats.json','w+') as fp:
        json.dump(total_stats,fp)

def main():
    args = get_args()
    prepro(args)

if __name__ == "__main__":
    main()