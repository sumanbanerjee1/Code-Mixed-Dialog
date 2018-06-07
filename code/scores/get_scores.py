import pandas as pda
import bleu
import rouge
import subprocess
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_path")
    parser.add_argument("--config_id")
    parser.add_argument("--lang")
    args = parser.parse_args()
    return args

def read_results(path,num):
    with open(path+"/labels"+str(num)+".txt","r") as fp:
        l=fp.readlines()
    with open(path+"/predictions"+str(num)+".txt","r") as fp:
        p=fp.readlines()
    
    return p,l

def exact_match(p,l):
    c=0
    for i1,i in enumerate(l):
        if p[i1]==l[i1]:
            c+=1
    print("Exact Match: ",c/len(l))


def moses_bl_rouge(p,l):
    bl = bleu.moses_multi_bleu(p,l)
    x = rouge.rouge(p,l)
    print('Moses BLEU: %f\nROUGE1-F: %f\nROUGE1-P: %f\nROUGE1-R: %f\nROUGE2-F: %f\nROUGE2-P: %f\nROUGE2-R: %f\nROUGEL-F: %f\nROUGEL-P: %f\nROUGEL-R: %f'%(bl,x['rouge_1/f_score'],x['rouge_1/p_score'],x['rouge_1/r_score'],x['rouge_2/f_score'],
                                                    x['rouge_2/p_score'],x['rouge_2/r_score'],x['rouge_l/f_score'],x['rouge_l/p_score'],x['rouge_l/r_score']))


def pycoco_bl(path,num):
    exit_code = subprocess.Popen("python2 pycocoevalcap/eval.py "+path+"/labels"+str(num)+".txt  "+path+"/predictions"+str(num)+".txt",stdout=subprocess.PIPE,shell=True)
    o, e = exit_code.communicate()
    start_index= str(o).find('Bleu_4')
    bl=float(str(o)[start_index:start_index+13].split(' ')[1])
    print("Pycoco-BLEU: ",bl)

def per_dialogue(p,l,lang):
    if lang=='english':
        message='you are welcome <EOS>'
    elif lang=='hindi':
        message='welcome , alvida <EOS>'
    elif lang=='bengali':
        message='you are welcome <EOS>'
    elif lang=='gujarati':
        message='tamaru swagat chhe <EOS>'
    elif lang=='tamil':
        message='you are welcome <EOS>'
    c=0
    n=0
    
    matches=[]
    ld=''
    pd=''
    for l,p in zip(l,p):
        
        ld=ld+l
        pd=pd+p  
        
        if message in l:
            if ld==pd:
                c+=1
                matches.append([ld,pd])
            n+=1
            ld=''
            pd=''
    
    print("Per Dialog Acc: ",c/n*(100))

if __name__=='__main__':
    args = get_args()
    result_path = args.preds_path
    config_id = args.config_id
    lang = args.lang
    preds,labels = read_results(result_path,config_id)
    exact_match(preds,labels)
    moses_bl_rouge(preds,labels)
    pycoco_bl(result_path,config_id)
    per_dialogue(preds,labels,lang)
    
