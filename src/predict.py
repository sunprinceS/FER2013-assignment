#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
def acc_2(probs,ans):
    correct = 0
    for a,p in zip(ans,probs):
        if a in p.argsort()[-2:][::-1]:
            correct += 1
    return correct/len(ans)

def main():
    parser = argparse.ArgumentParser(prog='predict.py',
            description='FER2013 testing script.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    model_path = os.path.join(exp_dir,store_path,'model.h5')

    emotion_classifier = load_model(model_path)
    emotion_classifier.summary()
    te_feats,te_labels,_ = read_dataset('privateTest')
    ans,_,_ = get_labels('privateTest')
    probs = emotion_classifier.predict(te_feats,batch_size=args.batch)
    acc = emotion_classifier.evaluate(te_feats,te_labels,batch_size=args.batch)[1]
    print(colored('\nAccuracy on privateTest is {:.6f}'.format(acc),'yellow',attrs=['bold']))
    acc2 = acc_2(probs,ans)
    print(colored('\nAccuracy on privateTest is {:.6f}'.format(acc2),'yellow',attrs=['bold']))



if __name__ == "__main__":
    main()
