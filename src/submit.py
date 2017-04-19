#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')

def main():
    parser = argparse.ArgumentParser(prog='submit.py',
            description='FER2013 submitting script.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE','DNN'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}".format(args.model,args.epoch)
    store_path += '_{}'.format(args.idx)

    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    model_path = os.path.join(exp_dir,store_path,'model.h5')

    emotion_classifier = load_model(model_path)
    emotion_classifier.summary()
    te_feats = read_dataset('test',False)
    
    ans = emotion_classifier.predict_classes(te_feats,batch_size=args.batch)
    with open('Strong Baseline','w') as f:
        f.write('id,label\n')
        for idx,a in enumerate(ans):
            f.write('{},{}\n'.format(idx,a))

if __name__ == "__main__":
    main()
