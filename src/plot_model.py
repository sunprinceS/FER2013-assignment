#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from termcolor import colored, cprint
import argparse
from keras.utils.visualize_util import plot
from keras.models import load_model

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
def main():
    parser = argparse.ArgumentParser(prog='predict.py',
            description='FER2013 testing script.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE','easy'],
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
    plot(emotion_classifier,to_file='xx.png')


if __name__ == "__main__":
    main()
