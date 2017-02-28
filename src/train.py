#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from utils import *
import model

def main():
    parser = argparse.ArgumentParser(prog='train.py',
            description='FER2013 training script.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    args = parser.parse_args()

    emotion_classifier = model.build_model(args.model)
    tr_feats,tr_labels,_ = read_dataset('train')
    dev_feats,dev_labels,_ = read_dataset('publicTest')
    emotion_classifier.fit(x = tr_feats,y = tr_labels,
            batch_size=args.batch,nb_epoch=args.epoch,validation_data=(dev_feats,dev_labels))

if __name__ == "__main__":
    main()
