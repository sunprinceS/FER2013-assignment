#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import model
from keras.callbacks import TensorBoard
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')

def main():
    parser = argparse.ArgumentParser(prog='train.py',
            description='FER2013 training script.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE','DNN'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    args = parser.parse_args()
    
    dir_cnt = 0
    log_path = "{}_epoch{}".format(args.model,str(args.epoch))
    log_path += '_'
    store_path = os.path.join(exp_dir,log_path+str(dir_cnt))
    while dir_cnt < 150:
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
            break
        else:
            dir_cnt += 1
            store_path = os.path.join(exp_dir,log_path+str(dir_cnt))

    emotion_classifier = model.build_model(args.model)
    tr_feats,tr_labels,_ = read_dataset('train')
    dev_feats,dev_labels,_ = read_dataset('publicTest')
    # emotion_classifier.fit(x = tr_feats,y = tr_labels,
            # batch_size=args.batch,nb_epoch=args.epoch,validation_data=(dev_feats,dev_labels),
            # callbacks=[TensorBoard(log_dir=store_path,histogram_freq=1,write_graph=True)])
    history = model.History()
    emotion_classifier.fit(x = tr_feats,y = tr_labels,
            batch_size=args.batch,nb_epoch=args.epoch,validation_data=(dev_feats,dev_labels),
            callbacks=[history])
    dump_history(store_path,history)
    # print(history.tr_losses)
    # print(history.tr_accs)
    # print(history.val_losses)
    # print(history.val_accs)
    emotion_classifier.save(os.path.join(store_path,'model.h5'))

if __name__ == "__main__":
    main()
