#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import model
from keras.callbacks import TensorBoard
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')

SEMI_PARTIAL = 14000
def main():
    parser = argparse.ArgumentParser(prog='train.py',
            description='FER2013 training script.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE','DNN'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--partial',type=int,metavar='<partial_num>',default=28709)
    parser.add_argument('--semi-freq',type=int,metavar='<semi_freq>',default=5)
    parser.add_argument('--semi-mode',type=str,metavar='<semi_mode>',default='confident',choices=['all','prob','confident'])
    args = parser.parse_args()
    
    dir_cnt = 0
    log_path = "semi_{}_epoch{}".format(args.model,str(args.epoch))
    if args.partial != 28709:
        log_path += "_partial{}_semi-freq{}_semi-mode{}".format(args.partial,args.semi_freq,args.semi_mode)
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
    unlabel_feats = tr_feats[args.partial:]
    label_feats = tr_feats[:args.partial]
    labels = tr_labels[:args.partial]
    dev_feats,dev_labels,_ = read_dataset('publicTest')
    semi_labels = None
    semi_feats = None

    for e in range(args.epoch//args.semi_freq):
        print('Epoch {}'.format(e*args.semi_freq))
        history = model.History()

        if semi_labels is None:
            emotion_classifier.fit(x=label_feats,y=labels,
                    batch_size=args.batch,nb_epoch=args.semi_freq,
                    validation_data=(dev_feats,dev_labels), callbacks=[history])
        else:
            emotion_classifier.fit(x=np.vstack((label_feats,semi_feats)),
                y=np.vstack((labels,semi_labels)),batch_size=args.batch,nb_epoch=args.semi_freq,
                validation_data=(dev_feats,dev_labels), callbacks=[history])

        if e*args.semi_freq >= 0:
            print('Self-training')
            if args.semi_mode == 'all':
                semi_feats = unlabel_feats
                semi_labels = to_categorical(emotion_classifier.predict_classes(unlabel_feats,batch_size=args.batch))
            if args.semi_mode == 'confident':
                proba = emotion_classifier.predict(unlabel_feats,batch_size=args.batch)
                cls = emotion_classifier.predict_classes(unlabel_feats,batch_size=args.batch)
                semi_feats = unlabel_feats[np.max(proba,axis=1) > 0.4]
                semi_labels = to_categorical(cls[np.max(proba,axis=1) > 0.4])
                unlabel_feats = unlabel_feats[np.max(proba,axis=1)<=0.4]
                print('**',semi_labels.shape,'**')

                if len(unlabel_feats) < args.batch:
                    break

        dump_history(store_path,history)

    emotion_classifier.save(os.path.join(store_path,'model.h5'))

if __name__ == "__main__":
    main()
