#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
from sklearn.metrics import confusion_matrix
from utils import *

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')

def main():
    parser = argparse.ArgumentParser(prog='find_err_image.py',
            description='FER2013 Find those images that were misclassified.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','NTUEE','DNN'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=20)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    store_path = "{}_epoch{}_{}".format(args.model,args.epoch,args.idx)
    print(colored("Loading model from {}".format(store_path),'yellow',attrs=['bold']))
    model_path = os.path.join(exp_dir,store_path,'model.h5')

    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    te_feats,_,_ = read_dataset('privateTest')
    predictions = emotion_classifier.predict_classes(te_feats,batch_size=args.batch)
    probs = emotion_classifier.predict(te_feats,batch_size=args.batch)
    te_labels,label_counts,categories = get_labels('privateTest')
    conf_mat = confusion_matrix(te_labels,predictions)
    np.save(os.path.join(exp_dir,store_path,'conf_mat'),conf_mat)
    print(conf_mat)
    # with open(os.path.join(exp_dir,store_path,'class_acc'),'w') as f:
        # for label_id, (name, label_count) in enumerate(zip(categories,label_counts)):
            # f.write('{} class accuracy: {:.3f}\n'.format(name,conf_mat[label_id][label_id]/label_count))

    with open(os.path.join(exp_dir,store_path,'err_id'),'w') as f:
        for idx,(ans,pred) in enumerate(zip(te_labels,predictions)):
            if ans != pred:
                f.write('\nidx:{} ans:{} pred:{}\n'.format(idx,categories[ans],categories[pred]))
                prob = probs[idx]
                order = prob.argsort()[::-1]
                f.write(' > '.join([categories[i] for i in order]))


if __name__ == "__main__":
    main()
