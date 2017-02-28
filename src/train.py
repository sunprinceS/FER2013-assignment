#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.optimizers import SGD,Adam,Adagrad
from utils import *
import model

def main():
    emotion_classifier = model.build_model()
    # opt = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
    opt = Adam(lr=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
    tr_feats,tr_labels,_ = read_dataset('train')
    dev_feats,dev_labels,_ = read_dataset('publicTest')
    emotion_classifier.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    emotion_classifier.fit(x = tr_feats,y = tr_labels,batch_size=64,nb_epoch=5,validation_data=(dev_feats,dev_labels))

    f,l,_ = read_dataset('publicTest')
    print(emotion_classifier.evaluate(f,l,batch_size=64))
    f,l,_ = read_dataset('privateTest')
    print(emotion_classifier.evaluate(f,l,batch_size=64))


if __name__ == "__main__":
    main()
