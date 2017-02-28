#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import random
from keras.utils.np_utils import to_categorical

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
data_dir = os.path.join(base_dir,'data')

def read_dataset(mode='train'):
    """
    Return:
        # features: (int. list) list
        # labels: int. list
        data_ids: int. list
    """
    # num_data = 0
    datas = []

    with open(os.path.join(data_dir,'{}.csv'.format(mode))) as file:
        for line_id,line in enumerate(file):
            label, feat=line.split(',')
            feat = np.fromstring(feat,dtype=int,sep=' ')
            # print(feat)
            feat = np.reshape(feat,(48,48,1))

            datas.append((feat,int(label),line_id))

    # random.shuffle(datas)  # shuffle outside
    feats,labels,line_ids = zip(*datas)
    feats = np.asarray(feats)
    labels = to_categorical(np.asarray(labels,dtype=np.int32))

    return feats,labels,line_ids


if __name__ == "__main__":
    feats,labels,line_ids = read_dataset('toy')
    print(labels)
    print(line_ids)
    print(feats.shape)
