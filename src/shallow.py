#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from utils import *
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')

def main():
    tr_feats,tr_labels = read_shallow_dataset('train')
    dev_feats,dev_labels = read_shallow_dataset('publicTest')
    # clf = SVC()
    clf=KNeighborsClassifier()
    clf = clf.fit(tr_feats,tr_labels)
    print(clf.score(dev_feats,dev_labels))
    # print(accuracy_score(dev_labels,clf.predict(dev_feats)))

if __name__ == "__main__":
    main()
