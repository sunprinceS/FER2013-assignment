from cnn import *
from utils import *
import os
import numpy as np
import argparse
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(batch_size, num_epoch, pretrain, train_pixels, train_labels, val_pixels, val_labels, model_name=None):
    if pretrain == False:
        model = build_model()
    else:
        model = load_model(model_name)

    num_instances = len(train_labels)
    iter_per_epoch = (num_instances / batch_size) + 1
    batch_cutoff = [0]
    for i in range(iter_per_epoch - 1):
        batch_cutoff.append(batch_size * (i+1))
    batch_cutoff.append(num_instances)

    total_start_t = time.time()
    for e in range(num_epoch):
        rand_idxs = np.random.permutation(num_instances)
        print ('#######')
        print ('Epoch ' + str(e+1))
        print ('#######')
        start_t = time.time()

        for i in range(iter_per_epoch):
            if i % 20 == 0:
                print ('Iteration ' + str(i+1))
            X_batch = []
            Y_batch = []
            for n in range(batch_cutoff[i], batch_cutoff[i+1]):
                X_batch.append(train_pixels[rand_idxs[n]])
                Y_batch.append(np.zeros((7, ), dtype=np.float))
                X_batch[-1] = np.fromstring(X_batch[-1], dtype=float, sep=' ').reshape((48, 48, 1))
                Y_batch[-1][int(train_labels[rand_idxs[n]])] = 1.
            model.train_on_batch(np.asarray(X_batch), np.asarray(Y_batch))

        loss_and_metrics = model.evaluate(val_pixels, val_labels, batch_size)
        print ('\nloss & metrics:')
        print (loss_and_metrics)

        print ('Elapsed time in epoch ' + str(e+1) + ': ' + str(time.time() - start_t))
        model.save('model/model-%d.h5' %(e+1))
    print ('Elapsed time in total: ' + str(time.time() - total_start_t))

def main():
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--pretrain', type=bool, default=False)
    args = parser.parse_args()

    train_pixels = load_pickle('../fer2013/train_pixels.pkl')
    train_labels = load_pickle('../fer2013/train_labels.pkl')
    print ('# of training instances: ' + str(len(train_labels)))
    val_pixels = load_pickle('../fer2013/publicTest_pixels.pkl')
    val_labels = load_pickle('../fer2013/publicTest_labels.pkl')
    print ('# of val instances: ' + str(len(val_labels)))

    for i in range(len(val_labels)):
        val_pixels[i] = np.fromstring(val_pixels[i], dtype=float, sep=' ').reshape((48, 48, 1))
        onehot = np.zeros((7, ), dtype=np.float)
        onehot[int(val_labels[i])] = 1.
        val_labels[i] = onehot

    model_name = 'model/model-1'
    train(args.batch, args.epoch, args.pretrain,
          train_pixels, train_labels,
          np.asarray(val_pixels), np.asarray(val_labels),
          model_name)

if __name__=='__main__':
    main()
