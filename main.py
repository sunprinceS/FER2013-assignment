from train import *
from utils import *
import numpy as np
import time

pixels = load_pickle('../fer2013/train_pixels.pkl')
labels = load_pickle('../fer2013/train_labels.pkl')
print ('# of instances: ' + str(len(pixels)))
# print (len(labels))
# pixels = load_pickle('../fer2013/publicTest_pixels.pkl')
# labels = load_pickle('../fer2013/publicTest_labels.pkl')
# print (len(pixels))
# print (len(labels))
# pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
# labels = load_pickle('../fer2013/privateTest_labels.pkl')
# print (len(pixels))
# print (len(labels))

batch_size = 32
model = build_model()
compile_model(model)

num_instances = len(labels)
iter_per_epoch = (num_instances / batch_size) + 1
batch_cutoff = [0]
for i in range(iter_per_epoch - 1):
    batch_cutoff.append(batch_size * (i+1))
batch_cutoff.append(num_instances)

num_epoch = 1
total_start_t = time.time()
for e in range(num_epoch):
    rand_idxs = np.random.permutation(num_instances)
    print ('#######')
    print ('Epoch ' + str(e+1))
    print ('#######')
    start_t = time.time()
    for i in range(iter_per_epoch):
        print ('Iteration ' + str(i+1))
        X_batch = []
        Y_batch = []
        for n in range(batch_cutoff[i], batch_cutoff[i+1]):
            X_batch.append(pixels[rand_idxs[n]])
            Y_batch.append(np.zeros((7, ), dtype=np.float))
            X_batch[-1] = np.fromstring(X_batch[-1], dtype=float, sep=' ').reshape((48, 48, 1))
            Y_batch[-1][int(labels[rand_idxs[n]])] = 1.
        train_on_batch(model, np.asarray(X_batch), np.asarray(Y_batch), batch_size)
    print ('Elapsed time in epoch %d: %d' %(e+1, (time.time - start_t)))
print ('Elapsed time in total: %d' %(time.time - total_start_t))
