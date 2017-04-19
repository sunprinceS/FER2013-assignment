#import cPickle as pickle
import pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s...' %filename)
        return file

def save_pickle(objext, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
        print ('Saved %s...' %filename)
