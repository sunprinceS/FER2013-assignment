import cPickle as pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s...' %filename)
        return file
