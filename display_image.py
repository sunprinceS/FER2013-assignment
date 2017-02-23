import numpy as np
import os
from matplotlib import pyplot as plt
import sys

file  = sys.argv[1]
# line_num = int(sys.argv[2])

fp = open(file)
for i, line in enumerate(fp):
    data = line.split(",")[1]
    data  = np.fromstring(data, dtype=int, sep=' ')
    data = np.reshape(data, (48, 48))
    plt.imshow(data, cmap = 'gray')
    fig = plt.gcf()
    plt.draw()
    fig.savefig('image/{}.png'.format(i),dpi=100)

fp.close()


# plt.show()
