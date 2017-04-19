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
    if i == 59 or i==60 :
        plt.imshow(data, cmap = 'gray')
        fig = plt.gcf()
        plt.draw()
        fig.savefig('../image/{}.png'.format(i),dpi=1000)
        break

fp.close()


# plt.show()
