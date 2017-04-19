import numpy as np
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    # x *= 255
    # x = x.transpose((1, 2, 0))
    # x = np.clip(x, 0, 255).astype('uint8')
    return x
