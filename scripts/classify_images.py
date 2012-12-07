
from __future__ import division

import os
import os.path
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

from autopath import datadir

images_path = os.path.join(datadir, 'images.p')
classification_path = os.path.join(datadir, 'image_classifications.p')

print classification_path

with open(images_path) as fp:
    images = pickle.load(fp)

with open(classification_path) as fp:
    classifications = pickle.load(fp)

plt.figure(1)

while len(classifications) < len(images):

    plt.clf()
    image, label_locations = images[len(classifications)]
    plt.imshow(image)
    plt.draw()
    plt.show()

    cls = raw_input('classification? ')
    cls = cls.lower().strip()
    if not cls or cls in ('q','quit'):
        break

    if cls == 'save':
        print 'saving'
        os.rename(classification_path, classification_path + '.back')
        with open(classification_path, 'w') as fp:
            pickle.dump(classifications, fp)

    elif cls == 'revert':
        print 'verting'
        with open(classification_path) as fp:
            classifications = pickle.load(fp)

    else:
        classifications.append(cls)
