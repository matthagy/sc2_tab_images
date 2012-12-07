
from __future__ import division

import os
import os.path
import gzip
import cPickle as pickle
from contextlib import closing
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import autopath
from autopath import datadir

with closing(gzip.open(os.path.join(datadir, 'attic/extract_images2.p.gz'))) as fp:
    data2 = pickle.load(fp)

acc = defaultdict(lambda : [None, defaultdict(lambda : defaultdict(int))])
for label,image,locations in data2:
    image[24::, 12::] = 0
    entry = acc[image.tostring()]
    entry[0] = image
    entry[0] = image
    for x,y in locations:
        entry[1][label][x,y] += 1

images = acc.values()
images.sort(key=lambda (image,label_locations) :
            (sum(sum(locations.values())
                 for locations in label_locations.values()),
             image.tostring()),
            reverse=True)

print len(images)

with closing(gzip.open(os.path.join(datadir, 'attic/extract_images3.p.gz'))) as fp:
    data3 = pickle.load(fp)

for label,image,locations in data3:
    image[24::, 12::] = 0
    entry = acc[image.tostring()]
    entry[0] = image
    entry[0] = image
    for x,y in locations:
        entry[1][label][x,y] += 1

seen = set(image.tostring() for image,_ in images)
for image_string,(image,label_locations) in acc.iteritems():
    if image_string in seen:
        continue
    images.append([image, label_locations])

print len(images)

images = [(image, dict((label, dict(locations))
                      for label,locations in label_locations.iteritems()))
          for image,label_locations in images]

with open(os.path.join(datadir, 'images.p'), 'w') as fp:
    pickle.dump(images, fp, pickle.HIGHEST_PROTOCOL)
