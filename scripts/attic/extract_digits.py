
from __future__ import division

import os
import gzip
import cPickle as pickle
from contextlib import closing
from collections import defaultdict

import numpy as np
import Image
import ImageDraw

from util import create_composite_grid_image

import random


with closing(gzip.open('extract_images2.p.gz')) as fp:
    images = pickle.load(fp)


acc = []
for label,image,locations in images:
    for i in xrange(3):
        xb = 36 - 3 - 10 * i
        xa = xb - 10
        ya = 24
        yb = 34
        digit = image[ya:yb, xa:xb]
        acc.append([i, digit, locations[::, 1]])

uniq = defaultdict(lambda : [None, None, [], 0])
for index,digit,ys in acc:
    entry = uniq[index,digit.tostring()]
    entry[0] = digit
    entry[1] = index
    entry[2] = np.concatenate([ys, entry[2]])
    entry[3] += 1


values = uniq.values()
random.shuffle(values)
digits, indices, ys, counts = zip(*values)

digits = np.array(digits)
indices = np.array(indices)
counts = np.array(counts)
ym = np.array([np.mean(ysi) for ysi in ys])
ystd = np.array([np.std(ysi) for ysi in ys])

np.savez('digits.npz', digits=digits, indices=indices, counts=counts, ym=ym, ystd=ystd)
