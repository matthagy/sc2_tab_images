
from __future__ import division

import os
import gzip
import cPickle as pickle
from contextlib import closing
from collections import defaultdict

import numpy as np
import Image
import ImageDraw

import autopath
from autopath import datadir

digits_path = datadir + '/digits.npz'

with closing(np.load(digits_path)) as data:
    digits0 = data['digits']
    indices0 = data['indices']
    counts0 = data['counts']
    ym0 = data['ym']
    ystd0 = data['ystd']

with closing(gzip.open(datadir + '/attic/extract_images3.p.gz')) as fp:
    images3 = pickle.load(fp)


#with closing(gzip.open(datadir + '/attic/extract_images3.p.gz', 'w')) as fp:
#    pickle.dump(images3, fp)

digits_order = dict((image.tostring(), i)
                    for i,image in enumerate(digits0))



acc = []
for label,image,locations in images3:
    for i in xrange(3):
        xb = 36 - 3 - 10 * i
        xa = xb - 10
        ya = 24
        yb = 34
        digit = image[ya:yb, xa:xb]
        assert digit.size
        acc.append([i, digit, locations[::, 1]])

uniq = defaultdict(lambda : [None, None, [], 0])
for index,digit,ys in acc:
    entry = uniq[index,digit.tostring()]
    entry[0] = digit
    entry[1] = index
    entry[2] = np.concatenate([ys, entry[2]])
    entry[3] += 1

values = uniq.values()
values.sort(key=lambda (image,index,ys,count): digits_order.get(image.tostring(), np.random.randint(1e6, 1e8)))
digits, indices, ys, counts = zip(*values)

digits = np.array(digits)
indices = np.array(indices)
counts = np.array(counts)
ym = np.array([np.mean(ysi) for ysi in ys])
ystd = np.array([np.std(ysi) for ysi in ys])

np.savez(digits_path, digits=digits, indices=indices, counts=counts, ym=ym, ystd=ystd)
