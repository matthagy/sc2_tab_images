
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


resize_factor = 2

with closing(gzip.open('extract_images2.p.gz')) as fp:
    images = pickle.load(fp)
#images = [(label,image,locations) for label,image,locations in images
#          if label == 'u']
random.shuffle(images)

images = images[:6**2:]
images.sort(key=lambda entry: (entry[1].shape, entry[2][::,1].mean()))

acc = []
for label,image,locations in images:
    image = Image.fromarray(image)
    image = image.resize(resize_factor*np.array(image.size))
    d = ImageDraw.ImageDraw(image)
    for i in xrange(3):
        xb = 36 - 3 - 10 * i
        xa = xb - 10
        ya = 24
        yb = 34
        xa,xb,ya,yb = resize_factor * np.array([xa,xb,ya,yb]) + 0.5*resize_factor
        d.rectangle((xa,ya,xb,yb), None, (255,0,0))

    ys = locations[::, 1]

#    d.text((3, 3), '%s %.1f' % (label, ys.mean()), (255,255,255))
    acc.append(np.asarray(image))

create_composite_grid_image(acc).save('show_numbers_select.png')
