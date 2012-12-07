
from __future__ import division

import gzip
import cPickle as pickle
from contextlib import closing
from collections import defaultdict

import numpy as np
import Image
import ImageDraw

from util import create_composite_grid_image

with closing(gzip.open('extract_images2.p.gz')) as fp:
    data = pickle.load(fp)


acc = defaultdict(lambda : [None, defaultdict(lambda : defaultdict(int))])
for label,image,locations in data:
    image[24::, 12::] = 0
    entry = acc[image.tostring()]
    entry[0] = image
    entry[0] = image
    for x,y in locations:
        entry[1][label][x,y] += 1



images = acc.values()
images.sort(key=lambda (image,label_locations): sum(sum(locations.values())
                                               for locations in label_locations.values()
                                               ))


acc = []
for image,label_locations in images[::-1][:9**2]:
    image = Image.fromarray(image)
    image = image.resize(3*np.array(image.size))
    d = ImageDraw.ImageDraw(image)
    d.rectangle((0, 80, image.size[0], image.size[1]), (235,)*3)
    d.text((20, 82),
        ' '.join(['%s%d' % (label, sum(locations.values()))
                           for label,locations in label_locations.iteritems()]),
           (0, 0, 0))
    acc.append(np.asarray(image))
comp = create_composite_grid_image(acc)
comp.save('comp.png')
