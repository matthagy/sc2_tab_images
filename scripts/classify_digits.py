
from __future__ import division

import os
import os.path
import gzip
import cPickle as pickle
from contextlib import closing
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from autopath import datadir

digits_path = os.path.join(datadir, 'digits.npz')
classification_path = os.path.join(datadir, 'digits_classifications.npy')


def main():
    global images
    with closing(np.load(digits_path)) as data:
        images = data['digits']

    global classifications
    load()

    unclassified = classifications == -1
    if not unclassified.sum():
        print 'no unclassified digits'
        return

    global indicies
    indicies = np.where(unclassified)[0]

    global fig, ax
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    global index
    index = 0

    show_current()
    print 'show'
    plt.show()

def load():
    global classifications
    if not os.path.exists(classification_path):
        classifications = -1 * np.ones(len(images), dtype=int)
    else:
        classifications = np.load(classification_path)

def on_key_press(event):
    global classifications
    global index
    assert classifications[indicies[index]] == -1
    if event.key in map(str, range(10)) + ['*']:
        if event.key == '*':
            classifications[indicies[index]] = 11
        else:
            classifications[indicies[index]] = int(event.key)
        index += 1
        if index > len(indicies):
            print 'all image classified'
            exit()
        show_current()

    elif event.key == 'v':
        print 'save'
        np.save(classification_path, classifications)

    elif event.key == 'u':
        if index == 0:
            print 'no undo information'
            return
        print 'undo'
        index -= 1
        classifications[indicies[index]] = -1
        show_current()

    elif event.key == 'l':
        print 'load'
        load()

    elif event.key == 'q':
        print 'quit'
        exit()

    else:
        print 'unkown key', event.key

    return True

def show_current():
    global index, indicies
    assert classifications[indicies[index]] == -1
    print 'showing', index
    image = images[indicies[index]]
    global fig, ax
    fig.clf()
    ax = fig.add_subplot(111)
    ax.imshow(image, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.canvas.draw()

__name__ == '__main__' and main()
