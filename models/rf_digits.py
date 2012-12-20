
from __future__ import division

import os
import os.path
import cPickle as pickle
from contextlib import closing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold

from autopath import datadir
classification_path = os.path.join(datadir, 'digits_classifications.npy')
digits_path = os.path.join(datadir, 'digits.npz')

with closing(np.load(digits_path)) as data:
    images = data['digits']

classifications = np.load(classification_path)

mask = classifications != -1
print mask.sum()
X = images[mask, ...].reshape(mask.sum(), np.prod(images.shape[1::]))
print X.shape
Y = classifications[mask]

acc = []
acc_correct = []
acc_incorrect = []
acc_x_incorrect = []
k_fold = 8
for train_inx, valid_inx in StratifiedKFold(Y, k_fold):
    rf = RandomForestClassifier(n_estimators=100, verbose=3, oob_score=True, compute_importances=True)
    rf.fit(X[train_inx], Y[train_inx])
    Yp = rf.predict(X[valid_inx])
    correct = Yp== Y[valid_inx]
    rf.predict_proba(X[valid_inx])
    p_correct = rf.predict_proba(X[valid_inx]).max(axis=1)
    acc_correct.append(p_correct[correct])
    acc_incorrect.append(p_correct[~correct])

    score = correct.mean()
    print score
    acc.append(score)

    acc_x_incorrect.append([images[mask][valid_inx[~correct]],
                            Y[valid_inx[~correct]],
                            Yp[~correct]])

print 'score', np.mean(acc)

plt.figure(1)
plt.clf()
plt.hist(np.concatenate(acc_correct), color='b', alpha=0.5, log=True, label='Correct')
plt.hist(np.concatenate(acc_incorrect), color='r', alpha=0.5, log=True, label='Incorrect')
plt.legend(loc='upper left', title='Classification')
plt.xlabel('Confidence in Prediction')
plt.ylabel('Count')
plt.yscale('log')
plt.savefig('confidance.png', bbox_inches='tight')
plt.draw()
plt.show()

def flatten(x):
    return np.concatenate([i for i in x if len(i)], axis=0)
ximages, labels, predict = map(flatten, zip(*acc_x_incorrect))

if 0:
    i = 1
    plt.figure(2)
    plt.clf()
    plt.imshow(ximages[i], interpolation='nearest')
    print
    print labels[i], predict[i]
    plt.draw()
    plt.show()

rf = RandomForestClassifier(n_estimators=100, verbose=3, oob_score=True, compute_importances=True)
rf.fit(X, Y)
print 'oob score', rf.oob_score_

with open('rf_digits.p', 'w') as fp:
    pickle.dump(rf, fp)

