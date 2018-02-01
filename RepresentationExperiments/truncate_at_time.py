"""
This script performs several experiments to test whether the concatenation
of the rnn states sufficies as a "good representation" of the word, i.e.,
if vectors obtained in such a way can be compared for similarity to conclude
that the corresponding "heared" words are similar.
"""

import pickle
import re
import numpy as np
from sklearn import svm
from sklearn.model_selection import LeaveOneOut

#
# DATA LOADING
#

def extract_key(keyname):
    match = extract_key.re.match(keyname)
    if match == None:
        raise "Cannot match a label in key: %s" % (keyname)
    return match.group(1)

extract_key.re = re.compile(r".*/(\d+)")

data = None
with (open("activations-small.pkl", "rb")) as file:
    data = pickle.load(file)

xs = []
ys = []

for key in data.keys():
    xs.append(data[key])
    ys.append(extract_key(key))


# Truncating each example at time-step 27 and concatenating the features into a
# single 27 x 2048 (=55296) elements vector

MAX_LEN = 27
truncated_xs = np.array([x[:MAX_LEN,:].ravel() for x in xs])
ys = np.array(ys)

# Fitting an SVC onto the built dataset

loo = LeaveOneOut()

results = []
for train_index, test_index in loo.split(truncated_xs):
    print(test_index)

    # Probably we could initialize svc only once outside the loop,
    # but just to be safe we get a new one at each iteration
    svc = svm.SVC()
    svc.fit(truncated_xs[train_index], ys[train_index])
    predicted_ys = svc.predict(truncated_xs[test_index])
    results.append(np.average(predicted_ys == ys[test_index]))

np.average(results)
