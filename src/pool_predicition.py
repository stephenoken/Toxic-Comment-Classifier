import numpy as np
import pickle
from multiprocessing import Pool
from functools import reduce
import time

def predict(clfs, x):
    return [clf.predict_proba(x)[0][1] for clf in clfs]

def pool_predict(classifiers, batch_size, xs):
    batches = np.array_split(xs, round(xs.shape[1] * batch_size))
    pool = Pool(2)
    predicted_batches = pool.map_async(lambda batch: [predict(classifiers, x) for x in batch], batches)
    return reduce(lambda res, cur: res + cur, predicted_batches.get(), [])

if __name__ == '__main__':
    with open('./pickles/naive-classifiers.p', 'rb') as f1, open('./pickles/x-test.p', 'rb') as f2:
        classifiers = pickle.load(f1)
        X_test = pickle.load(f2)
        print("Loaded files")
        y_pred = pool_predict(classifiers, round(X_test[:500].shape[1] * .33), X_test[:500])
        pickle.dump(y_pred, open('./pickles/y-pred.p', 'wb'))
